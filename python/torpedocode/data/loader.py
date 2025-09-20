"""Utilities for ingesting limit order book event streams."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Iterator, Protocol, List, Tuple, Optional
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..config import DataConfig
from ..features.lob import build_lob_feature_matrix
from ..features.topological import TopologicalFeatureGenerator
from ..utils.scaler import SplitSafeStandardScaler
from .preprocess import label_instability
from ..config import TopologyConfig


class EventStream(Protocol):
    """Protocol describing an iterator over event-level dictionaries."""

    def __iter__(self) -> Iterator[Dict]: ...


@dataclass(slots=True)
class MarketDataLoader:
    """Load raw LOB data feeds and normalise them to a canonical schema."""

    config: DataConfig

    def load_events(self, instrument: str, *, row_slice: slice | None = None) -> pd.DataFrame:
        """Return a canonical event table for a specific instrument.

        Parameters
        ----------
        instrument:
            Instrument identifier used when caching.
        row_slice:
            Optional slice restricting which rows are read from the cached parquet file.
            This enables memory-friendly windowed loading without materialising the full
            dataset in RAM. Negative bounds are interpreted relative to the end of the
            file, similar to Python slicing semantics.
        """

        path = (self.config.cache_root / instrument).with_suffix(".parquet")
        if not path.exists():
            raise FileNotFoundError(
                f"Expected cached parquet file at {path}. Generate caches via LOBPreprocessor first."
            )

        if row_slice is None:
            frame = pd.read_parquet(path)
        else:
            frame = self._read_parquet_slice(path, row_slice)
        expected_columns = {"timestamp", "event_type", "size"}
        missing = expected_columns.difference(frame.columns)
        if missing:
            raise ValueError(f"Missing required columns {missing} in cached frame {path}")

        frame = frame.sort_values("timestamp").reset_index(drop=True)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        return frame

    def _read_parquet_slice(self, path: Path, row_slice: slice) -> pd.DataFrame:
        """Read a subset of rows from a parquet file without loading the entire table."""

        try:
            import pyarrow as pa  # type: ignore
            import pyarrow.parquet as pq  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "pyarrow is required to read slices from cached parquet files; install pyarrow"
            ) from exc

        pf = pq.ParquetFile(path)
        total = pf.metadata.num_rows
        start = row_slice.start if row_slice.start is not None else 0
        stop = row_slice.stop if row_slice.stop is not None else total
        step = row_slice.step
        if step not in (None, 1):
            raise ValueError("row_slice.step is not supported; expected None or 1")

        if start < 0:
            start = max(total + start, 0)
        if stop < 0:
            stop = max(total + stop, 0)

        start = max(0, min(start, total))
        stop = max(0, min(stop, total))

        if stop <= start or total == 0:
            # Return empty frame with the correct schema
            try:
                empty = pf.schema.empty_table()
            except AttributeError:  # pragma: no cover - older pyarrow
                arrays = []
                for field in pf.schema:
                    arrays.append(pa.array([], type=field.type))
                empty = pa.Table.from_arrays(arrays, names=pf.schema.names)
            return empty.to_pandas()

        tables = []
        current = 0
        for rg_idx in range(pf.num_row_groups):
            rg_meta = pf.metadata.row_group(rg_idx)
            rg_rows = rg_meta.num_rows
            rg_start = current
            rg_stop = current + rg_rows
            current = rg_stop
            if rg_stop <= start:
                continue
            if rg_start >= stop:
                break
            table = pf.read_row_group(rg_idx)
            s = max(0, start - rg_start)
            e = min(rg_rows, stop - rg_start)
            if s > 0 or e < rg_rows:
                table = table.slice(s, e - s)
            tables.append(table)

        if not tables:
            return pd.DataFrame(columns=pf.schema.names)

        table = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
        return table.to_pandas()

    def row_count(self, instrument: str) -> int:
        """Return the number of cached rows for an instrument without loading them."""

        path = (self.config.cache_root / instrument).with_suffix(".parquet")
        if not path.exists():
            raise FileNotFoundError(
                f"Expected cached parquet file at {path}. Generate caches via LOBPreprocessor first."
            )
        try:
            import pyarrow.parquet as pq  # type: ignore
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "pyarrow is required to inspect cached parquet metadata; install pyarrow"
            ) from exc

        pf = pq.ParquetFile(path)
        return int(pf.metadata.num_rows)


@dataclass(slots=True)
class LOBDatasetBuilder:
    """Turn canonical event streams into model-ready tensors."""

    config: DataConfig

    def build_sequence(
        self, instrument: str, *, row_slice: slice | None = None
    ) -> Dict[str, np.ndarray]:
        """Construct raw feature arrays and labels for one instrument."""

        mdl = MarketDataLoader(self.config)
        df = mdl.load_events(instrument, row_slice=row_slice)
        count_windows = None
        if getattr(self.config, "count_windows_s", None) is not None:
            import pandas as _pd

            count_windows = tuple(
                _pd.to_timedelta(int(s), unit="s") for s in self.config.count_windows_s
            )
        ewma_halflives = None
        if getattr(self.config, "ewma_halflives_s", None) is not None:
            ewma_halflives = tuple(float(x) for x in self.config.ewma_halflives_s)
        base, aux = build_lob_feature_matrix(
            df,
            levels=self.config.levels,
            count_windows=(count_windows if count_windows is not None else ()),
            ewma_halflives=ewma_halflives,
        )
        feats: List[np.ndarray] = [base]
        names: List[str] = [
            *(f"bsize_{i+1}" for i in range(self.config.levels)),
            *(f"asize_{i+1}" for i in range(self.config.levels)),
        ]

        def add_block(block: np.ndarray, prefix: str):
            nonlocal feats, names
            if block.ndim == 1:
                feats.append(block.reshape(-1, 1))
                names.append(prefix)
            elif block.ndim == 2 and block.shape[1] > 0:
                feats.append(block)
                names.extend([f"{prefix}_{i}" for i in range(block.shape[1])])

        add_block(aux.get("imbalance@k", np.zeros((len(df), 0), dtype=np.float32)), "imb_k")
        add_block(aux.get("cum_depth_b", np.zeros((len(df), 0), dtype=np.float32)), "cdb")
        add_block(aux.get("cum_depth_a", np.zeros((len(df), 0), dtype=np.float32)), "cda")
        add_block(aux.get("ret", np.zeros((len(df), 0), dtype=np.float32)), "ret")
        add_block(aux.get("evt_counts", np.zeros((len(df), 0), dtype=np.float32)), "cnt")
        add_block(aux.get("delta_t", np.zeros((len(df),), dtype=np.float32)), "dt")
        add_block(aux.get("spreads", np.zeros((len(df),), dtype=np.float32)), "spread")
        add_block(aux.get("tod_sin", np.zeros((len(df),), dtype=np.float32)), "tod_sin")
        add_block(aux.get("tod_cos", np.zeros((len(df),), dtype=np.float32)), "tod_cos")
        add_block(aux.get("tod_progress", np.zeros((len(df),), dtype=np.float32)), "tod_progress")
        add_block(aux.get("dow_sin", np.zeros((len(df),), dtype=np.float32)), "dow_sin")
        add_block(aux.get("dow_cos", np.zeros((len(df),), dtype=np.float32)), "dow_cos")
        qab = aux.get("queue_age_b", None)
        qaa = aux.get("queue_age_a", None)
        if isinstance(qab, np.ndarray) and qab.ndim == 2 and qab.shape[1] > 0:
            add_block(qab, "qage_b")
        if isinstance(qaa, np.ndarray) and qaa.ndim == 2 and qaa.shape[1] > 0:
            add_block(qaa, "qage_a")
        add_block(aux.get("queue_age_b1", np.zeros((len(df),), dtype=np.float32)), "qage_b1")
        add_block(aux.get("queue_age_a1", np.zeros((len(df),), dtype=np.float32)), "qage_a1")

        X_raw = (
            np.concatenate(feats, axis=1).astype(np.float32)
            if feats
            else np.zeros((len(df), 0), dtype=np.float32)
        )

        label_cols = [
            c
            for c in df.columns
            if c.startswith("instability_s_") or c.startswith("instability_e_")
        ]
        labels: Dict[str, np.ndarray] = {}
        if not label_cols:
            labs = label_instability(
                mid=pd.Series(aux.get("mid", np.zeros((len(df),), dtype=np.float32))),
                timestamps=df["timestamp"],
                horizons_s=self.config.forecast_horizons_s,
                horizons_events=self.config.forecast_horizons_events,
                threshold_eta=float(getattr(self.config, "instability_threshold_eta", 0.0)),
            )
            for k, s in labs.items():
                labels[k] = s.to_numpy(dtype=np.int64)
        else:
            for c in label_cols:
                labels[c] = df[c].to_numpy(dtype=np.int64)

        et_ids = None
        if "event_type" in df.columns:
            et_series = df["event_type"].astype(str)
            if (
                bool(getattr(self.config, "expand_event_types_by_level", False))
                and "level" in df.columns
            ):
                lvl = pd.to_numeric(df["level"], errors="coerce").fillna(-1).astype(int)
                mask = et_series.isin(["LO+", "LO-", "CX+", "CX-"]) & (lvl >= 1)
                et_series = et_series.where(~mask, et_series + "@" + lvl.astype(str))
            vocab = {t: i for i, t in enumerate(sorted(set(et_series)))}
            et_ids = et_series.map(vocab).to_numpy(dtype=np.int64)
        ts = pd.to_datetime(df["timestamp"], utc=True).astype("int64").to_numpy()
        if ts.size == 0:
            delta_t = np.zeros((0,), dtype=np.float32)
        else:
            delta_t = (np.diff(ts, prepend=ts[0]) / 1e9).astype(np.float32)
        sizes = (
            pd.to_numeric(df.get("size", 0.0), errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
        )

        return {
            "timestamps": pd.to_datetime(df["timestamp"], utc=True).to_numpy(),
            "features_raw": X_raw,
            "feature_names": np.array(names, dtype=object),
            "labels": labels,
            "event_type_ids": et_ids,
            "delta_t": delta_t,
            "sizes": sizes,
        }

    def build_walkforward_splits(
        self,
        instrument: str,
        *,
        label_key: str,
        topology: Optional[TopologyConfig] = None,
        topo_stride: int = 1,
        folds: int = 3,
        artifact_dir: Optional[Path] = None,
        row_slice: slice | None = None,
    ) -> List[Tuple[Dict, Dict, Dict, SplitSafeStandardScaler]]:
        """Return a list of (train,val,test,scaler) splits for k-step walk-forward.

        Uses cumulative train windows and equal-sized val/test windows across folds.
        """
        rec = self.build_sequence(instrument, row_slice=row_slice)
        T = len(rec["timestamps"])
        k = max(1, int(folds))
        # Partition the last 40% into k segments; training grows cumulatively from 0 to each segment start
        base = int(0.6 * T)
        tail = T - base
        seg = max(1, tail // k)
        out: List[Tuple[Dict, Dict, Dict, SplitSafeStandardScaler]] = []
        for i in range(k):
            v0 = base + i * seg
            t1 = base + (i + 1) * seg if i < k - 1 else T
            mid = v0 + max(1, (t1 - v0) // 2)
            train_idx = slice(0, v0)
            val_idx = slice(v0, mid)
            test_idx = slice(mid, t1)
            scaler = SplitSafeStandardScaler()
            X = rec["features_raw"]
            feature_names = rec["feature_names"].tolist()
            X_train = scaler.fit_transform(X[train_idx], feature_names=feature_names)
            X_val = scaler.transform(X[val_idx])
            X_test = scaler.transform(X[test_idx])
            cfg = topology or TopologyConfig()
            topo_gen = TopologicalFeatureGenerator(cfg)

            def build(split: slice) -> Dict:
                ts_split = rec["timestamps"][split]
                X_scaled = scaler.transform(X[split])
                use_raw = False
                if cfg.complex_type == "cubical" and getattr(
                    cfg, "use_raw_liquidity_surface", False
                ):
                    use_raw = True
                if cfg.complex_type == "vietoris_rips" and getattr(cfg, "use_raw_for_vr", False):
                    use_raw = True
                X_for_tda = rec["features_raw"][split] if use_raw else X_scaled
                Z_split = topo_gen.rolling_transform(ts_split, X_for_tda, stride=topo_stride)
                y_split = rec["labels"][label_key][split]
                d = {
                    "features": X_scaled.astype(np.float32),
                    "topology": Z_split.astype(np.float32),
                    "labels": y_split.astype(np.int64),
                    "delta_t": rec["delta_t"][split].astype(np.float32),
                }
                if rec["event_type_ids"] is not None:
                    d["event_type_ids"] = rec["event_type_ids"][split].astype(np.int64)
                if rec["sizes"] is not None:
                    d["sizes"] = rec["sizes"][split].astype(np.float32)
                return d

            train = build(train_idx)
            val = build(val_idx)
            test = build(test_idx)
            if artifact_dir is not None:
                try:
                    ad = Path(artifact_dir) / f"fold_{i+1}"
                    ad.mkdir(parents=True, exist_ok=True)
                    from .. import __version__ as _version
                    import json

                    with open(ad / "split_indices.json", "w") as f:
                        json.dump(
                            {
                                "train_idx": list(range(0, v0)),
                                "val_idx": list(range(v0, mid)),
                                "test_idx": list(range(mid, t1)),
                            },
                            f,
                            indent=2,
                        )
                except Exception:
                    pass
            out.append((train, val, test, scaler))
        return out

    def build_splits(
        self,
        instrument: str,
        *,
        label_key: str,
        topology: Optional[TopologyConfig] = None,
        topo_stride: int = 1,
        artifact_dir: Optional[Path] = None,
        row_slice: slice | None = None,
    ) -> Tuple[Dict, Dict, Dict, SplitSafeStandardScaler]:
        """Create walk-forward train/val/test sets with split-safe scaling and PH features."""

        rec = self.build_sequence(instrument, row_slice=row_slice)
        T = len(rec["timestamps"])
        t0 = int(0.6 * T)
        v0 = int(0.8 * T)

        scaler = SplitSafeStandardScaler()
        X = rec["features_raw"]
        X_train = scaler.fit_transform(X[:t0], feature_names=rec["feature_names"].tolist())
        X_val = scaler.transform(X[t0:v0])
        X_test = scaler.transform(X[v0:])

        cfg = topology or TopologyConfig()
        topo_gen = TopologicalFeatureGenerator(cfg)
        if (
            cfg.persistence_representation == "image"
            and bool(getattr(cfg, "image_auto_range", False))
            and (
                getattr(cfg, "image_birth_range", None) is None
                or getattr(cfg, "image_pers_range", None) is None
            )
        ):
            _ = topo_gen.rolling_transform(rec["timestamps"][:t0], X_train, stride=topo_stride)
            b_rng = getattr(topo_gen, "_active_birth_range", None)
            p_rng = getattr(topo_gen, "_active_pers_range", None)
            if b_rng is not None and p_rng is not None:
                base = asdict(cfg)
                base.update(
                    {
                        "image_auto_range": False,
                        "image_birth_range": b_rng,
                        "image_pers_range": p_rng,
                    }
                )
                cfg = TopologyConfig(**base)
                topo_gen = TopologicalFeatureGenerator(cfg)

        def build_split(s: slice) -> Dict:
            ts_split = rec["timestamps"][s]
            X_scaled = scaler.transform(X[s])
            use_raw = False
            if cfg.complex_type == "cubical" and getattr(cfg, "use_raw_liquidity_surface", False):
                use_raw = True
            if cfg.complex_type == "vietoris_rips" and getattr(cfg, "use_raw_for_vr", False):
                use_raw = True
            X_for_tda = rec["features_raw"][s] if use_raw else X_scaled
            Z_split = topo_gen.rolling_transform(ts_split, X_for_tda, stride=topo_stride)
            y_split = rec["labels"][label_key][s]
            out = {
                "features": X_scaled.astype(np.float32),
                "topology": Z_split.astype(np.float32),
                "labels": y_split.astype(np.int64),
                "delta_t": rec["delta_t"][s].astype(np.float32),
            }
            if rec["event_type_ids"] is not None:
                out["event_type_ids"] = rec["event_type_ids"][s].astype(np.int64)
            if rec["sizes"] is not None:
                out["sizes"] = rec["sizes"][s].astype(np.float32)
            return out

        train = build_split(slice(0, t0))
        val = build_split(slice(t0, v0))
        test = build_split(slice(v0, T))

        if artifact_dir is not None:
            ad = Path(artifact_dir)
            ad.mkdir(parents=True, exist_ok=True)
            scaler.save_schema(str(ad / "scaler_schema.json"))
            from .. import __version__ as _version

            schema = {
                "version": str(_version),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "instrument": instrument,
                "levels": int(self.config.levels),
                "feature_names": rec["feature_names"].tolist(),
                "topology": (asdict(cfg) if cfg is not None else asdict(TopologyConfig())),
                "label_key": label_key,
            }
            if (
                cfg.persistence_representation == "image"
                and getattr(cfg, "image_birth_range", None) is not None
                and getattr(cfg, "image_pers_range", None) is not None
            ):
                schema["topology_ranges"] = {
                    "birth": [float(cfg.image_birth_range[0]), float(cfg.image_birth_range[1])],
                    "persistence": [float(cfg.image_pers_range[0]), float(cfg.image_pers_range[1])],
                }
            import json

            with open(ad / "feature_schema.json", "w") as f:
                json.dump(schema, f, indent=2)
            try:
                splits_art = {
                    "train_idx": list(range(0, t0)),
                    "val_idx": list(range(t0, v0)),
                    "test_idx": list(range(v0, T)),
                }
                with open(ad / "split_indices.json", "w") as f:
                    json.dump(splits_art, f, indent=2)
            except Exception:
                pass

        return train, val, test, scaler

    def build_synthetic_batch(
        self,
        batch_size: int = 2,
        T: int = 16,
        num_event_types: int = 3,
        feature_dim: int | None = None,
        topo_dim: int = 4,
    ) -> Dict[str, np.ndarray]:
        """Create a minimal synthetic batch to exercise TPP loss on CPU.

        Returns a dict with keys: features [B,T,F], topology [B,T,Z], instability_labels [B,T],
        event_type_ids [B,T], delta_t [B,T], sizes [B,T].
        """

        rng = np.random.default_rng(0)
        F = feature_dim if feature_dim is not None else self.config.levels * 2
        features = rng.normal(size=(batch_size, T, F)).astype(np.float32)
        topology = rng.normal(size=(batch_size, T, topo_dim)).astype(np.float32)
        event_type_ids = rng.integers(0, num_event_types, size=(batch_size, T), dtype=np.int64)
        delta_t = rng.exponential(scale=0.5, size=(batch_size, T)).astype(np.float32)
        sizes = np.exp(rng.normal(0.0, 0.5, size=(batch_size, T))).astype(np.float32)
        logits = rng.normal(0.0, 1.0, size=(batch_size, T)).astype(np.float32)
        probs = 1 / (1 + np.exp(-logits))
        instability_labels = (rng.uniform(size=(batch_size, T)) < probs).astype(np.float32)

        return {
            "features": features,
            "topology": topology,
            "event_type_ids": event_type_ids,
            "delta_t": delta_t,
            "sizes": sizes,
            "instability_labels": instability_labels,
        }


__all__ = ["MarketDataLoader", "LOBDatasetBuilder", "EventStream"]
