"""Preprocessing pipeline for raw market data feeds."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from ..config import DataConfig
from .itch import ITCHParseConfig, parse_itch_minimal
from .ouch import OUCHParseConfig, parse_ouch_minimal
from .lobster import LOBSTERParseConfig, parse_lobster_pair
from .native import parse_itch_native, parse_ouch_native
from .preprocess import (
    HarmoniseConfig,
    harmonise_ndjson,
    adjust_corporate_actions,
    round_prices_to_tick,
    label_instability,
    _load_corporate_actions_csv,
)


@dataclass(slots=True)
class LOBPreprocessor:
    """Canonicalise raw venue feeds into aligned event streams."""

    config: DataConfig

    def harmonise(
        self,
        source_files: Iterable[Path],
        instrument: str,
        *,
        tick_size: Optional[float] = None,
        price_scale: Optional[float] = None,
    ) -> pd.DataFrame:
        """Convert raw message logs into the canonical schema with UTC normalization."""

        frames: List[pd.DataFrame] = []
        files: List[Path] = []
        for p in source_files:
            p = Path(p)
            if p.is_dir():
                files.extend(
                    sorted(
                        [
                            *p.glob("*.ndjson"),
                            *p.glob("*.jsonl"),
                            *p.glob("*.itch"),
                            *p.glob("*.itc"),
                            *p.glob("*.bin"),
                            *p.glob("*.ouch"),
                            *p.glob("*.ouc"),
                            *p.glob("*.csv"),
                        ]
                    )
                )
            else:
                files.append(p)
        used = set()
        for idx, file in enumerate(files):
            if file.suffix.lower() in {".ndjson", ".jsonl"}:
                cfg = HarmoniseConfig(
                    time_zone=self.config.time_zone,
                    drop_auctions=self.config.drop_auctions,
                    session_time_zone=getattr(self.config, "session_time_zone", "America/New_York"),
                    tick_size=tick_size if self.config.normalise_prices_to_ticks else None,
                    price_scale=price_scale,
                    symbol=instrument,
                )
                try:
                    frames.append(harmonise_ndjson(file, cfg=cfg))
                    used.add(idx)
                except Exception:
                    continue
            elif file.suffix.lower() in {".itch", ".itc", ".bin"}:
                try:
                    df = parse_itch_native(
                        file,
                        tick_size=tick_size,
                        symbol=instrument,
                        spec=self.config.itch_spec,
                    )
                    if df is None:
                        df = parse_itch_minimal(
                            file, cfg=ITCHParseConfig(tick_size=tick_size, symbol=instrument)
                        )
                    frames.append(df)
                    used.add(idx)
                except Exception:
                    continue
            elif file.suffix.lower() in {".ouch", ".ouc"}:
                try:
                    df = parse_ouch_native(
                        file,
                        tick_size=tick_size,
                        symbol=instrument,
                        spec=self.config.ouch_spec,
                    )
                    if df is None:
                        df = parse_ouch_minimal(
                            file, cfg=OUCHParseConfig(tick_size=tick_size, symbol=instrument)
                        )
                    frames.append(df)
                    used.add(idx)
                except Exception:
                    continue
            elif file.suffix.lower() == ".csv" and (
                "orderbook" in file.stem.lower() or "message" in file.stem.lower()
            ):
                if idx in used:
                    continue
                stem = file.stem.lower()
                is_msg = "message" in stem
                for j, other in enumerate(files):
                    if j == idx or j in used:
                        continue
                    if other.suffix.lower() != ".csv":
                        continue
                    if is_msg and "orderbook" in other.stem.lower() and other.parent == file.parent:
                        try:
                            df = parse_lobster_pair(
                                file,
                                other,
                                cfg=LOBSTERParseConfig(tick_size=tick_size, symbol=instrument),
                            )
                            frames.append(df)
                            used.update({idx, j})
                        except Exception:
                            pass
                        break
                    if (
                        (not is_msg)
                        and "message" in other.stem.lower()
                        and other.parent == file.parent
                    ):
                        try:
                            df = parse_lobster_pair(
                                other,
                                file,
                                cfg=LOBSTERParseConfig(tick_size=tick_size, symbol=instrument),
                            )
                            frames.append(df)
                            used.update({idx, j})
                        except Exception:
                            pass
                        break
            else:
                continue

        if not frames:
            return pd.DataFrame(
                columns=[
                    "timestamp",
                    "event_type",
                    "price",
                    "size",
                    "level",
                    "side",
                    "symbol",
                    "venue",
                ]
            )

        frame = pd.concat(frames, axis=0, ignore_index=True)
        frame = frame.sort_values("timestamp").reset_index(drop=True)
        if getattr(self.config, "corporate_actions_csv", None):
            try:
                ca_df = _load_corporate_actions_csv(
                    Path(self.config.corporate_actions_csv),
                    date_col=self.config.corporate_actions_date_col,
                    symbol_col=self.config.corporate_actions_symbol_col,
                    factor_col=self.config.corporate_actions_factor_col,
                )
                frame = adjust_corporate_actions(
                    frame, ca_df, mode=str(getattr(self.config, "corporate_actions_mode", "divide"))
                )
                if (
                    bool(getattr(self.config, "normalise_prices_to_ticks", True))
                    and tick_size is not None
                    and float(tick_size) > 0
                ):
                    frame = round_prices_to_tick(frame, float(tick_size))
            except Exception:
                pass
        return frame

    @staticmethod
    def discover_lobster_pairs(root: Path) -> List[Tuple[Path, Path]]:
        """Pair LOBSTER message/orderbook CSVs in a directory.

        Rules:
        - Look for files containing 'message' and 'orderbook' with matching stems up to that token.
        - Return list of (message_csv, orderbook_csv) pairs.
        """
        root = Path(root)
        csvs = sorted(root.glob("*.csv"))
        msgs = [p for p in csvs if "message" in p.stem.lower()]
        obs = [p for p in csvs if "orderbook" in p.stem.lower()]
        pairs: List[Tuple[Path, Path]] = []
        used = set()
        for m in msgs:
            key = m.stem.lower().split("message")[0]
            for ob in obs:
                if ob in used:
                    continue
                if ob.stem.lower().startswith(key):
                    pairs.append((m, ob))
                    used.add(ob)
                    break
        return pairs

    def cache(self, frame: pd.DataFrame, instrument: str) -> Path:
        """Persist the harmonised event stream for faster experiments."""

        destination = (self.config.cache_root / instrument).with_suffix(".parquet")
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(destination, index=False)
        return destination

    def add_instability_labels(
        self,
        frame: pd.DataFrame,
        *,
        eta: float = 0.0,
    ) -> pd.DataFrame:
        """Append clock- and event-horizon labels to a canonical event frame."""

        df = frame.copy()
        if {"bid_price_1", "ask_price_1"}.issubset(df.columns):
            best_bid = pd.to_numeric(df["bid_price_1"], errors="coerce")
            best_ask = pd.to_numeric(df["ask_price_1"], errors="coerce")
            mid = (best_bid + best_ask) / 2.0
        else:
            mid = pd.to_numeric(df.get("price", 0.0), errors="coerce")
        labs = label_instability(
            mid,
            df["timestamp"],
            horizons_s=self.config.forecast_horizons_s,
            horizons_events=self.config.forecast_horizons_events,
            threshold_eta=eta,
        )
        for name, ser in labs.items():
            df[name] = ser
        return df


__all__ = ["LOBPreprocessor"]
