"""Topological data analysis features.

Provides a rolling-window persistent homology workflow with optional backends
(`ripser`, `persim`, `gudhi`). If dependencies are missing, returns zeros with
correct shape so that training code remains runnable. Vectorisations include
landscapes and images as specified in :class:`TopologyConfig`.
"""

from __future__ import annotations

from dataclasses import dataclass
import warnings
from typing import Optional, Tuple, Iterable, List

import numpy as np

from ..config import TopologyConfig

try:  # pragma: no cover - optional
    import torpedocode_tda as _tda_native  # type: ignore
except Exception:  # pragma: no cover
    _tda_native = None


@dataclass(slots=True)
class TopologicalFeatureGenerator:
    """Compute persistent homology descriptors from rolling windows."""

    config: TopologyConfig
    _active_birth_range: Optional[Tuple[float, float]] = None
    _active_pers_range: Optional[Tuple[float, float]] = None

    def transform(self, tensor: np.ndarray) -> np.ndarray:
        """Convert liquidity windows into vectorised persistence summaries."""

        num_samples = tensor.shape[0]
        embedding_dim = self._embedding_dim()
        if num_samples == 0:
            return np.zeros((0, embedding_dim), dtype=np.float32)

        reps = []
        for i in range(num_samples):
            slab = tensor[i]
            try:
                diag = self._compute_diagram(slab)
                rep = self._vectorise(diag)
            except Exception:
                if self._strict():
                    raise
                rep = np.zeros((embedding_dim,), dtype=np.float32)
            reps.append(rep)
        return np.stack(reps, axis=0).astype(np.float32)

    def rolling_transform(
        self,
        timestamps: np.ndarray,
        series: np.ndarray,
        *,
        window_sizes_s: Optional[Iterable[int]] = None,
        stride: int = 1,
    ) -> np.ndarray:
        """Apply PH to rolling causal windows over a time series of vectors."""
        if series.size == 0:
            return np.zeros((0, self._embedding_dim()), dtype=np.float32)

        self._active_birth_range = getattr(self.config, "image_birth_range", None)
        self._active_pers_range = getattr(self.config, "image_pers_range", None)

        ts = np.asarray(timestamps)
        if np.issubdtype(ts.dtype, np.datetime64):
            ts_ns = ts.astype("datetime64[ns]").astype("int64")
        else:
            try:
                import pandas as pd

                td = pd.to_datetime(ts, utc=True)
                try:
                    ts_ns = td.view("int64")  # type: ignore[attr-defined]
                except Exception:
                    ts_ns = td.astype("int64").to_numpy()
            except Exception:
                ts_ns = ts.astype("datetime64[ns]").astype("int64")
        T, F = series.shape
        wlist = (
            list(window_sizes_s) if window_sizes_s is not None else list(self.config.window_sizes_s)
        )
        reps_all = []
        for w in wlist:
            w_ns = int(w) * 1_000_000_000
            reps = np.zeros((T, self._embedding_dim()), dtype=np.float32)
            use_image = self.config.persistence_representation == "image"
            need_auto = bool(getattr(self.config, "image_auto_range", False)) and (
                getattr(self.config, "image_birth_range", None) is None
                or getattr(self.config, "image_pers_range", None) is None
            )
            births: List[float] = []
            pers: List[float] = []
            if use_image and need_auto:
                for i in range(0, T, max(1, int(stride))):
                    left = ts_ns[i] - w_ns
                    j0 = int(np.searchsorted(ts_ns, left, side="right"))
                    slab = series[j0 : i + 1]
                    try:
                        if self.config.complex_type == "cubical" and getattr(
                            self.config, "use_liquidity_surface", True
                        ):
                            field = self._liquidity_surface_field(slab)
                            diags = self._compute_diagram(field)
                        else:
                            diags = self._compute_diagram(slab)
                        for D in diags[: self.config.max_homology_dimension + 1]:
                            if isinstance(D, np.ndarray) and D.size:
                                births.extend((D[:, 0]).tolist())
                                pers.extend((D[:, 1]).tolist())
                    except Exception:
                        if self._strict():
                            raise
                        continue
                if births and pers:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", message="invalid value encountered in subtract")
                        bmin, bmax = np.quantile(births, [0.01, 0.99]).astype(float)
                        pmin, pmax = np.quantile(pers, [0.01, 0.99]).astype(float)
                    self._active_birth_range = (float(bmin), float(bmax))
                    self._active_pers_range = (float(pmin), float(pmax))
                else:
                    self._active_birth_range = None
                    self._active_pers_range = None
            else:
                self._active_birth_range = getattr(self.config, "image_birth_range", None)
                self._active_pers_range = getattr(self.config, "image_pers_range", None)

            for i in range(0, T, max(1, int(stride))):
                left = ts_ns[i] - w_ns
                j0 = int(np.searchsorted(ts_ns, left, side="right"))
                slab = series[j0 : i + 1]
                try:
                    if self.config.complex_type == "cubical" and getattr(
                        self.config, "use_liquidity_surface", True
                    ):
                        field = self._liquidity_surface_field(slab)
                        diags = self._compute_diagram(field)
                    else:
                        diags = self._compute_diagram(slab)
                    reps[i] = self._vectorise(diags)
                except Exception:
                    if self._strict():
                        raise
                    reps[i] = 0.0
            reps_all.append(reps)
        return np.concatenate(reps_all, axis=1).astype(np.float32)

    def _embedding_dim(self) -> int:
        if self.config.persistence_representation == "landscape":
            return self.config.landscape_levels * (self.config.max_homology_dimension + 1)
        if self.config.persistence_representation == "image":
            return (self.config.max_homology_dimension + 1) * (self.config.image_resolution**2)
        raise ValueError(f"Unsupported representation {self.config.persistence_representation}")

    def _compute_diagram(self, slab: np.ndarray) -> List[np.ndarray]:
        """Return diagrams for H0..Hmax as a list [H0, H1, ...]."""
        if self.config.complex_type == "vietoris_rips":
            X = np.atleast_2d(slab)
            return self._ripser_diagrams(X)
        elif self.config.complex_type == "cubical":
            field = np.array(slab, dtype=float)
            return self._cubical_diagrams(field)
        else:
            raise ValueError(f"Unsupported complex_type {self.config.complex_type}")

    def _ripser_diagrams(self, X: np.ndarray) -> List[np.ndarray]:
        if X.shape[0] < 3:
            return [np.zeros((0, 2)) for _ in range(self.config.max_homology_dimension + 1)]
        if bool(getattr(self.config, "vr_zscore", True)):
            Xc = X.copy().astype(float)
            mu = np.nanmean(Xc, axis=0, keepdims=True)
            sd = np.nanstd(Xc, axis=0, keepdims=True)
            sd[sd == 0] = 1.0
            X = (Xc - mu) / sd
        try:
            from ripser import ripser
        except Exception:
            if self._strict():
                raise RuntimeError("ripser not available and TORPEDOCODE_STRICT_TDA=1")
            return [np.zeros((0, 2)) for _ in range(self.config.max_homology_dimension + 1)]
        ripser_kwargs = {"maxdim": self.config.max_homology_dimension, "metric": "euclidean"}
        if self.config.vr_auto_epsilon and X.shape[0] >= 2:
            if getattr(self.config, "vr_epsilon_rule", "mst_quantile") == "largest_cc":
                eps = self._epsilon_for_lcc(X, threshold=self.config.vr_lcc_threshold)
            else:
                eps = self._estimate_vr_epsilon(X, q=self.config.vr_connectivity_quantile)
            if np.isfinite(eps) and eps > 0:
                ripser_kwargs["thresh"] = float(eps)

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The input point cloud has more columns than rows; did you mean to transpose?",
            )
            warnings.filterwarnings(
                "ignore",
                message="The input matrix is square, but the distance_matrix flag is off.",
            )
            res = ripser(X, **ripser_kwargs)
        diagrams = res.get("dgms", [])
        out: List[np.ndarray] = []
        for d in range(self.config.max_homology_dimension + 1):
            if d < len(diagrams):
                D = diagrams[d]
                bp = np.stack([D[:, 0], D[:, 1] - D[:, 0]], axis=1)
                out.append(bp)
            else:
                out.append(np.zeros((0, 2)))
        return out

    def _estimate_vr_epsilon(self, X: np.ndarray, q: float = 0.99) -> float:
        """Estimate epsilon_max via MST edges so that ~q fraction of nodes are connected."""
        if _tda_native is not None:
            try:
                return float(_tda_native.estimate_vr_epsilon(np.asarray(X, dtype=float), float(q)))
            except Exception:
                pass
        n = int(X.shape[0])
        if n <= 2:
            d = float(np.linalg.norm(X[1] - X[0])) if n == 2 else 1.0
            return d
        D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
        np.fill_diagonal(D, np.inf)
        in_tree = np.zeros(n, dtype=bool)
        in_tree[0] = True
        min_edge = D[0].copy()
        edges = []
        for _ in range(n - 1):
            j = int(np.argmin(min_edge))
            w = float(min_edge[j])
            if not np.isfinite(w):
                break
            edges.append(w)
            in_tree[j] = True
            min_edge = np.minimum(min_edge, D[j])
            min_edge[in_tree] = min_edge[in_tree]
        if not edges:
            finite = D[np.isfinite(D)]
            return float(np.quantile(finite, 0.95)) if finite.size else 1.0
        edges_sorted = sorted(edges)
        k = max(1, min(len(edges_sorted), int(np.ceil(q * (n - 1)))))
        return float(edges_sorted[k - 1])

    def _epsilon_for_lcc(self, X: np.ndarray, threshold: float = 0.99) -> float:
        """Smallest epsilon where largest connected component has >= threshold of points.

        Uses a coarse grid of distance quantiles to avoid O(N^3). Suitable for small windows.
        """
        if _tda_native is not None:
            try:
                return float(_tda_native.epsilon_for_lcc(np.asarray(X, dtype=float), float(threshold)))
            except Exception:
                pass
        n = int(X.shape[0])
        if n <= 1:
            return 0.0
        D = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
        np.fill_diagonal(D, np.inf)
        finite = D[np.isfinite(D)]
        if finite.size == 0:
            return 1.0
        g = max(5, int(getattr(self.config, "vr_lcc_grid_size", 25)))
        qs = np.linspace(0.01, 0.99, g)
        cand = np.quantile(finite, qs)
        cand = np.unique(cand)

        def lcc_frac(eps: float) -> float:
            A = D <= eps
            np.fill_diagonal(A, True)
            visited = np.zeros(n, dtype=bool)
            best = 0
            for i in range(n):
                if visited[i]:
                    continue
                stack = [i]
                visited[i] = True
                size = 0
                while stack:
                    u = stack.pop()
                    size += 1
                    nbrs = np.where(A[u])[0]
                    for v in nbrs:
                        if not visited[v]:
                            visited[v] = True
                            stack.append(v)
                if size > best:
                    best = size
                if best >= threshold * n:
                    break
            return best / max(n, 1)

        out = cand[-1]
        for eps in cand:
            if lcc_frac(float(eps)) >= threshold:
                out = float(eps)
                break
        return float(out)

    def _cubical_diagrams(self, field: np.ndarray) -> List[np.ndarray]:
        try:
            import gudhi as gd  # type: ignore
        except Exception:
            if self._strict():
                raise RuntimeError("gudhi not available and TORPEDOCODE_STRICT_TDA=1")
            return [np.zeros((0, 2)) for _ in range(self.config.max_homology_dimension + 1)]

        cc = gd.CubicalComplex(dimensions=field.shape, top_dimensional_cells=field.flatten())
        cc.persistence()
        dgms = [
            np.array(cc.persistence_generators_in_dimension(d) or [])
            for d in range(self.config.max_homology_dimension + 1)
        ]
        out: List[np.ndarray] = []
        for d, D in enumerate(dgms):
            if D.size == 0:
                out.append(np.zeros((0, 2)))
                continue
            if D.ndim == 1:
                D = D.reshape(-1, 2)
            bp = np.stack([D[:, 0], D[:, 1] - D[:, 0]], axis=1)
            out.append(bp)
        return out

    def _liquidity_surface_field(self, slab: np.ndarray) -> np.ndarray:
        """Construct a 2D price-level x time imbalance surface from base LOB sizes.

        Assumes first 2*L columns of the feature matrix are bid and ask sizes per level.
        Uses levels_hint from config if provided; otherwise infers L by halving the first
        2L block size that is <= number of columns.
        """
        T, F = slab.shape
        L = self.config.levels_hint
        if L is None:
            for cand in (10, 20, 40):
                if 2 * cand <= F:
                    L = cand
                    break
            if L is None:
                L = F // 2
        L = max(1, int(min(L, F // 2)))
        bids = slab[:, :L]
        asks = slab[:, L : 2 * L]
        eps = float(getattr(self.config, "imbalance_eps", 1e-6))
        field_type = getattr(self.config, "cubical_scalar_field", "imbalance")
        if field_type == "bid":
            surf = bids
        elif field_type == "ask":
            surf = asks
        elif field_type == "net":
            surf = bids - asks
        else:  
            surf = (bids - asks) / (np.abs(bids) + np.abs(asks) + eps)
        return np.asarray(surf, dtype=np.float32)

    def _vectorise(self, diag: List[np.ndarray]) -> np.ndarray:
        if self.config.persistence_representation == "landscape":
            return self._landscape(diag)
        elif self.config.persistence_representation == "image":
            return self._image(diag)
        else:
            raise ValueError(f"Unsupported representation {self.config.persistence_representation}")

    def _landscape(self, diag: List[np.ndarray]) -> np.ndarray:
        try:
            from persim import PersLandscapeApprox
        except Exception:
            levels = self.config.landscape_levels
            vecs = []
            for D in diag[: self.config.max_homology_dimension + 1]:
                if D.size == 0:
                    vecs.append(np.zeros((levels,), dtype=np.float32))
                else:
                    pers = D[:, 1]
                    q = (
                        np.quantile(pers, np.linspace(0, 1, levels + 1)[1:])
                        if pers.size > 0
                        else np.zeros((levels,))
                    )
                    vecs.append(q.astype(np.float32))
            return np.concatenate(vecs, axis=0)

        grids = []
        for d, D in enumerate(diag[: self.config.max_homology_dimension + 1]):
            if D.size == 0:
                grids.append(np.zeros((self.config.landscape_levels,), dtype=np.float32))
                continue
            b = D[:, 0]
            p = D[:, 1]
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                pla = PersLandscapeApprox(
                    hom_deg=d, num_landscapes=self.config.landscape_levels, resolution=64
                )
                pla.fit([np.stack([b, b + p], axis=1)])
                _x, y = pla.plot_diagrams(return_data=True)
            if getattr(self.config, "landscape_summary", "mean") == "max":
                red = np.nanmax(y, axis=1)
            else:
                red = np.nanmean(y, axis=1)
            grids.append(red.astype(np.float32))
        return np.concatenate(grids, axis=0)

    def _image(self, diag: List[np.ndarray]) -> np.ndarray:
        try:
            from persim import PersistenceImager
        except Exception:
            if self._strict():
                raise RuntimeError("persim not available and TORPEDOCODE_STRICT_TDA=1")
            res = self.config.image_resolution
            img = np.zeros((res, res), dtype=np.float32)
            for D in diag[: self.config.max_homology_dimension + 1]:
                if D.size == 0:
                    continue
                B = D[:, 0]
                P = D[:, 1]
                if B.size == 0:
                    continue
                if hasattr(self, "_active_birth_range") and self._active_birth_range is not None:
                    b0, b1 = self._active_birth_range
                else:
                    b0, b1 = float(np.min(B)), float(np.max(B))
                if hasattr(self, "_active_pers_range") and self._active_pers_range is not None:
                    p0, p1 = self._active_pers_range
                else:
                    p0, p1 = float(np.min(P)), float(np.max(P))
                denom_b = (b1 - b0) if (b1 - b0) > 1e-8 else 1.0
                denom_p = (p1 - p0) if (p1 - p0) > 1e-8 else 1.0
                bi = np.clip(((B - b0) / denom_b) * (res - 1), 0, res - 1).astype(int)
                pi = np.clip(((P - p0) / denom_p) * (res - 1), 0, res - 1).astype(int)
                for x, y in zip(bi, pi):
                    img[y, x] += 1.0
            return img.flatten()

        b_range = getattr(self, "_active_birth_range", None) or getattr(
            self.config, "image_birth_range", None
        ) or (0.0, 1.0)
        p_range = getattr(self, "_active_pers_range", None) or getattr(
            self.config, "image_pers_range", None
        ) or (0.0, 1.0)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="invalid value encountered in subtract")
            pim = PersistenceImager(
                pixel_size=1.0 / self.config.image_resolution,
                birth_range=tuple(map(float, b_range)),
                pers_range=tuple(map(float, p_range)),
                kernel_params={"sigma": self.config.image_bandwidth},
            )
        imgs = []
        for D in diag[: self.config.max_homology_dimension + 1]:
            if D.size == 0:
                imgs.append(
                    np.zeros(
                        (self.config.image_resolution, self.config.image_resolution),
                        dtype=np.float32,
                    )
                )
                continue
            B = np.asarray(D[:, 0], dtype=float)
            P = np.asarray(D[:, 1], dtype=float)
            BP = np.stack([np.nan_to_num(B, nan=0.0), np.nan_to_num(P, nan=0.0)], axis=1)
            BP[:, 0] = np.clip(BP[:, 0], float(b_range[0]), float(b_range[1]))
            BP[:, 1] = np.clip(BP[:, 1], float(p_range[0]), float(p_range[1]))
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                try:
                    img = pim.transform([BP])[0]
                except Exception:
                    img = np.zeros(
                        (self.config.image_resolution, self.config.image_resolution),
                        dtype=np.float32,
                    )
            imgs.append(img.astype(np.float32))
        return np.concatenate([im.flatten() for im in imgs], axis=0)


    def _strict(self) -> bool:
        import os as _os
        env = _os.environ.get("TORPEDOCODE_STRICT_TDA", "0").lower() in {"1", "true"}
        return bool(getattr(self.config, "strict_tda", False)) or env


__all__ = ["TopologicalFeatureGenerator"]
