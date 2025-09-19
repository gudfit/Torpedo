#![deny(warnings)]
use ndarray::parallel::prelude::*;
use ndarray::{s, Array2, ArrayView2, Axis};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyModule, PyTuple};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use serde::Deserialize;
use std::cmp::Ordering;
use std::sync::Arc;

fn l2_distance_matrix(x: &Array2<f64>) -> Array2<f64> {
    let n = x.nrows();
    let mut d = Array2::<f64>::zeros((n, n));
    for i in 0..n {
        for j in (i + 1)..n {
            let mut s = 0.0;
            for k in 0..x.ncols() {
                let diff = x[(i, k)] - x[(j, k)];
                s += diff * diff;
            }
            let v = s.sqrt();
            d[(i, j)] = v;
            d[(j, i)] = v;
        }
    }
    for i in 0..n {
        d[(i, i)] = f64::INFINITY;
    }
    d
}

fn estimate_vr_epsilon_impl(x: &Array2<f64>, q: f64) -> f64 {
    let n = x.nrows();
    if n <= 1 {
        return 1.0;
    }
    if n == 2 {
        let mut s = 0.0;
        for k in 0..x.ncols() {
            let diff = x[(0, k)] - x[(1, k)];
            s += diff * diff;
        }
        return s.sqrt();
    }
    let d = l2_distance_matrix(x);
    let mut in_tree = vec![false; n];
    let mut min_edge = vec![f64::INFINITY; n];
    in_tree[0] = true;
    for j in 0..n {
        min_edge[j] = d[(0, j)];
    }
    let mut edges: Vec<f64> = Vec::with_capacity(n - 1);
    for _ in 0..(n - 1) {
        let mut best_j = None;
        let mut best_w = f64::INFINITY;
        for j in 0..n {
            if !in_tree[j] && min_edge[j].is_finite() && min_edge[j] < best_w {
                best_w = min_edge[j];
                best_j = Some(j);
            }
        }
        if let Some(j) = best_j {
            edges.push(best_w);
            in_tree[j] = true;
            for k in 0..n {
                let w = d[(j, k)];
                if !in_tree[k] && w < min_edge[k] {
                    min_edge[k] = w;
                }
            }
        } else {
            break;
        }
    }
    if edges.is_empty() {
        return 1.0;
    }
    edges.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let m = edges.len();
    let k = ((q.clamp(0.0, 1.0)) * (m as f64)).ceil() as usize;
    let idx = if k == 0 { 0 } else { k.saturating_sub(1) };
    edges[idx]
}

#[pyfunction]
fn estimate_vr_epsilon<'py>(_py: Python<'py>, x: PyReadonlyArray2<f64>, q: f64) -> PyResult<f64> {
    let x = x.as_array();
    Ok(estimate_vr_epsilon_impl(&x.to_owned(), q))
}

fn lcc_fraction(d: &Array2<f64>, eps: f64) -> f64 {
    let n = d.nrows();
    if n == 0 {
        return 0.0;
    }
    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
    for i in 0..n {
        for j in 0..n {
            if i != j && d[(i, j)] <= eps {
                adj[i].push(j);
            }
        }
    }
    let mut visited = vec![false; n];
    let mut best = 0usize;
    for i in 0..n {
        if visited[i] {
            continue;
        }
        let mut stack: Vec<usize> = vec![i];
        visited[i] = true;
        let mut size = 0usize;
        while let Some(u) = stack.pop() {
            size += 1;
            for &v in adj[u].iter() {
                if !visited[v] {
                    visited[v] = true;
                    stack.push(v);
                }
            }
        }
        if size > best {
            best = size;
        }
        if best as f64 >= 0.9999 * (n as f64) {
            break;
        }
    }
    best as f64 / (n as f64)
}

fn epsilon_for_lcc_impl(x: &Array2<f64>, threshold: f64) -> f64 {
    let n = x.nrows();
    if n <= 1 {
        return 0.0;
    }
    let d = l2_distance_matrix(x);
    let mut vals: Vec<f64> = Vec::with_capacity(n * n);
    for i in 0..n {
        for j in 0..n {
            let w = d[(i, j)];
            if w.is_finite() {
                vals.push(w);
            }
        }
    }
    if vals.is_empty() {
        return 1.0;
    }
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    vals.dedup_by(|a, b| (*a - *b).abs() <= f64::EPSILON);
    let mut lo = 0usize;
    let mut hi = vals.len().saturating_sub(1);
    let mut ans = vals[hi];
    while lo <= hi {
        let mid = (lo + hi) / 2;
        let eps = vals[mid];
        let frac = lcc_fraction(&d, eps);
        if frac >= threshold {
            ans = eps;
            if mid == 0 {
                break;
            }
            hi = mid.saturating_sub(1);
        } else {
            if mid == vals.len() - 1 {
                break;
            }
            lo = mid + 1;
        }
    }
    ans
}

#[derive(Debug, Clone, Deserialize)]
struct RollingConfig {
    complex_type: String,
    max_homology_dimension: usize,
    persistence_representation: String,
    landscape_levels: Option<usize>,
    landscape_summary: Option<String>,
    landscape_resolution: Option<usize>,
    image_resolution: Option<usize>,
    image_bandwidth: Option<f64>,
    image_birth_range: Option<(f64, f64)>,
    image_pers_range: Option<(f64, f64)>,
    image_auto_range: Option<bool>,
    vr_auto_epsilon: Option<bool>,
    vr_connectivity_quantile: Option<f64>,
    vr_epsilon_rule: Option<String>,
    vr_lcc_threshold: Option<f64>,
    vr_zscore: Option<bool>,
    use_liquidity_surface: Option<bool>,
    levels_hint: Option<usize>,
    imbalance_eps: Option<f64>,
    cubical_scalar_field: Option<String>,
}

impl RollingConfig {
    fn max_dim(&self) -> usize {
        self.max_homology_dimension
    }

    fn landscape_levels(&self) -> usize {
        self.landscape_levels.unwrap_or(5)
    }

    fn landscape_summary(&self) -> &str {
        self.landscape_summary.as_deref().unwrap_or("mean")
    }

    fn landscape_resolution(&self) -> usize {
        self.landscape_resolution.unwrap_or(64)
    }

    fn image_resolution(&self) -> usize {
        self.image_resolution.unwrap_or(64)
    }

    fn image_bandwidth(&self) -> f64 {
        self.image_bandwidth.unwrap_or(0.05)
    }

    fn image_auto_range(&self) -> bool {
        self.image_auto_range.unwrap_or(true)
    }

    fn vr_auto_epsilon(&self) -> bool {
        self.vr_auto_epsilon.unwrap_or(true)
    }

    fn vr_connectivity_quantile(&self) -> f64 {
        self.vr_connectivity_quantile.unwrap_or(0.99)
    }

    fn vr_epsilon_rule(&self) -> &str {
        self.vr_epsilon_rule.as_deref().unwrap_or("largest_cc")
    }

    fn vr_lcc_threshold(&self) -> f64 {
        self.vr_lcc_threshold.unwrap_or(0.99)
    }

    fn vr_zscore(&self) -> bool {
        self.vr_zscore.unwrap_or(true)
    }

    fn use_liquidity_surface(&self) -> bool {
        self.use_liquidity_surface.unwrap_or(true)
    }

    fn levels_hint(&self) -> Option<usize> {
        self.levels_hint
    }

    fn imbalance_eps(&self) -> f64 {
        self.imbalance_eps.unwrap_or(1e-6)
    }

    fn cubical_scalar_field(&self) -> &str {
        self.cubical_scalar_field.as_deref().unwrap_or("imbalance")
    }
}

fn embedding_dim(cfg: &RollingConfig) -> usize {
    match cfg.persistence_representation.as_str() {
        "landscape" => cfg.landscape_levels() * (cfg.max_dim() + 1),
        "image" => {
            let res = cfg.image_resolution().max(1);
            (cfg.max_dim() + 1) * res * res
        }
        _ => 0,
    }
}

fn searchsorted_right(arr: &[i64], hi: usize, target: i64) -> usize {
    let mut lo = 0usize;
    let mut hi_mut = hi.min(arr.len());
    while lo < hi_mut {
        let mid = (lo + hi_mut) / 2;
        if arr[mid] <= target {
            lo = mid + 1;
        } else {
            hi_mut = mid;
        }
    }
    lo
}

fn compute_liquidity_surface(slab: ArrayView2<f64>, cfg: &RollingConfig) -> Array2<f64> {
    let t = slab.nrows();
    let f = slab.ncols();
    if f == 0 {
        return Array2::<f64>::zeros((t, 0));
    }
    let mut levels = cfg.levels_hint().unwrap_or(0);
    if levels == 0 {
        for cand in [10usize, 20, 40] {
            if 2 * cand <= f {
                levels = cand;
                break;
            }
        }
        if levels == 0 {
            levels = f / 2;
        }
    }
    let max_levels = (f / 2).max(1);
    let levels = levels.max(1).min(max_levels);
    let end = (2 * levels).min(f);
    if end <= levels {
        return slab.to_owned();
    }
    let bids = slab.slice(s![.., 0..levels]).to_owned();
    let asks = slab.slice(s![.., levels..end]).to_owned();
    match cfg.cubical_scalar_field() {
        "bid" => bids,
        "ask" => asks,
        "net" => bids - asks,
        _ => {
            let eps = cfg.imbalance_eps();
            let mut out = bids.clone();
            out.zip_mut_with(&asks, |b, a| {
                let denom = b.abs() + a.abs() + eps;
                if denom.abs() < f64::EPSILON {
                    *b = 0.0;
                } else {
                    *b = (*b - *a) / denom;
                }
            });
            out
        }
    }
}

fn quantile(values: &mut [f64], q: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let q = q.clamp(0.0, 1.0);
    let n = values.len();
    let pos = q * ((n - 1) as f64);
    let lower = pos.floor() as usize;
    let upper = pos.ceil() as usize;
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    if lower == upper {
        values[lower]
    } else {
        let weight = pos - (lower as f64);
        values[lower] * (1.0 - weight) + values[upper] * weight
    }
}

fn vectorise_diagram(
    diag: &[Vec<(f64, f64)>],
    cfg: &RollingConfig,
    birth_range: Option<(f64, f64)>,
    pers_range: Option<(f64, f64)>,
) -> Vec<f32> {
    match cfg.persistence_representation.as_str() {
        "landscape" => {
            let dims = cfg.max_dim() + 1;
            let levels = cfg.landscape_levels();
            let resolution = cfg.landscape_resolution();
            let summary = cfg.landscape_summary().to_string();
            // Range implements `IndexedParallelIterator`, so `collect` preserves
            // the deterministic order of homology dimensions.
            let blocks: Vec<Vec<f32>> = (0..dims)
                .into_par_iter()
                .map(|d| match diag.get(d) {
                    Some(slice) if !slice.is_empty() => {
                        compute_landscape(slice, levels, resolution, summary.as_str())
                    }
                    _ => vec![0.0f32; levels],
                })
                .collect();
            let mut out = Vec::with_capacity(levels * dims);
            for mut block in blocks.into_iter() {
                if block.len() < levels {
                    block.resize(levels, 0.0f32);
                }
                out.extend_from_slice(&block);
            }
            out
        }
        "image" => {
            let dims = cfg.max_dim() + 1;
            let res = cfg.image_resolution();
            let sigma = cfg.image_bandwidth();
            let blocks: Vec<Vec<f32>> = (0..dims)
                .into_par_iter()
                .map(|d| match diag.get(d) {
                    Some(slice) if !slice.is_empty() => {
                        let mut births: Vec<f64> = Vec::with_capacity(slice.len());
                        let mut pers: Vec<f64> = Vec::with_capacity(slice.len());
                        for &(b, p) in slice.iter() {
                            births.push(b);
                            pers.push(p);
                        }
                        compute_image(&births, &pers, res, sigma, birth_range, pers_range)
                    }
                    _ => vec![0.0f32; res * res],
                })
                .collect();
            let mut out = Vec::with_capacity(dims * res * res);
            for mut block in blocks.into_iter() {
                if block.len() < res * res {
                    block.resize(res * res, 0.0f32);
                }
                out.extend_from_slice(&block);
            }
            out
        }
        _ => vec![0.0f32; embedding_dim(cfg)],
    }
}

fn compute_cubical_diagrams_py(
    field: ArrayView2<f64>,
    cfg: &RollingConfig,
    gudhi: &Py<PyModule>,
) -> PyResult<Vec<Vec<(f64, f64)>>> {
    let max_dim = cfg.max_dim();
    Python::with_gil(|py| {
        let dims = field.shape();
        let dims_tuple = PyTuple::new(py, dims.iter().map(|d| d.to_object(py)));
        let top_cells: Vec<f64> = field.iter().cloned().collect();
        let cells = PyArray1::from_vec(py, top_cells);
        let kwargs = PyDict::new(py);
        kwargs.set_item("dimensions", dims_tuple)?;
        kwargs.set_item("top_dimensional_cells", cells)?;
        let cls = gudhi.as_ref(py).getattr("CubicalComplex")?;
        let cc = cls.call((), Some(kwargs))?;
        cc.call_method0("persistence")?;
        let mut out: Vec<Vec<(f64, f64)>> = Vec::with_capacity(max_dim + 1);
        for dim in 0..=max_dim {
            let intervals = cc.call_method1("persistence_intervals_in_dimension", (dim,))?;
            let list = intervals.downcast::<PyList>()?;
            let mut pairs: Vec<(f64, f64)> = Vec::with_capacity(list.len());
            for item in list.iter() {
                let tuple = item.extract::<(f64, f64)>()?;
                let birth = tuple.0;
                let death = tuple.1;
                if birth.is_finite() && death.is_finite() {
                    let pers = death - birth;
                    if pers.is_finite() {
                        pairs.push((birth, pers));
                    }
                }
            }
            out.push(pairs);
        }
        Ok(out)
    })
}

fn compute_vr_diagrams_py(
    points: ArrayView2<f64>,
    cfg: &RollingConfig,
    ripser: &Py<PyAny>,
) -> PyResult<Vec<Vec<(f64, f64)>>> {
    let n = points.nrows();
    if n < 3 {
        return Ok(vec![Vec::new(); cfg.max_dim() + 1]);
    }
    let mut data = points.to_owned();
    if cfg.vr_zscore() {
        for mut col in data.axis_iter_mut(Axis(1)) {
            let mut mean = 0.0;
            let mut count = 0.0;
            for v in col.iter() {
                if v.is_finite() {
                    mean += *v;
                    count += 1.0;
                }
            }
            if count > 0.0 {
                mean /= count;
            }
            let mut var = 0.0;
            for v in col.iter() {
                if v.is_finite() {
                    let diff = *v - mean;
                    var += diff * diff;
                }
            }
            if count > 1.0 {
                var /= count - 1.0;
            }
            let std = if var <= 0.0 { 1.0 } else { var.sqrt() };
            for v in col.iter_mut() {
                *v = (*v - mean) / std;
            }
        }
    }
    let eps = if cfg.vr_auto_epsilon() && n >= 2 {
        match cfg.vr_epsilon_rule() {
            "largest_cc" => Some(epsilon_for_lcc_impl(&data, cfg.vr_lcc_threshold())),
            _ => Some(estimate_vr_epsilon_impl(
                &data,
                cfg.vr_connectivity_quantile(),
            )),
        }
    } else {
        None
    };
    Python::with_gil(|py| {
        let kwargs = PyDict::new(py);
        kwargs.set_item("maxdim", cfg.max_dim())?;
        kwargs.set_item("metric", "euclidean")?;
        if let Some(eps) = eps {
            if eps.is_finite() && eps > 0.0 {
                kwargs.set_item("thresh", eps)?;
            }
        }
        let arr = PyArray2::from_array(py, &data);
        let res = ripser.as_ref(py).call((arr,), Some(kwargs))?;
        let dgms_obj = res.get_item("dgms")?;
        let dgms_list = dgms_obj.downcast::<PyList>()?;
        let mut out: Vec<Vec<(f64, f64)>> = Vec::with_capacity(cfg.max_dim() + 1);
        for dim in 0..=cfg.max_dim() {
            match dgms_list.get_item(dim) {
                Ok(item) => {
                    let arr = item.downcast::<PyArray2<f64>>()?;
                    let view = unsafe { arr.as_array() };
                    let mut pairs: Vec<(f64, f64)> = Vec::with_capacity(view.shape()[0]);
                    for row in view.rows() {
                        let birth = row[0];
                        let death = row[1];
                        if birth.is_finite() && death.is_finite() {
                            let pers = death - birth;
                            if pers.is_finite() {
                                pairs.push((birth, pers));
                            }
                        }
                    }
                    out.push(pairs);
                }
                Err(_) => out.push(Vec::new()),
            }
        }
        Ok(out)
    })
}

fn compute_image(
    births: &[f64],
    pers: &[f64],
    resolution: usize,
    sigma: f64,
    birth_range: Option<(f64, f64)>,
    pers_range: Option<(f64, f64)>,
) -> Vec<f32> {
    let n = births.len().min(pers.len());
    let mut points: Vec<(f64, f64, f64)> = Vec::with_capacity(n);
    for i in 0..n {
        let b = births[i];
        let p = pers[i];
        if b.is_finite() && p.is_finite() {
            points.push((b, p, p.max(0.0)));
        }
    }
    let res = resolution.max(1);
    if points.is_empty() {
        return vec![0.0f32; res * res];
    }

    let (b_lo, mut b_hi) = birth_range.unwrap_or_else(|| {
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for &(b, _, _) in points.iter() {
            lo = lo.min(b);
            hi = hi.max(b);
        }
        if !lo.is_finite() || !hi.is_finite() {
            (0.0, 1.0)
        } else {
            (lo, hi)
        }
    });
    let (p_lo, mut p_hi) = pers_range.unwrap_or_else(|| {
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for &(_, p, _) in points.iter() {
            lo = lo.min(p);
            hi = hi.max(p);
        }
        if !lo.is_finite() || !hi.is_finite() {
            (0.0, 1.0)
        } else {
            (lo, hi)
        }
    });
    if b_hi <= b_lo {
        b_hi = b_lo + 1.0;
    }
    if p_hi <= p_lo {
        p_hi = p_lo + 1.0;
    }

    let mut grid = Array2::<f64>::zeros((res, res));
    let sigma = sigma.abs().max(1e-9);
    let inv_sigma_sq = 1.0 / (2.0 * sigma * sigma);
    let dx = if res > 1 {
        (b_hi - b_lo) / ((res - 1) as f64)
    } else {
        1.0
    };
    let dy = if res > 1 {
        (p_hi - p_lo) / ((res - 1) as f64)
    } else {
        1.0
    };

    grid.axis_iter_mut(Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(iy, mut row)| {
            let py = p_lo + (iy as f64) * dy;
            for (ix, val) in row.iter_mut().enumerate() {
                let px = b_lo + (ix as f64) * dx;
                let mut acc = 0.0f64;
                for &(b, p, w) in points.iter() {
                    let db = px - b;
                    let dp = py - p;
                    let dist2 = db * db + dp * dp;
                    acc += (-dist2 * inv_sigma_sq).exp() * w;
                }
                *val = acc;
            }
        });

    grid.iter().map(|v| *v as f32).collect()
}

fn compute_landscape(
    diagram: &[(f64, f64)],
    k: usize,
    resolution: usize,
    summary_mode: &str,
) -> Vec<f32> {
    let levels = k.max(1);
    if diagram.is_empty() {
        return vec![0.0f32; levels];
    }
    let mut pts: Vec<(f64, f64)> = Vec::with_capacity(diagram.len());
    for &(b, p) in diagram.iter() {
        let death = b + p;
        if b.is_finite() && death.is_finite() && death > b {
            pts.push((b, death));
        }
    }
    if pts.is_empty() {
        return vec![0.0f32; levels];
    }
    let mut x_min = f64::INFINITY;
    let mut x_max = f64::NEG_INFINITY;
    for &(b, d) in pts.iter() {
        x_min = x_min.min(b);
        x_max = x_max.max(d);
    }
    if !x_min.is_finite() || !x_max.is_finite() {
        x_min = 0.0;
        x_max = 1.0;
    }
    if x_max <= x_min {
        x_max = x_min + 1.0;
    }
    let res = resolution.max(2);
    let mut xs = Vec::with_capacity(res);
    for i in 0..res {
        let x = if res == 1 {
            (x_min + x_max) * 0.5
        } else {
            x_min + (x_max - x_min) * (i as f64) / ((res - 1) as f64)
        };
        xs.push(x);
    }

    let mut landscape = Array2::<f64>::zeros((levels, res));
    landscape
        .axis_iter_mut(Axis(1))
        .into_par_iter()
        .enumerate()
        .for_each(|(xi, mut col)| {
            col.fill(0.0);
            let x = xs[xi];
            let mut vals: Vec<f64> = Vec::with_capacity(pts.len());
            for &(b, d) in pts.iter() {
                if x <= b || x >= d {
                    continue;
                }
                let left = x - b;
                let right = d - x;
                let val = left.min(right);
                if val > 0.0 {
                    vals.push(val);
                }
            }
            if vals.is_empty() {
                return;
            }
            vals.sort_by(|a, b| b.partial_cmp(a).unwrap());
            for (level, slot) in col.iter_mut().enumerate() {
                if level < vals.len() {
                    *slot = vals[level];
                } else {
                    break;
                }
            }
        });

    let mut out = Vec::with_capacity(levels);
    for row in landscape.axis_iter(Axis(0)) {
        let summary = match summary_mode {
            "max" => row.fold(0.0f64, |acc, &v| acc.max(v)),
            _ => row.sum() / (res as f64),
        };
        out.push(summary as f32);
    }
    out
}

#[pyfunction]
#[pyo3(signature = (births, pers, resolution, sigma, birth_range=None, pers_range=None))]
fn persistence_image<'py>(
    py: Python<'py>,
    births: PyReadonlyArray1<f64>,
    pers: PyReadonlyArray1<f64>,
    resolution: usize,
    sigma: f64,
    birth_range: Option<(f64, f64)>,
    pers_range: Option<(f64, f64)>,
) -> PyResult<&'py PyArray1<f32>> {
    let births_vec = births.as_array().to_vec();
    let pers_vec = pers.as_array().to_vec();
    let out = py.allow_threads(|| {
        compute_image(
            &births_vec,
            &pers_vec,
            resolution,
            sigma,
            birth_range,
            pers_range,
        )
    });
    Ok(PyArray1::from_vec(py, out))
}

#[pyfunction]
#[pyo3(signature = (diagram, k, resolution, summary_mode="mean"))]
fn persistence_landscape<'py>(
    py: Python<'py>,
    diagram: PyReadonlyArray2<f64>,
    k: usize,
    resolution: usize,
    summary_mode: &str,
) -> PyResult<&'py PyArray1<f32>> {
    let diag_vec: Vec<(f64, f64)> = diagram
        .as_array()
        .rows()
        .into_iter()
        .map(|row| (row[0], row[1]))
        .collect();
    let out = py.allow_threads(|| compute_landscape(&diag_vec, k, resolution, summary_mode));
    Ok(PyArray1::from_vec(py, out))
}

#[pyfunction]
fn epsilon_for_lcc<'py>(
    _py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    threshold: f64,
) -> PyResult<f64> {
    let x = x.as_array();
    Ok(epsilon_for_lcc_impl(&x.to_owned(), threshold))
}

#[pyfunction]
#[pyo3(signature = (series, timestamps_ns, window_sizes_s, stride, cfg_json))]
fn rolling_topo<'py>(
    py: Python<'py>,
    series: PyReadonlyArray2<f64>,
    timestamps_ns: PyReadonlyArray1<i64>,
    window_sizes_s: Vec<i64>,
    stride: usize,
    cfg_json: &str,
) -> PyResult<&'py PyArray2<f32>> {
    let cfg: RollingConfig = serde_json::from_str(cfg_json)
        .map_err(|err| PyValueError::new_err(format!("invalid topology config: {err}")))?;
    let stride = stride.max(1);
    let n_rows = series.shape().get(0).copied().unwrap_or(0);
    if n_rows == 0 {
        let empty = Array2::<f32>::zeros((0, 0));
        return Ok(PyArray2::from_owned_array(py, empty));
    }
    if window_sizes_s.is_empty() {
        let empty = Array2::<f32>::zeros((n_rows, 0));
        return Ok(PyArray2::from_owned_array(py, empty));
    }
    if cfg.persistence_representation != "image" && cfg.persistence_representation != "landscape" {
        return Err(PyValueError::new_err(
            "unsupported persistence representation",
        ));
    }
    if cfg.complex_type != "cubical" && cfg.complex_type != "vietoris_rips" {
        return Err(PyValueError::new_err("unsupported complex type"));
    }

    let series_arr = series.as_array().to_owned();
    let ts_vec = timestamps_ns.as_array().to_vec();
    let embed_per_window = embedding_dim(&cfg);
    let total_embed = embed_per_window * window_sizes_s.len();
    let mut output = vec![0.0f32; n_rows * total_embed];
    let dims = cfg.max_dim() + 1;

    let gudhi_module = if cfg.complex_type == "cubical" {
        Some(py.import("gudhi")?.into_py(py))
    } else {
        None
    };
    let ripser_fn = if cfg.complex_type == "vietoris_rips" {
        Some(py.import("ripser")?.getattr("ripser")?.into_py(py))
    } else {
        None
    };

    if cfg.complex_type == "cubical" && gudhi_module.is_none() {
        return Err(PyValueError::new_err("gudhi module not available"));
    }
    if cfg.complex_type == "vietoris_rips" && ripser_fn.is_none() {
        return Err(PyValueError::new_err("ripser module not available"));
    }

    let series_arc = Arc::new(series_arr);
    let ts_arc = Arc::new(ts_vec);

    for (w_idx, w_sec) in window_sizes_s.iter().enumerate() {
        if *w_sec < 0 {
            return Err(PyValueError::new_err("window size must be non-negative"));
        }
        let window_ns = w_sec
            .checked_mul(1_000_000_000)
            .ok_or_else(|| PyValueError::new_err("window size overflow"))?;
        let mut diag_store: Vec<Vec<Vec<(f64, f64)>>> = vec![vec![Vec::new(); dims]; n_rows];
        let mut births_all: Vec<f64> = Vec::new();
        let mut pers_all: Vec<f64> = Vec::new();
        let need_auto = cfg.persistence_representation == "image"
            && cfg.image_auto_range()
            && (cfg.image_birth_range.is_none() || cfg.image_pers_range.is_none());
        let indices: Vec<usize> = (0..n_rows).step_by(stride).collect();
        let results: PyResult<Vec<(usize, Vec<Vec<(f64, f64)>>)>> = py.allow_threads(|| {
            let cfg_ref = cfg.clone();
            let series_ref = Arc::clone(&series_arc);
            let ts_ref = Arc::clone(&ts_arc);
            let gudhi_ref = gudhi_module.clone();
            let ripser_ref = ripser_fn.clone();
            indices
                .into_par_iter()
                .map(|idx| -> PyResult<(usize, Vec<Vec<(f64, f64)>>)> {
                    let left = ts_ref[idx] - window_ns;
                    let start = searchsorted_right(&ts_ref, idx + 1, left);
                    if start > idx {
                        return Ok((idx, vec![Vec::new(); cfg_ref.max_dim() + 1]));
                    }
                    let slab = series_ref.slice(s![start..idx + 1, ..]);
                    let diag = if cfg_ref.complex_type == "cubical" {
                        let field = if cfg_ref.use_liquidity_surface() {
                            compute_liquidity_surface(slab, &cfg_ref)
                        } else {
                            slab.to_owned()
                        };
                        let module = gudhi_ref
                            .as_ref()
                            .ok_or_else(|| PyValueError::new_err("gudhi module not available"))?;
                        compute_cubical_diagrams_py(field.view(), &cfg_ref, module)?
                    } else {
                        let ripser = ripser_ref
                            .as_ref()
                            .ok_or_else(|| PyValueError::new_err("ripser module not available"))?;
                        compute_vr_diagrams_py(slab, &cfg_ref, ripser)?
                    };
                    Ok((idx, diag))
                })
                .collect()
        });
        let results = results?;
        for (idx, diag) in results.into_iter() {
            for d in 0..dims {
                let data = if d < diag.len() {
                    diag[d].clone()
                } else {
                    Vec::new()
                };
                if need_auto {
                    for &(b, p) in data.iter() {
                        if b.is_finite() && p.is_finite() {
                            births_all.push(b);
                            pers_all.push(p);
                        }
                    }
                }
                diag_store[idx][d] = data;
            }
        }

        let mut birth_range = cfg.image_birth_range;
        let mut pers_range = cfg.image_pers_range;
        if cfg.persistence_representation == "image" {
            if (birth_range.is_none() || pers_range.is_none())
                && need_auto
                && !births_all.is_empty()
                && !pers_all.is_empty()
            {
                let mut bvals = births_all.clone();
                let mut pvals = pers_all.clone();
                let bmin = quantile(&mut bvals, 0.01);
                let bmax = quantile(&mut bvals, 0.99);
                let pmin = quantile(&mut pvals, 0.01);
                let pmax = quantile(&mut pvals, 0.99);
                birth_range = Some((bmin, bmax));
                pers_range = Some((pmin, pmax));
            }
        }

        for idx in (0..n_rows).step_by(stride) {
            let vec = vectorise_diagram(&diag_store[idx], &cfg, birth_range, pers_range);
            let offset = idx * total_embed + w_idx * embed_per_window;
            if vec.len() == embed_per_window {
                output[offset..offset + embed_per_window].copy_from_slice(&vec);
            }
        }
    }

    let arr = Array2::from_shape_vec((n_rows, total_embed), output)
        .map_err(|_| PyValueError::new_err("invalid output shape"))?;
    Ok(PyArray2::from_owned_array(py, arr))
}

#[pymodule]
fn torpedocode_tda(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(estimate_vr_epsilon, m)?)?;
    m.add_function(wrap_pyfunction!(epsilon_for_lcc, m)?)?;
    m.add_function(wrap_pyfunction!(queue_age_series, m)?)?;
    m.add_function(wrap_pyfunction!(queue_age_series_with_halts, m)?)?;
    m.add_function(wrap_pyfunction!(queue_age_levels, m)?)?;
    m.add_function(wrap_pyfunction!(persistence_image, m)?)?;
    m.add_function(wrap_pyfunction!(persistence_landscape, m)?)?;
    m.add_function(wrap_pyfunction!(rolling_topo, m)?)?;
    Ok(())
}

#[pyfunction]
fn queue_age_series(
    sz: PyReadonlyArray1<f64>,
    pr: PyReadonlyArray1<f64>,
    dt: PyReadonlyArray1<f32>,
) -> PyResult<Vec<f32>> {
    let sz = sz.as_array();
    let pr = pr.as_array();
    let dt = dt.as_array();
    let n = sz.len();
    let mut age = vec![0.0f32; n];
    if n == 0 {
        return Ok(age);
    }
    for i in 1..n {
        let s_changed = sz[i] != sz[i - 1];
        let p_i = pr[i];
        let p_prev = pr[i - 1];
        let p_changed = !p_i.is_finite() || !p_prev.is_finite() || (p_i != p_prev);
        if s_changed || p_changed {
            age[i] = 0.0;
        } else {
            age[i] = age[i - 1] + dt[i].max(0.0);
        }
    }
    Ok(age)
}

#[pyfunction]
fn queue_age_series_with_halts(
    sz: PyReadonlyArray1<f64>,
    pr: PyReadonlyArray1<f64>,
    dt: PyReadonlyArray1<f32>,
    halted: PyReadonlyArray1<bool>,
) -> PyResult<Vec<f32>> {
    let sz = sz.as_array();
    let pr = pr.as_array();
    let dt = dt.as_array();
    let halted = halted.as_array();
    let n = sz.len();
    let mut age = vec![0.0f32; n];
    if n == 0 {
        return Ok(age);
    }
    for i in 1..n {
        let s_changed = sz[i] != sz[i - 1];
        let p_i = pr[i];
        let p_prev = pr[i - 1];
        let p_changed = !p_i.is_finite() || !p_prev.is_finite() || (p_i != p_prev);
        if halted.get(i).copied().unwrap_or(false) || s_changed || p_changed {
            age[i] = 0.0;
        } else {
            age[i] = age[i - 1] + dt[i].max(0.0);
        }
    }
    Ok(age)
}

#[pyfunction]
fn queue_age_levels(
    sz: PyReadonlyArray2<f64>,
    pr: PyReadonlyArray2<f64>,
    dt: PyReadonlyArray1<f32>,
    halted: Option<PyReadonlyArray1<bool>>,
) -> PyResult<Vec<f32>> {
    let sz = sz.as_array();
    let pr = pr.as_array();
    let dt = dt.as_array();
    let (t, l) = (sz.nrows(), sz.ncols());
    let mut out = vec![0.0f32; t * l];
    if t == 0 || l == 0 {
        return Ok(out);
    }
    let halted_opt = halted.as_ref().map(|h| h.as_array());
    for i in 1..t {
        let dti = dt[i].max(0.0);
        let halt = halted_opt.map(|h| h[i]).unwrap_or(false);
        for j in 0..l {
            let idx = i * l + j;
            if halt
                || sz[(i, j)] != sz[(i - 1, j)]
                || pr[(i, j)] != pr[(i - 1, j)]
                || !pr[(i, j)].is_finite()
                || !pr[(i - 1, j)].is_finite()
            {
                out[idx] = 0.0;
            } else {
                out[idx] = out[idx - l] + dti;
            }
        }
    }
    Ok(out)
}
