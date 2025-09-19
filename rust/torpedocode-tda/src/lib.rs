#![deny(warnings)]
use ndarray::{Array2, Axis};
use ndarray::parallel::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

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

#[pyfunction]
fn estimate_vr_epsilon<'py>(_py: Python<'py>, x: PyReadonlyArray2<f64>, q: f64) -> PyResult<f64> {
    let x = x.as_array();
    let n = x.nrows();
    if n <= 1 {
        return Ok(1.0);
    }
    if n == 2 {
        let mut s = 0.0;
        for k in 0..x.ncols() {
            let diff = x[(0, k)] - x[(1, k)];
            s += diff * diff;
        }
        return Ok(s.sqrt());
    }
    let d = l2_distance_matrix(&x.to_owned());
    // Prim's algorithm to build MST and collect edge weights
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
        return Ok(1.0);
    }
    edges.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let m = edges.len();
    let k = ((q.clamp(0.0, 1.0)) * (m as f64)).ceil() as usize;
    let idx = if k == 0 { 0 } else { k.saturating_sub(1) };
    Ok(edges[idx])
}

fn lcc_fraction(d: &Array2<f64>, eps: f64) -> f64 {
    let n = d.nrows();
    if n == 0 { return 0.0; }
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
        if visited[i] { continue; }
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
        if size > best { best = size; }
        if best as f64 >= 0.9999 * (n as f64) { break; }
    }
    best as f64 / (n as f64)
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

    grid
        .axis_iter_mut(Axis(0))
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
    let out = py.allow_threads(|| compute_image(&births_vec, &pers_vec, resolution, sigma, birth_range, pers_range));
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
fn epsilon_for_lcc<'py>(_py: Python<'py>, x: PyReadonlyArray2<f64>, threshold: f64) -> PyResult<f64> {
    let x = x.as_array();
    let n = x.nrows();
    if n <= 1 { return Ok(0.0); }
    let d = l2_distance_matrix(&x.to_owned());
    // Collect unique finite distances as candidate epsilons
    let mut vals: Vec<f64> = Vec::with_capacity(n * n);
    for i in 0..n { for j in 0..n { let w = d[(i,j)]; if w.is_finite() { vals.push(w); } } }
    if vals.is_empty() { return Ok(1.0); }
    vals.sort_by(|a,b| a.partial_cmp(b).unwrap());
    vals.dedup_by(|a,b| (*a - *b).abs() <= std::f64::EPSILON);
    // Monotone binary search for minimal epsilon achieving threshold LCC fraction
    let mut lo = 0usize;
    let mut hi = vals.len()-1;
    let mut ans = vals[hi];
    while lo <= hi {
        let mid = (lo + hi) / 2;
        let eps = vals[mid];
        let frac = lcc_fraction(&d, eps);
        if frac >= threshold {
            ans = eps;
            if mid == 0 { break; }
            hi = mid - 1;
        } else {
            if mid == vals.len()-1 { break; }
            lo = mid + 1;
        }
    }
    Ok(ans)
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
    Ok(())
}

#[pyfunction]
fn queue_age_series(sz: PyReadonlyArray1<f64>, pr: PyReadonlyArray1<f64>, dt: PyReadonlyArray1<f32>) -> PyResult<Vec<f32>> {
    let sz = sz.as_array();
    let pr = pr.as_array();
    let dt = dt.as_array();
    let n = sz.len();
    let mut age = vec![0.0f32; n];
    if n == 0 { return Ok(age); }
    for i in 1..n {
        let s_changed = sz[i] != sz[i-1];
        let p_i = pr[i];
        let p_prev = pr[i-1];
        let p_changed = !p_i.is_finite() || !p_prev.is_finite() || (p_i != p_prev);
        if s_changed || p_changed {
            age[i] = 0.0;
        } else {
            age[i] = age[i-1] + dt[i].max(0.0);
        }
    }
    Ok(age)
}

#[pyfunction]
fn queue_age_series_with_halts(sz: PyReadonlyArray1<f64>, pr: PyReadonlyArray1<f64>, dt: PyReadonlyArray1<f32>, halted: PyReadonlyArray1<bool>) -> PyResult<Vec<f32>> {
    let sz = sz.as_array();
    let pr = pr.as_array();
    let dt = dt.as_array();
    let halted = halted.as_array();
    let n = sz.len();
    let mut age = vec![0.0f32; n];
    if n == 0 { return Ok(age); }
    for i in 1..n {
        let s_changed = sz[i] != sz[i-1];
        let p_i = pr[i];
        let p_prev = pr[i-1];
        let p_changed = !p_i.is_finite() || !p_prev.is_finite() || (p_i != p_prev);
        if halted.get(i).copied().unwrap_or(false) || s_changed || p_changed {
            age[i] = 0.0;
        } else {
            age[i] = age[i-1] + dt[i].max(0.0);
        }
    }
    Ok(age)
}

#[pyfunction]
fn queue_age_levels(sz: PyReadonlyArray2<f64>, pr: PyReadonlyArray2<f64>, dt: PyReadonlyArray1<f32>, halted: Option<PyReadonlyArray1<bool>>) -> PyResult<Vec<f32>> {
    let sz = sz.as_array();
    let pr = pr.as_array();
    let dt = dt.as_array();
    let (t, l) = (sz.nrows(), sz.ncols());
    let mut out = vec![0.0f32; t * l];
    if t == 0 || l == 0 { return Ok(out); }
    let halted_opt = halted.as_ref().map(|h| h.as_array());
    for i in 1..t {
        let dti = dt[i].max(0.0);
        let halt = halted_opt.map(|h| h[i]).unwrap_or(false);
        for j in 0..l {
            let idx = i * l + j;
            if halt || sz[(i,j)] != sz[(i-1,j)] || pr[(i,j)] != pr[(i-1,j)] || !pr[(i,j)].is_finite() || !pr[(i-1,j)].is_finite() {
                out[idx] = 0.0;
            } else {
                out[idx] = out[idx - l] + dti;
            }
        }
    }
    Ok(out)
}
