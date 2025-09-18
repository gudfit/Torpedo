#![deny(warnings)]
use ndarray::Array2;
use numpy::PyReadonlyArray2;
use pyo3::prelude::*;
use numpy::PyReadonlyArray1;

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
