#![allow(deprecated)]
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::fs::File;
use std::io::Read;

fn to_canonical<'a>(py: Python<'a>, ts_ns: u64, event_type: &str, size: f64, price: f64, side: Option<&str>, symbol: &str, venue: &str) -> PyResult<&'a PyDict> {
    let d = PyDict::new(py);
    d.set_item("timestamp", ts_ns as i128)?; 
    d.set_item("event_type", event_type)?;
    d.set_item("size", size)?;
    d.set_item("price", price)?;
    d.set_item("level", py.None())?;
    if let Some(s) = side { d.set_item("side", s)?; } else { d.set_item("side", py.None())?; }
    d.set_item("symbol", if symbol.is_empty() { py.None() } else { symbol.into_py(py) })?;
    d.set_item("venue", venue)?;
    Ok(d)
}

fn to_price_scaled(v: f64, tick_size: f64) -> f64 {
    if tick_size > 0.0 {
        return (v / tick_size).round() * tick_size;
    }
    v
}

fn read_file(path: &str) -> PyResult<Vec<u8>> {
    let mut f = File::open(path).map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf).map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))?;
    Ok(buf)
}

// Minimal custom binary format (timestamp 8 LE, type, payload LE) 
fn parse_itch_minimal<'a>(py: Python<'a>, buf: &[u8], tick_size: f64, symbol: &str) -> PyResult<PyObject> {
    let mut out: Vec<PyObject> = Vec::new();
    let mut i = 0usize;
    while i + 9 <= buf.len() {
        let ts_ns = u64::from_le_bytes(buf[i..i + 8].try_into().unwrap()); i += 8;
        let m = buf[i] as char; i += 1;
        match m {
            'A' => {
                if i + 8 + 1 + 4 + 8 + 8 > buf.len() { break; }
                i += 8;
                let side = buf[i] as char; i += 1;
                let shares = u32::from_le_bytes(buf[i..i+4].try_into().unwrap()) as f64; i += 4;
                let price_i = u64::from_le_bytes(buf[i..i+8].try_into().unwrap()) as f64; i += 8;
                // stock id/hash
                i += 8;
                let price = to_price_scaled(price_i * 1e-4, tick_size);
                let d = to_canonical(py, ts_ns, if side == 'B' { "LO+" } else { "LO-" }, shares, price, Some(&side.to_string()), symbol, "ITCH")?;
                out.push(d.into());
            },
            'F' => {
                if i + 8 + 1 + 4 + 8 + 4 > buf.len() { break; }
                i += 8;
                let side = buf[i] as char; i += 1;
                let shares = u32::from_le_bytes(buf[i..i+4].try_into().unwrap()) as f64; i += 4;
                let price_i = u64::from_le_bytes(buf[i..i+8].try_into().unwrap()) as f64; i += 8;
                i += 4;
                let price = to_price_scaled(price_i * 1e-4, tick_size);
                let d = to_canonical(py, ts_ns, if side == 'B' { "LO+" } else { "LO-" }, shares, price, Some(&side.to_string()), symbol, "ITCH")?;
                out.push(d.into());
            },
            'E' => {
                if i + 8 + 4 > buf.len() { break; }
                i += 8; 
                let executed = u32::from_le_bytes(buf[i..i+4].try_into().unwrap()) as f64; i += 4;
                let d = to_canonical(py, ts_ns, "MO+", executed, f64::NAN, None, symbol, "ITCH")?;
                out.push(d.into());
            },
            'X' => {
                if i + 8 + 4 > buf.len() { break; }
                i += 8;
                let canceled = u32::from_le_bytes(buf[i..i+4].try_into().unwrap()) as f64; i += 4;
                let d = to_canonical(py, ts_ns, "CX+", canceled, f64::NAN, None, symbol, "ITCH")?;
                out.push(d.into());
            },
            'P' => {
                if i + 8 + 1 + 4 + 8 > buf.len() { break; }
                i += 8; 
                let side = buf[i] as char; i += 1;
                let shares = u32::from_le_bytes(buf[i..i+4].try_into().unwrap()) as f64; i += 4;
                let price_i = u64::from_le_bytes(buf[i..i+8].try_into().unwrap()) as f64; i += 8;
                let price = to_price_scaled(price_i * 1e-4, tick_size);
                let d = to_canonical(py, ts_ns, "MO+", shares, price, Some(&side.to_string()), symbol, "ITCH")?;
                out.push(d.into());
            },
            'D' => {
                if i + 8 > buf.len() { break; }
                i += 8;
                let d = to_canonical(py, ts_ns, "CX+", 0.0, f64::NAN, None, symbol, "ITCH")?;
                out.push(d.into());
            },
            'C' => {
                if i + 8 + 4 + 8 > buf.len() { break; }
                i += 8;
                let executed = u32::from_le_bytes(buf[i..i+4].try_into().unwrap()) as f64; i += 4;
                let price_i = u64::from_le_bytes(buf[i..i+8].try_into().unwrap()) as f64; i += 8;
                let price = to_price_scaled(price_i * 1e-4, tick_size);
                let d = to_canonical(py, ts_ns, "MO+", executed, price, None, symbol, "ITCH")?;
                out.push(d.into());
            },
            'U' => {
                if i + 8 + 8 + 4 + 8 > buf.len() { break; }
                i += 8; // orig
                i += 8; // new
                let new_shares = u32::from_le_bytes(buf[i..i+4].try_into().unwrap()) as f64; i += 4;
                let price_i = u64::from_le_bytes(buf[i..i+8].try_into().unwrap()) as f64; i += 8;
                let price = to_price_scaled(price_i * 1e-4, tick_size);
                let d = to_canonical(py, ts_ns, "LO+", new_shares, price, None, symbol, "ITCH")?;
                out.push(d.into());
            },
            _ => {
                // Unknown byte at this position; advance by one and continue scanning
                continue;
            }
        }
    }
    Ok(PyList::new(py, out).into())
}

// NASDAQ ITCH 5.0 exact subset (big-endian fields, 6-byte timestamp) 
fn be_u16(b: &[u8]) -> u16 { ((b[0] as u16) << 8) | (b[1] as u16) }
fn be_u32(b: &[u8]) -> u32 { ((b[0] as u32) << 24) | ((b[1] as u32) << 16) | ((b[2] as u32) << 8) | (b[3] as u32) }
fn be_u48_to_u64(b: &[u8]) -> u64 {
    ((b[0] as u64) << 40) | ((b[1] as u64) << 32) | ((b[2] as u64) << 24) | ((b[3] as u64) << 16) | ((b[4] as u64) << 8) | (b[5] as u64)
}

fn parse_itch_nasdaq_50<'a>(py: Python<'a>, buf: &[u8], tick_size: f64, symbol: &str) -> PyResult<PyObject> {
    let mut i = 0usize;
    let mut out: Vec<PyObject> = Vec::new();
    let is_type = |b: u8| matches!(b as char, 'S'|'R'|'H'|'L'|'A'|'F'|'E'|'C'|'X'|'D'|'U'|'P'|'Q'|'B');
    
    while i + 1 <= buf.len() {
        let m = buf[i] as char; i += 1;
        match m {
            // System Event (S) — skip (non-book impacting)
            'S' => {
                if i + 2+2+6+1 > buf.len() { break; }
                i += 2; i += 2; // stock locate, tracking
                i += 6; // ts
                i += 1; // event code
                while i < buf.len() && !is_type(buf[i]) { i += 1; }
            }
            // LULD Auction Collar (L) — skip (metadata)
            'L' => {
                // Consume fields conservatively: headers + stock + reference + upper + lower
                let need = 2 + 2 + 6  // stock locate, tracking, ts
                    + 8               // stock
                    + 4 + 4 + 4;      // reference price, upper collar, lower collar (prices)
                if i + need > buf.len() { break; }
                i += need;
                while i < buf.len() && !is_type(buf[i]) { i += 1; }
            }
            // Stock Directory (R) — skip (metadata)
            'R' => {
                // Consume full message conservatively per ITCH 5.0
                let need = 2+2+6 + 8 + 1+1 + 4 + 1 + 1 + 2 + 1 + 1 + 1 + 1 + 4 + 1;
                if i + need > buf.len() { break; }
                i += need;
                while i < buf.len() && !is_type(buf[i]) { i += 1; }
            }
            // Trading Action (H) — propagate trading_state and reason
            'H' => {
                if i + 2+2+6 + 8 + 1 + 1 + 4 > buf.len() { break; }
                i += 2; // stock locate
                i += 2; // tracking
                let ts = be_u48_to_u64(&buf[i..i+6]); i += 6; // ts
                let _stock = &buf[i..i+8]; i += 8; // stock (ignored)
                let state_c = buf[i] as char; i += 1; // trading state code
                i += 1; // reserved
                let reason = be_u32(&buf[i..i+4]); i += 4; // reason code (raw)
                // Emit a canonical META event with trading_state and reason; size=0, price=NaN
                let d = to_canonical(py, ts, "META", 0.0, f64::NAN, None, symbol, "ITCH")?;
                d.set_item("trading_state", state_c.to_string())?;
                d.set_item("state_reason", reason)?;
                // Convenience boolean
                let halted = matches!(state_c, 'H' | 'P');
                d.set_item("halted", halted)?;
                out.push(d.into());
            }
            // Add Order (A)
            'A' => {
                if i + 2+2+6+8+1+4+8+4 > buf.len() { break; }
                let _stock_loc = be_u16(&buf[i..i+2]); i += 2;
                let _track = be_u16(&buf[i..i+2]); i += 2;
                let ts = be_u48_to_u64(&buf[i..i+6]); i += 6;
                i += 8; // order ref
                let side = buf[i] as char; i += 1;
                let shares = be_u32(&buf[i..i+4]) as f64; i += 4;
                i += 8; // stock (8 bytes)
                let price_i = be_u32(&buf[i..i+4]) as f64; i += 4;
                let price = to_price_scaled(price_i * 1e-4, tick_size);
                let d = to_canonical(py, ts, if side == 'B' { "LO+" } else { "LO-" }, shares, price, Some(&side.to_string()), symbol, "ITCH")?;
                out.push(d.into());
            }
            // Add Order with MPID (F)
            'F' => {
                if i + 2+2+6+8+1+4+8+4+4 > buf.len() { break; }
                i += 2; i += 2; // stock locate, tracking
                let ts = be_u48_to_u64(&buf[i..i+6]); i += 6;
                i += 8; // order ref
                let side = buf[i] as char; i += 1;
                let shares = be_u32(&buf[i..i+4]) as f64; i += 4;
                i += 8; // stock
                let price_i = be_u32(&buf[i..i+4]) as f64; i += 4;
                i += 4; // mpid
                let price = to_price_scaled(price_i * 1e-4, tick_size);
                let d = to_canonical(py, ts, if side == 'B' { "LO+" } else { "LO-" }, shares, price, Some(&side.to_string()), symbol, "ITCH")?;
                out.push(d.into());
            }
            // Order Executed (E)
            'E' => {
                if i + 2+2+6+8+4+8 > buf.len() { break; }
                i += 2; i += 2;
                let ts = be_u48_to_u64(&buf[i..i+6]); i += 6;
                i += 8; // order ref
                let executed = be_u32(&buf[i..i+4]) as f64; i += 4;
                i += 8; // match
                let d = to_canonical(py, ts, "MO+", executed, f64::NAN, None, symbol, "ITCH")?;
                out.push(d.into());
            }
            // Order Executed With Price (C)
            'C' => {
                if i + 2+2+6+8+4+8+1+4 > buf.len() { break; }
                i += 2; i += 2;
                let ts = be_u48_to_u64(&buf[i..i+6]); i += 6;
                i += 8; // order ref
                let executed = be_u32(&buf[i..i+4]) as f64; i += 4;
                i += 8; // match
                let _printable = buf[i]; i += 1;
                let price_i = be_u32(&buf[i..i+4]) as f64; i += 4;
                let price = to_price_scaled(price_i * 1e-4, tick_size);
                let d = to_canonical(py, ts, "MO+", executed, price, None, symbol, "ITCH")?;
                out.push(d.into());
            }
            // Order Cancel (X)
            'X' => {
                if i + 2+2+6+8+4 > buf.len() { break; }
                i += 2; i += 2;
                let ts = be_u48_to_u64(&buf[i..i+6]); i += 6;
                i += 8; // order ref
                let canceled = be_u32(&buf[i..i+4]) as f64; i += 4;
                let d = to_canonical(py, ts, "CX+", canceled, f64::NAN, None, symbol, "ITCH")?;
                out.push(d.into());
            }
            // Order Delete (D)
            'D' => {
                if i + 2+2+6+8 > buf.len() { break; }
                i += 2; i += 2;
                let ts = be_u48_to_u64(&buf[i..i+6]); i += 6;
                i += 8; // order ref
                let d = to_canonical(py, ts, "CX+", 0.0, f64::NAN, None, symbol, "ITCH")?;
                out.push(d.into());
            }
            // Order Replace (U)
            'U' => {
                if i + 2+2+6+8+8+4+4 > buf.len() { break; }
                i += 2; i += 2;
                let ts = be_u48_to_u64(&buf[i..i+6]); i += 6;
                i += 8; // orig order ref
                i += 8; // new order ref
                let new_shares = be_u32(&buf[i..i+4]) as f64; i += 4;
                let price_i = be_u32(&buf[i..i+4]) as f64; i += 4;
                let price = to_price_scaled(price_i * 1e-4, tick_size);
                let d = to_canonical(py, ts, "LO+", new_shares, price, None, symbol, "ITCH")?;
                out.push(d.into());
            }
            // Trade (Non-Cross) (P)
            'P' => {
                if i + 2+2+6+8+1+4+8+4+8 > buf.len() { break; }
                i += 2; i += 2;
                let ts = be_u48_to_u64(&buf[i..i+6]); i += 6;
                i += 8; // order ref
                let side = buf[i] as char; i += 1;
                let shares = be_u32(&buf[i..i+4]) as f64; i += 4;
                i += 8; // stock
                let price_i = be_u32(&buf[i..i+4]) as f64; i += 4;
                i += 8; // match
                let price = to_price_scaled(price_i * 1e-4, tick_size);
                let d = to_canonical(py, ts, "MO+", shares, price, Some(&side.to_string()), symbol, "ITCH")?;
                out.push(d.into());
            }
            // Cross Trade (Q)
            'Q' => {
                if i + 2+2+6 + 8 + 8 + 4 + 8 + 1 > buf.len() { break; }
                i += 2; i += 2;
                let ts = be_u48_to_u64(&buf[i..i+6]); i += 6;
                let shares = u64::from_be_bytes([buf[i],buf[i+1],buf[i+2],buf[i+3],buf[i+4],buf[i+5],buf[i+6],buf[i+7]]) as f64; i += 8;
                i += 8; // stock
                let price_i = be_u32(&buf[i..i+4]) as f64; i += 4;
                i += 8; // match
                i += 1; // cross type
                let price = to_price_scaled(price_i * 1e-4, tick_size);
                let d = to_canonical(py, ts, "MO+", shares, price, None, symbol, "ITCH")?;
                out.push(d.into());
            }
            // Broken Trade (B)
            'B' => {
                if i + 2+2+6 + 8 > buf.len() { break; }
                i += 2; i += 2;
                let ts = be_u48_to_u64(&buf[i..i+6]); i += 6;
                i += 8; // match
                let d = to_canonical(py, ts, "CX+", 0.0, f64::NAN, None, symbol, "ITCH")?;
                out.push(d.into());
            }
            _ => { if i < buf.len() { /* advance in next loop */ } continue; }
        }
    }
    Ok(PyList::new(py, out).into())
}

#[pyfunction]
fn parse_itch_file(py: Python<'_>, path: &str, tick_size: f64, symbol: &str, spec: &str) -> PyResult<PyObject> {
    let buf = read_file(path)?;
    let want_nasdaq = spec.to_ascii_lowercase().starts_with("nasdaq-itch-5.0");
    let try_nasdaq = || parse_itch_nasdaq_50(py, &buf, tick_size, symbol);
    let try_min = || parse_itch_minimal(py, &buf, tick_size, symbol);
    let is_type = |c: u8| matches!(c as char, 'A'|'F'|'E'|'C'|'X'|'D'|'U'|'P');

    if want_nasdaq {
        if buf.len() >= 9 && !matches!(buf[0] as char, 'S'|'R'|'H'|'A'|'F'|'E'|'C'|'X'|'D'|'U'|'P'|'Q'|'B') && is_type(buf[8]) {
            return try_min();
        }
        let o1 = try_nasdaq()?;
        let l1 = o1.as_ref(py).downcast::<PyList>().map(|l| l.len()).unwrap_or(0);
        if l1 > 0 { return Ok(o1); }
        return try_min();
    }
    try_min()
}

#[pyfunction]
fn parse_ouch_file(py: Python<'_>, path: &str, tick_size: f64, symbol: &str, spec: &str) -> PyResult<PyObject> {
    let mut f = File::open(path).map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf).map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))?;
    // Choose NASDAQ 4.2 if spec hint provided; else parse minimal format (with 8-byte ns prefix)
    if spec.to_ascii_lowercase().starts_with("nasdaq-ouch-4.2") {
        let is_min = buf.len() >= 9 && {
            let t0 = buf[0] as char;
            let t8 = buf[8] as char;
            !matches!(t0, 'O'|'U'|'X'|'E'|'C'|'D') && matches!(t8, 'O'|'U'|'X'|'E'|'C'|'D')
        };
        if is_min {
            return parse_ouch_minimal(py, &buf, tick_size, symbol);
        }
        let o = parse_ouch_nasdaq_42(py, &buf, tick_size, symbol)?;
        let l = o.as_ref(py).downcast::<PyList>().map(|l| l.len()).unwrap_or(0);
        if l == 0 {
            // Try scanning to the first usable OUCH type (skip any leading system events)
            let mut j = 0usize;
            let is_type2 = |c: u8| matches!(c as char, 'S'|'A'|'O'|'U'|'X'|'E'|'C'|'R'|'T');
            while j < buf.len() {
                let ch = buf[j] as char;
                if ch == 'S' {
                    j = j.saturating_add(2); // skip 'S' + 1-byte code
                    continue;
                }
                if is_type2(buf[j]) { break; }
                j += 1;
            }
            if j < buf.len() {
                let o2 = parse_ouch_nasdaq_42(py, &buf[j..], tick_size, symbol)?;
                let l2 = o2.as_ref(py).downcast::<PyList>().map(|l| l.len()).unwrap_or(0);
                if l2 > 0 { return Ok(o2); }
            }
            return parse_ouch_minimal(py, &buf, tick_size, symbol);
        }
        return Ok(o);
    }
    parse_ouch_minimal(py, &buf, tick_size, symbol)
}

fn parse_ouch_minimal<'a>(py: Python<'a>, buf: &[u8], tick_size: f64, symbol: &str) -> PyResult<PyObject> {
    let mut out: Vec<PyObject> = Vec::new();
    let mut i = 0usize;
    while i + 9 <= buf.len() {
        let ts_ns = u64::from_le_bytes(buf[i..i + 8].try_into().unwrap()); i += 8;
        let m = buf[i] as char; i += 1;
        match m {
            'O' => {
                if i + 8 + 1 + 4 + 8 > buf.len() { break; }
                i += 8; // client_order_id
                let side = buf[i] as char; i += 1;
                let shares = u32::from_le_bytes(buf[i..i+4].try_into().unwrap()) as f64; i += 4;
                let price_i = u64::from_le_bytes(buf[i..i+8].try_into().unwrap()) as f64; i += 8;
                let price = to_price_scaled(price_i * 1e-4, tick_size);
                let d = to_canonical(py, ts_ns, if side == 'B' { "LO+" } else { "LO-" }, shares, price, Some(&side.to_string()), symbol, "OUCH")?;
                out.push(d.into());
            },
            'U' => {
                if i + 8 + 8 + 4 + 8 > buf.len() { break; }
                i += 8; // orig
                i += 8; // new
                let new_shares = u32::from_le_bytes(buf[i..i+4].try_into().unwrap()) as f64; i += 4;
                let price_i = u64::from_le_bytes(buf[i..i+8].try_into().unwrap()) as f64; i += 8;
                let price = to_price_scaled(price_i * 1e-4, tick_size);
                let d = to_canonical(py, ts_ns, "LO+", new_shares, price, None, symbol, "OUCH")?;
                out.push(d.into());
            },
            'X' => {
                if i + 8 + 4 > buf.len() { break; }
                i += 8; // client_id
                let canceled = u32::from_le_bytes(buf[i..i+4].try_into().unwrap()) as f64; i += 4;
                let d = to_canonical(py, ts_ns, "CX+", canceled, f64::NAN, None, symbol, "OUCH")?;
                out.push(d.into());
            },
            'E' => {
                if i + 8 + 4 > buf.len() { break; }
                i += 8; // client_id
                let executed = u32::from_le_bytes(buf[i..i+4].try_into().unwrap()) as f64; i += 4;
                let d = to_canonical(py, ts_ns, "MO+", executed, f64::NAN, None, symbol, "OUCH")?;
                out.push(d.into());
            },
            'D' => {
                if i + 8 > buf.len() { break; }
                i += 8; // client_id
                let d = to_canonical(py, ts_ns, "CX+", 0.0, f64::NAN, None, symbol, "OUCH")?;
                out.push(d.into());
            },
            _ => { break; }
        }
    }
    Ok(PyList::new(py, out).into())
}

fn parse_ouch_nasdaq_42<'a>(py: Python<'a>, buf: &[u8], tick_size: f64, symbol: &str) -> PyResult<PyObject> {
    // OUCH 4.2 (server) subset with byte-accurate widths for core messages.
    let mut out: Vec<PyObject> = Vec::new();
    let mut i = 0usize;
    let is_type = |b: u8| matches!(b as char, 'S'|'A'|'O'|'U'|'X'|'E'|'C'|'R'|'T'|'P'|'D'|'B');
    while i < buf.len() {
        // Find the start of the next message
        while i < buf.len() && !is_type(buf[i]) {
            i += 1;
        }
        if i >= buf.len() { break; }

        let m = buf[i] as char;
        i += 1;

        match m {
            // System event (1 byte code) — skip
            'S' => { if i + 1 > buf.len() { break; } i += 1; }
            // Accepted (A)
            'A' => {
                let need = 14 + 1 + 4 + 8 + 4;
                if i + need > buf.len() { break; }
                let tok_bytes = &buf[i..i+14];
                let token = String::from_utf8_lossy(tok_bytes).trim_end().to_string();
                i += 14;
                let side = buf[i] as char; i += 1;
                let shares = be_u32(&buf[i..i+4]) as f64; i += 4;
                i += 8; // stock
                let price_i = be_u32(&buf[i..i+4]) as f64; i += 4;
                let price = to_price_scaled(price_i * 1e-4, tick_size);
                let d = to_canonical(py, 0, if side == 'B' { "LO+" } else { "LO-" }, shares, price, Some(&side.to_string()), symbol, "OUCH")?;
                d.set_item("token", token)?;
                out.push(d.into());
            }
            // 'O' is a client-enter message in OUCH; server stream uses 'A' Accepted. Skip its fixed-length payload.
            'O' => {
                // Per OUCH 4.2 spec, "Enter Order" is 49 bytes.
                let need = 49;
                if i + need > buf.len() { break; }
                i += need;
            }
            // Replace (U): orig_token[14], new_token[14], shares[4], price[4], tif[4]
            'U' => {
                let need = 14 + 14 + 4 + 4 + 4;
                if i + need > buf.len() { break; }
                let tok_orig = String::from_utf8_lossy(&buf[i..i+14]).trim_end().to_string(); i += 14; // orig
                let tok_new = String::from_utf8_lossy(&buf[i..i+14]).trim_end().to_string(); i += 14; // new
                let new_shares = be_u32(&buf[i..i+4]) as f64; i += 4;
                let price_i = be_u32(&buf[i..i+4]) as f64; i += 4;
                i += 4; // tif
                let price = to_price_scaled(price_i * 1e-4, tick_size);
                let d = to_canonical(py, 0, "LO+", new_shares, price, None, symbol, "OUCH")?;
                d.set_item("token_orig", tok_orig)?;
                d.set_item("token_new", tok_new)?;
                out.push(d.into());
            }
            // Cancel (X): token[14], shares[4]
            'X' => {
                let need = 14 + 4;
                if i + need > buf.len() { break; }
                let tok = String::from_utf8_lossy(&buf[i..i+14]).trim_end().to_string(); i += 14; // token
                let canceled = be_u32(&buf[i..i+4]) as f64; i += 4;
                let d = to_canonical(py, 0, "CX+", canceled, f64::NAN, None, symbol, "OUCH")?;
                d.set_item("token", tok)?;
                out.push(d.into());
            }
            // Executed (E): token[14], shares[4], match[8]
            'E' => {
                let need = 14 + 4 + 8;
                if i + need > buf.len() { break; }
                let tok = String::from_utf8_lossy(&buf[i..i+14]).trim_end().to_string(); i += 14; // token
                let executed = be_u32(&buf[i..i+4]) as f64; i += 4;
                let match_id = u64::from_be_bytes([buf[i],buf[i+1],buf[i+2],buf[i+3],buf[i+4],buf[i+5],buf[i+6],buf[i+7]]); i += 8; // match
                let d = to_canonical(py, 0, "MO+", executed, f64::NAN, None, symbol, "OUCH")?;
                d.set_item("token", tok)?;
                d.set_item("match", match_id)?;
                out.push(d.into());
            }
            // Trade (T): token[14], shares[4], price[4], match[8]
            'T' => {
                let need = 14 + 4 + 4 + 8;
                if i + need > buf.len() { break; }
                let tok = String::from_utf8_lossy(&buf[i..i+14]).trim_end().to_string(); i += 14; // token
                let executed = be_u32(&buf[i..i+4]) as f64; i += 4;
                let price_i = be_u32(&buf[i..i+4]) as f64; i += 4;
                let match_id = u64::from_be_bytes([buf[i],buf[i+1],buf[i+2],buf[i+3],buf[i+4],buf[i+5],buf[i+6],buf[i+7]]); i += 8; // match
                let price = to_price_scaled(price_i * 1e-4, tick_size);
                let d = to_canonical(py, 0, "MO+", executed, price, None, symbol, "OUCH")?;
                d.set_item("token", tok)?;
                d.set_item("match", match_id)?;
                out.push(d.into());
            }
            // Canceled (C): token[14], shares[4], reason[1]
            'C' => {
                let need = 14 + 4 + 1;
                if i + need > buf.len() { break; }
                let tok = String::from_utf8_lossy(&buf[i..i+14]).trim_end().to_string(); i += 14;
                let canceled = be_u32(&buf[i..i+4]) as f64; i += 4;
                let reason = buf[i] as char; i += 1; // reason
                let d = to_canonical(py, 0, "CX+", canceled, f64::NAN, None, symbol, "OUCH")?;
                d.set_item("token", tok)?;
                d.set_item("reason", reason.to_string())?;
                out.push(d.into());
            }
            // Priority Update (P): token[14], new_price[4], new_shares[4]
            'P' => {
                let need = 14 + 4 + 4;
                if i + need > buf.len() { break; }
                // Consume but do not emit canonical event (book metadata only)
                i += need;
            }
            // System Cancel (D): token[14]
            'D' => {
                let need = 14;
                if i + need > buf.len() { break; }
                let tok = String::from_utf8_lossy(&buf[i..i+14]).trim_end().to_string(); i += 14; // token
                // Map to cancel with zero size
                let d = to_canonical(py, 0, "CX+", 0.0, f64::NAN, None, symbol, "OUCH")?;
                d.set_item("token", tok)?;
                d.set_item("reason", "SYSTEM_CANCEL")?;
                out.push(d.into());
            }
            // Trade Correction (B): token[14], match_number[8], reason[1]
            'B' => {
                let need = 14 + 8 + 1;
                if i + need > buf.len() { break; }
                let tok = String::from_utf8_lossy(&buf[i..i+14]).trim_end().to_string(); i += 14; // token
                let match_id = u64::from_be_bytes([buf[i],buf[i+1],buf[i+2],buf[i+3],buf[i+4],buf[i+5],buf[i+6],buf[i+7]]); i += 8;  // match number
                let reason = buf[i] as char; i += 1;  // reason
                // Emit a neutralizing event to reflect correction (no size impact in canonical)
                let d = to_canonical(py, 0, "CX+", 0.0, f64::NAN, None, symbol, "OUCH")?;
                d.set_item("token", tok)?;
                d.set_item("match", match_id)?;
                d.set_item("reason", reason.to_string())?;
                out.push(d.into());
            }
            // Rejected (R): token[14], reason[1]
            'R' => {
                let need = 14 + 1;
                if i + need > buf.len() { break; }
                let tok = String::from_utf8_lossy(&buf[i..i+14]).trim_end().to_string(); i += 14;
                let reason = buf[i] as char; i += 1; // reason
                let d = to_canonical(py, 0, "CX+", 0.0, f64::NAN, None, symbol, "OUCH")?;
                d.set_item("token", tok)?;
                d.set_item("reason", reason.to_string())?;
                out.push(d.into());
            }
            _ => { /* Should not be reached if is_type is comprehensive */ }
        }
    }
    Ok(PyList::new(py, out).into())
}
#[pymodule]
fn torpedocode_ingest(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(parse_itch_file, m)?)?;
    m.add_function(wrap_pyfunction!(parse_ouch_file, m)?)?;
    Ok(())
}
