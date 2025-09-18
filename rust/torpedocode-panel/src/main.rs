use csv::{ReaderBuilder, WriterBuilder};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;
use std::path::Path;

#[derive(Debug, Deserialize, Serialize, Clone)]
struct Row {
    #[serde(default)]
    market: String,
    #[serde(default)]
    symbol: String,
    #[serde(default, rename = "median_daily_notional")]
    median_daily_notional: f64,
    #[serde(default)]
    tick_size: Option<f64>,
    // Optional fields computed by the tool; include in JSON when present
    #[serde(skip_serializing_if = "Option::is_none")]
    liq_decile: Option<u8>,
    #[serde(skip_serializing_if = "Option::is_none")]
    match_group: Option<u64>,
}

fn parse_args() -> (String, Vec<String>, Option<String>) {
    let mut input: Option<String> = None;
    let mut by: Vec<String> = vec!["liq_decile".into(), "tick_size".into()];
    let mut output: Option<String> = None;
    let mut i = 1;
    let args: Vec<String> = env::args().collect();
    while i < args.len() {
        match args[i].as_str() {
            "--input" => {
                i += 1;
                input = args.get(i).cloned();
            }
            "--by" => {
                i += 1;
                let mut cols = Vec::new();
                while i < args.len() && !args[i].starts_with("--") {
                    cols.push(args[i].clone());
                    i += 1;
                }
                i -= 1; // step back one because loop will add one
                if !cols.is_empty() {
                    by = cols;
                }
            }
            "--output" => {
                i += 1;
                output = args.get(i).cloned();
            }
            _ => {}
        }
        i += 1;
    }
    let inp = input.expect("--input is required");
    (inp, by, output)
}

fn compute_liquidity_deciles(rows: &mut [Row]) {
    // sort by notional and assign deciles 0..9 using rank positions
    let mut idx: Vec<usize> = (0..rows.len()).collect();
    idx.sort_by(|&a, &b| rows[a]
        .median_daily_notional
        .partial_cmp(&rows[b].median_daily_notional)
        .unwrap_or(std::cmp::Ordering::Equal));
    let n = rows.len().max(1);
    for (rank, &i) in idx.iter().enumerate() {
        let q = (rank as f64) / (n as f64);
        let dec = (q * 10.0).floor().min(9.0).max(0.0) as u8;
        rows[i].liq_decile = Some(dec);
    }
}

fn match_groups(rows: &mut [Row], by: &[String]) {
    // Create a grouping key by concatenating requested fields
    fn key_for(row: &Row, by: &[String]) -> String {
        let mut parts: Vec<String> = Vec::new();
        for c in by {
            let v = match c.as_str() {
                "liq_decile" => row.liq_decile.map(|x| x.to_string()).unwrap_or_default(),
                "tick_size" => row
                    .tick_size
                    .map(|x| format!("{:.8}", x))
                    .unwrap_or_default(),
                "market" => row.market.clone(),
                "symbol" => row.symbol.clone(),
                other => {
                    // unknown; ignore
                    let _ = other;
                    String::new()
                }
            };
            parts.push(v);
        }
        parts.join("|")
    }
    let mut map: HashMap<String, u64> = HashMap::new();
    let mut next: u64 = 0;
    for r in rows.iter_mut() {
        let k = key_for(r, by);
        let id = *map.entry(k).or_insert_with(|| {
            let v = next;
            next += 1;
            v
        });
        r.match_group = Some(id);
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (input, by, output) = parse_args();
    let mut rdr = ReaderBuilder::new().from_path(&input)?;
    let mut rows: Vec<Row> = Vec::new();
    for result in rdr.deserialize() {
        let mut row: Row = result?;
        // Ensure NaN replaced with 0 for notional
        if !row.median_daily_notional.is_finite() {
            row.median_daily_notional = 0.0;
        }
        rows.push(row);
    }
    compute_liquidity_deciles(&mut rows);
    match_groups(&mut rows, &by);

    if let Some(out) = output {
        let path = Path::new(&out);
        if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
            if ext.eq_ignore_ascii_case("json") {
                let json = serde_json::to_string_pretty(&rows)?;
                std::fs::write(path, json)?;
            } else {
                let mut wtr = WriterBuilder::new().from_path(path)?;
                wtr.serialize(("market", "symbol", "median_daily_notional", "tick_size", "liq_decile", "match_group"))?;
                for r in rows {
                    wtr.serialize((r.market, r.symbol, r.median_daily_notional, r.tick_size, r.liq_decile, r.match_group))?;
                }
                wtr.flush()?;
            }
        } else {
            // default JSON
            let json = serde_json::to_string_pretty(&rows)?;
            std::fs::write(path, json)?;
        }
    } else {
        let json = serde_json::to_string_pretty(&rows)?;
        println!("{}", json);
    }
    Ok(())
}
