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
    // Optional: session hours for downstream alignment
    #[serde(skip_serializing_if = "Option::is_none")]
    session_start: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    session_end: Option<String>,
}

#[derive(Debug, Default)]
struct Opts {
    input: String,
    by: Vec<String>,
    by_market: bool,
    output: Option<String>,
    tick_quant: Option<f64>,
    per_market_deciles: bool,
    session_hours: Option<String>,
}

fn print_help() {
    println!("torpedocode-panel {}", env!("CARGO_PKG_VERSION"));
    println!("Build liquidity panel and instrument match groups\n");
    println!("USAGE:");
    println!("  torpedocode-panel --input FILE [--by COL ...] [--by-market] [--output PATH]\\");
    println!("                  [--tick-quant F] [--per-market-deciles] [--session-hours FILE]\n");
    println!("FLAGS:");
    println!("  --help                   Show this help and exit");
    println!("  --by-market              Preset grouping: market + liq_decile + tick_size");
    println!("  --per-market-deciles     Compute liquidity deciles per market");
    println!("OPTIONS:");
    println!("  --input FILE             Input CSV with columns market,symbol,median_daily_notional,tick_size");
    println!("  --by COL ...             Grouping columns for match groups (e.g., liq_decile tick_size)");
    println!("  --output PATH            Output path (.json for JSON, else CSV)");
    println!("  --tick-quant F           Quantize tick_size to nearest multiple of F");
    println!("  --session-hours FILE     CSV with market,session_start,session_end to annotate rows");
}

fn parse_args() -> Opts {
    let mut opts = Opts::default();
    // Default grouping
    let mut by: Vec<String> = vec!["market".into(), "liq_decile".into(), "tick_size".into()];
    let args: Vec<String> = env::args().collect();
    if args.len() == 1 || args.iter().any(|a| a == "--help" || a == "-h") {
        print_help();
        std::process::exit(if args.len() == 1 { 1 } else { 0 });
    }
    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "--input" => { i += 1; opts.input = args.get(i).cloned().unwrap_or_default(); }
            "--by-market" => { opts.by_market = true; }
            "--by" => {
                i += 1;
                let mut cols = Vec::new();
                while i < args.len() && !args[i].starts_with("--") { cols.push(args[i].clone()); i += 1; }
                i -= 1;
                if !cols.is_empty() { by = cols; }
            }
            "--output" => { i += 1; opts.output = args.get(i).cloned(); }
            "--tick-quant" => { i += 1; if let Some(s) = args.get(i) { opts.tick_quant = s.parse::<f64>().ok(); } }
            "--per-market-deciles" => { opts.per_market_deciles = true; }
            "--session-hours" => { i += 1; opts.session_hours = args.get(i).cloned(); }
            _ => {}
        }
        i += 1;
    }
    if opts.by_market { opts.by = vec!["market".into(), "liq_decile".into(), "tick_size".into()]; }
    else { opts.by = by; }
    if opts.input.is_empty() { eprintln!("--input is required. Use --help for usage."); std::process::exit(1); }
    opts
}

fn compute_liquidity_deciles(rows: &mut [Row], per_market: bool) {
    if per_market {
        // group by market
        let mut groups: HashMap<String, Vec<usize>> = HashMap::new();
        for (i, r) in rows.iter().enumerate() {
            groups.entry(r.market.clone()).or_default().push(i);
        }
        for idxs in groups.values() {
            let mut idx = idxs.clone();
            idx.sort_by(|&a, &b| rows[a]
                .median_daily_notional
                .partial_cmp(&rows[b].median_daily_notional)
                .unwrap_or(std::cmp::Ordering::Equal));
            let n = idx.len().max(1);
            for (rank, &i) in idx.iter().enumerate() {
                let q = (rank as f64) / (n as f64);
                let dec = (q * 10.0).floor().min(9.0).max(0.0) as u8;
                rows[i].liq_decile = Some(dec);
            }
        }
    } else {
        // global deciles
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
    let opts = parse_args();
    let by = opts.by.clone();
    let mut rdr = ReaderBuilder::new().from_path(&opts.input)?;
    let mut rows: Vec<Row> = Vec::new();
    for result in rdr.deserialize() {
        let mut row: Row = result?;
        // Ensure NaN replaced with 0 for notional
        if !row.median_daily_notional.is_finite() {
            row.median_daily_notional = 0.0;
        }
        // Optional tick quantisation
        if let (Some(q), Some(t)) = (opts.tick_quant, row.tick_size) {
            if q > 0.0 {
                let quant = (t / q).round() * q;
                row.tick_size = Some(quant);
            }
        }
        rows.push(row);
    }
    // Optional session hours mapping from CSV
    if let Some(path) = opts.session_hours.as_ref() {
        let mut m: HashMap<String, (String, String)> = HashMap::new();
        let mut r2 = ReaderBuilder::new().from_path(&path)?;
        for rec in r2.records() {
            let rec = rec?;
            if rec.len() >= 3 {
                m.insert(rec[0].to_string(), (rec[1].to_string(), rec[2].to_string()));
            }
        }
        for r in rows.iter_mut() {
            if let Some((s, e)) = m.get(&r.market) {
                r.session_start = Some(s.clone());
                r.session_end = Some(e.clone());
            }
        }
    }

    compute_liquidity_deciles(&mut rows, opts.per_market_deciles);
    match_groups(&mut rows, &by);

    if let Some(out) = opts.output.as_ref() {
        let path = Path::new(out);
        if let Some(ext) = path.extension().and_then(|s| s.to_str()) {
            if ext.eq_ignore_ascii_case("json") {
                let json = serde_json::to_string_pretty(&rows)?;
                std::fs::write(path, json)?;
            } else {
                let mut wtr = WriterBuilder::new().from_path(path)?;
                wtr.serialize(("market", "symbol", "median_daily_notional", "tick_size", "liq_decile", "match_group", "session_start", "session_end"))?;
                for r in rows {
                    wtr.serialize((r.market, r.symbol, r.median_daily_notional, r.tick_size, r.liq_decile, r.match_group, r.session_start, r.session_end))?;
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
