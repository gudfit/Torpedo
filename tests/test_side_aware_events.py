import numpy as np
import pandas as pd
from pathlib import Path

from torpedocode.data.lobster import parse_lobster_pair, LOBSTERParseConfig


def test_lobster_side_aware_mapping(tmp_path: Path):
    # Minimal CSVs with one message row and stub orderbook
    msg = tmp_path / "msg.csv"
    ob = tmp_path / "ob.csv"
    # time,type,order_id,size,price,direction
    msg.write_text("0.0,3,1,100,10.0,-1\n")  # type=3 cancel, direction=-1 (sell)
    # orderbook columns: build 1 level
    ob.write_text("1.0,10.0,1.0,10.0\n")

    df_na = parse_lobster_pair(msg, ob, cfg=LOBSTERParseConfig(symbol="X", side_aware=False))
    assert df_na.loc[0, "event_type"].startswith("CX") and df_na.loc[0, "event_type"] == "CX+"

    df_sa = parse_lobster_pair(msg, ob, cfg=LOBSTERParseConfig(symbol="X", side_aware=True))
    assert df_sa.loc[0, "event_type"] == "CX-"  # sell-side cancel

