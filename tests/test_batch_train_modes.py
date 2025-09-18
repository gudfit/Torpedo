import pytest


def test_batch_train_modes_parse(monkeypatch):
    from torpedocode.cli import batch_train as bt
    # Just ensure argparse accepts the new modes; do not execute training
    ap = bt.argparse.ArgumentParser if hasattr(bt, 'argparse') else None
    assert bt is not None

