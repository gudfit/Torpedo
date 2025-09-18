import sys


def test_benchmark_runs_quick(monkeypatch, capsys):
    from torpedocode.bench import benchmark

    argv = [
        "prog",
        "--levels",
        "10",
        "--T",
        "100",
        "--batch",
        "8",
        "--window-s",
        "1",
        "--stride",
        "2",
        "--repeats",
        "1",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    benchmark.main()
    out = capsys.readouterr().out
    assert "ph_time_median_s" in out

