import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("os13", ROOT / "scripts" / "13_ocen_studium.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_aggregate_pools_closed_across_sheets():
    m = _load()
    per_sheet = [
        ([{"id": 1, "typ": "zamkniete", "raw": "<odpowiedz>A</odpowiedz>", "odpowiedz": "A"}], {1: "A"}),
        ([{"id": 1, "typ": "zamkniete", "raw": "<odpowiedz>B</odpowiedz>", "odpowiedz": "B"}], {1: "C"}),
    ]
    fmt, acc, n = m.aggregate(per_sheet)
    assert n == 2            # 2 closed tasks pooled
    assert fmt == 1.0        # both strict-compliant
    assert acc == 0.5        # 1 of 2 correct
