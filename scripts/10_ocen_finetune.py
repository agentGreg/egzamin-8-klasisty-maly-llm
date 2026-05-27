"""Faza 10: dekompozycja forma-vs-matematyka, baza vs fine-tune.

Liczy dla pary (base, ft):
- format_compliance (zamknięte, ścisłe)
- conditional_accuracy (zamknięte, na sparsowanych)
- total_score (z ocena_szczegolowa.json jeśli jest)
Wynik: results/finetune_decomposition.json + tabela na stdout.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.metryki_format import format_compliance, conditional_accuracy  # noqa: E402
KLUCZ = ROOT / "data" / "klucz_odpowiedzi.json"
OCENA = ROOT / "results" / "ocena_szczegolowa.json"
ZADANIA = ROOT / "data" / "zadania.json"
OUT = ROOT / "results" / "finetune_decomposition.json"

PARY = [
    ("PLLuM-8B base", "results/pllum8_2512_odpowiedzi.json", "llama_pllum8_2512"),
    ("PLLuM-8B +LoRA", "results/pllum8_2512_ft_odpowiedzi.json", "llama_pllum8_2512_ft"),
]


def closed_rows_and_ids(path: Path):
    data = json.load(open(path, encoding="utf-8"))
    rows = [r for r in data if r["typ"] == "zamkniete"]
    ids = [r["id"] for r in rows]
    return rows, ids


def main() -> None:
    klucz = {k["id"]: k["odpowiedz"] for k in json.load(open(KLUCZ, encoding="utf-8"))}
    totals = {}
    if OCENA.exists():
        oc = json.load(open(OCENA, encoding="utf-8"))
        totals = {m: oc[m]["suma"] for m in oc.get("modele", []) if m in oc}

    out = []
    for label, rel, judge_key in PARY:
        rows, ids = closed_rows_and_ids(ROOT / rel)
        out.append({
            "model": label,
            "format_compliance": round(format_compliance(rows), 3),
            "conditional_accuracy": round(conditional_accuracy(rows, ids, klucz), 3),
            "total_score": totals.get(judge_key),
        })

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{'model':16} {'format':>8} {'cond.acc':>9} {'score/30':>9}")
    for r in out:
        s = "-" if r["total_score"] is None else r["total_score"]
        print(f"{r['model']:16} {r['format_compliance']:>8} {r['conditional_accuracy']:>9} {str(s):>9}")
    print(f"OK: {OUT}")


if __name__ == "__main__":
    main()
