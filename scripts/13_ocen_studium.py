"""Faza 13: dekompozycja forma-vs-matematyka, oba modele, pula 2024+2025+2026."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.metryki_format import strict_compliant  # noqa: E402
from scripts.parser_odpowiedzi import normalize, wylusk_zamkniete  # noqa: E402

SHEETS = {
    "2024": ROOT / "data/testy/2024.json",
    "2025": ROOT / "data/testy/2025.json",
    "2026": ROOT / "data/zadania.json",
}
KEY_2026 = ROOT / "data/klucz_odpowiedzi.json"
STUD = ROOT / "results/studium"
OUT = ROOT / "results/studium_decomposition.json"
MODELS = [("Llama-PLLuM 8B", "pllum_base", "pllum_ft"),
          ("Bielik-Minitron 7B", "bielik_base", "bielik_ft")]


def sheet_key(sheet: str) -> dict[int, str]:
    if sheet == "2026":
        return {k["id"]: k["odpowiedz"] for k in json.load(open(KEY_2026, encoding="utf-8"))}
    return {z["id"]: z["odpowiedz"] for z in json.load(open(SHEETS[sheet], encoding="utf-8"))}


def aggregate(per_sheet: list[tuple[list[dict], dict[int, str]]]):
    """per_sheet: list of (closed_rows, key). Returns (format_compliance, cond_acc, n_closed).

    Scores each sheet against its OWN key inline and pools the counts — pooling
    rows into one id→key dict would collide (every sheet has an id=1).
    """
    compliant = parsed = correct = total = 0
    for rows, key in per_sheet:
        for r in rows:
            total += 1
            if strict_compliant(r["raw"], r.get("odpowiedz", "")):
                compliant += 1
            guess = wylusk_zamkniete(r["raw"], r.get("odpowiedz", ""))
            if guess:
                parsed += 1
                if guess == normalize(key[r["id"]]):
                    correct += 1
    fmt = compliant / total if total else 0.0
    acc = correct / parsed if parsed else 0.0
    return fmt, acc, total


def closed_rows(model_tag: str, sheet: str) -> list[dict]:
    data = json.load(open(STUD / f"{model_tag}_{sheet}.json", encoding="utf-8"))
    return [r for r in data if r["typ"] == "zamkniete"]


def main() -> None:
    out = []
    for label, base_tag, ft_tag in MODELS:
        for variant, tag in [("base", base_tag), ("+LoRA", ft_tag)]:
            per_sheet = [(closed_rows(tag, s), sheet_key(s)) for s in SHEETS]
            fmt, acc, n = aggregate(per_sheet)
            out.append({"model": label, "variant": variant,
                        "format_compliance": round(fmt, 3),
                        "conditional_accuracy": round(acc, 3), "n_closed": n})
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{'model':20} {'variant':7} {'format':>7} {'cond.acc':>9} {'n':>4}")
    for r in out:
        print(f"{r['model']:20} {r['variant']:7} {r['format_compliance']:>7} {r['conditional_accuracy']:>9} {r['n_closed']:>4}")
    print(f"OK: {OUT}")


if __name__ == "__main__":
    main()
