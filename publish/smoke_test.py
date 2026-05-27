"""
Smoke test każdej skonwertowanej wersji MLX.

Dla każdego MLX folderu:
- Załaduj przez mlx_lm.load (potwierdza że format jest poprawny)
- Wygeneruj 200 tokenów na zadaniu 1 z benchmarku (potwierdza że generuje sensowny PL)
- Wypisz first/last 100 znaków + tokens/s

Wynik: pass/fail per wariant + raport.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

ROOT = Path(__file__).resolve().parents[1]
ZADANIA = ROOT / "data" / "zadania.json"
OUT_BASE = Path.home() / ".cache" / "huggingface" / "local-mlx"

WARIANTY = [
    ("Bielik-Minitron-7B-mlx-4bit",  "4bit"),
    ("Bielik-Minitron-7B-mlx-6bit",  "6bit"),
    ("Bielik-Minitron-7B-mlx-8bit",  "8bit"),
    ("Bielik-Minitron-7B-mlx-bf16",  "bf16"),
]

SYSTEM = """Jesteś uczniem 8 klasy szkoły podstawowej zdającym egzamin ósmoklasisty z matematyki. Rozwiąż zadanie krok po kroku, na końcu podaj wynik w bloku <odpowiedz>X</odpowiedz>."""


def test_wariant(sufix: str, label: str, zadanie: dict) -> dict:
    path = OUT_BASE / sufix
    if not path.exists():
        return {"label": label, "status": "BRAK", "error": "katalog nie istnieje"}

    print(f"\n=== {label} ({path}) ===")
    try:
        model, tokenizer = load(str(path))
    except Exception as e:
        return {"label": label, "status": "LOAD_FAIL", "error": str(e)[:200]}

    user = f"{zadanie['tresc']}\n\nOpcje:\n" + "\n".join(
        f"  {k}. {v}" for k, v in zadanie["opcje"].items()
    )
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": user},
    ]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    t0 = time.perf_counter()
    text = generate(
        model, tokenizer,
        prompt=prompt,
        max_tokens=400,
        sampler=make_sampler(temp=0.0),
        verbose=False,
    )
    dt = time.perf_counter() - t0
    toks = len(tokenizer.encode(text))
    tps = toks / dt

    has_odp = "<odpowiedz>" in text.lower()
    snippet = text[:200].replace("\n", " ") + "..."
    print(f"  {toks} tok, {dt:.1f}s, {tps:.0f} tok/s")
    print(f"  tag <odpowiedz>: {'TAK' if has_odp else 'NIE'}")
    print(f"  {snippet}")

    return {
        "label": label,
        "status": "OK" if has_odp else "WARN_NO_TAG",
        "tokens": toks,
        "seconds": round(dt, 1),
        "tps": round(tps, 0),
        "snippet": snippet,
    }


def main() -> None:
    zadania = json.loads(ZADANIA.read_text(encoding="utf-8"))
    zadanie_1 = next(z for z in zadania if z["id"] == 1)

    wyniki = []
    for sufix, label in WARIANTY:
        wyniki.append(test_wariant(sufix, label, zadanie_1))

    print("\n=== PODSUMOWANIE ===")
    for w in wyniki:
        status = w["status"]
        if status == "OK":
            print(f"  ✓ {w['label']:5}  {w['tps']:>4} tok/s")
        elif status == "WARN_NO_TAG":
            print(f"  ⚠ {w['label']:5}  {w['tps']:>4} tok/s (brak <odpowiedz>, ale model działa)")
        else:
            print(f"  ✗ {w['label']:5}  {status}: {w.get('error', '')}")

    bad = [w for w in wyniki if w["status"] in ("LOAD_FAIL", "BRAK")]
    if bad:
        sys.exit(f"\nUWAGA: {len(bad)} wariantów nie przeszło smoke testu")


if __name__ == "__main__":
    main()
