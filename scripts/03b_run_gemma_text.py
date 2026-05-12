"""
Faza 3c: uruchomienie Gemma 3 4B IT (TEXT-ONLY przez mlx-lm) na 20 zadaniach.

Identyczny pipeline jak 04_run_bielik.py — dla uczciwego porównania.
Gemma nie widzi obrazków, dostaje tylko `opis_obrazka` w tekście.

Wynik: results/gemma_text_odpowiedzi.json
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "mlx-community/gemma-3-4b-it-4bit"

ZADANIA = ROOT / "data" / "zadania.json"
SYS_ZAMK = (ROOT / "prompts" / "system_zamkniete.txt").read_text(encoding="utf-8")
SYS_OTW = (ROOT / "prompts" / "system_otwarte.txt").read_text(encoding="utf-8")
OUT = ROOT / "results" / "gemma_text_odpowiedzi.json"


def format_opcje(opcje: dict[str, str]) -> str:
    if not opcje:
        return ""
    return "\n".join(f"  {k}. {v}" for k, v in opcje.items())


def build_user_prompt(zadanie: dict) -> str:
    parts = [zadanie["tresc"]]
    if zadanie.get("opis_obrazka"):
        parts.append(f"\nOpis rysunku do zadania:\n{zadanie['opis_obrazka']}")
    if zadanie["typ"] == "zamkniete" and zadanie.get("opcje"):
        parts.append("\nOpcje odpowiedzi:")
        parts.append(format_opcje(zadanie["opcje"]))
    return "\n".join(parts)


def extract_odpowiedz(text: str) -> str:
    m = re.search(r"<odpowiedz>\s*(.*?)\s*</odpowiedz>", text, re.S | re.I)
    return m.group(1).strip() if m else ""


def main() -> None:
    print(f"Ładuję {MODEL_ID} (mlx-lm, text-only)...")
    model, tokenizer = load(MODEL_ID)
    print("Model załadowany")

    zadania = json.loads(ZADANIA.read_text(encoding="utf-8"))
    wyniki = []
    sampler = make_sampler(temp=0.0)

    for z in zadania:
        sys_prompt = SYS_ZAMK if z["typ"] == "zamkniete" else SYS_OTW
        user_prompt = build_user_prompt(z)

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        t0 = time.perf_counter()
        text = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=1500,
            sampler=sampler,
            verbose=False,
        )
        dt = time.perf_counter() - t0

        toks = len(tokenizer.encode(text))
        tps = toks / dt if dt > 0 else None
        odp = extract_odpowiedz(text)

        print(f"  z{z['id']:02d}: {dt:5.1f}s, {toks} tok, odp={odp[:40]!r}")

        wyniki.append({
            "id": z["id"],
            "typ": z["typ"],
            "punkty_max": z["punkty_max"],
            "ma_obrazek": bool(z.get("obrazek")),
            "czas_s": round(dt, 2),
            "tokens": toks,
            "tokens_per_s": round(tps, 1) if tps else None,
            "raw": text,
            "odpowiedz": odp,
        })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(wyniki, ensure_ascii=False, indent=2), encoding="utf-8")
    suma_t = sum(w["czas_s"] for w in wyniki)
    print(f"\nOK: {OUT}, łączny czas {suma_t:.0f}s ({suma_t/60:.1f} min)")


if __name__ == "__main__":
    main()
