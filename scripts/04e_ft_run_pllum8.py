"""Faza 4e-FT: Llama-PLLuM-8B-instruct-2512 + adapter LoRA (spike) na 20 zadaniach.

Wariant fine-tune: ten sam model bazowy co 04e, ale z doczepionym adapterem LoRA
z adapters/pllum8_spike (trening na arkuszach 2021-2023). Test = arkusz 2026 (held-out).
Wynik: results/pllum8_2512_ft_odpowiedzi.json
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = str(Path.home() / ".cache/huggingface/local-mlx/Llama-PLLuM-8B-instruct-2512-mlx-8bit")
ADAPTER = str(ROOT / "adapters" / "pllum8_spike")

ZADANIA = ROOT / "data" / "zadania.json"
SYS_ZAMK = (ROOT / "prompts" / "system_zamkniete.txt").read_text(encoding="utf-8")
SYS_OTW = (ROOT / "prompts" / "system_otwarte.txt").read_text(encoding="utf-8")
OUT = ROOT / "results" / "pllum8_2512_ft_odpowiedzi.json"


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
    print(f"Ładuję {MODEL_ID} + adapter {ADAPTER}...")
    model, tokenizer = load(MODEL_ID, adapter_path=ADAPTER)
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
