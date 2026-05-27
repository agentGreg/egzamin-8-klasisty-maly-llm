"""Parametrized inferencja na dowolnym arkuszu testowym (MLX, text-only)."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

ROOT = Path(__file__).resolve().parents[1]
SYS_ZAMK = (ROOT / "prompts" / "system_zamkniete.txt").read_text(encoding="utf-8")
SYS_OTW = (ROOT / "prompts" / "system_otwarte.txt").read_text(encoding="utf-8")


def format_opcje(opcje: dict[str, str]) -> str:
    return "\n".join(f"  {k}. {v}" for k, v in opcje.items()) if opcje else ""


def build_user_prompt(z: dict) -> str:
    parts = [z["tresc"]]
    if z.get("opis_obrazka"):
        parts.append(f"\nOpis rysunku do zadania:\n{z['opis_obrazka']}")
    if z["typ"] == "zamkniete" and z.get("opcje"):
        parts.append("\nOpcje odpowiedzi:")
        parts.append(format_opcje(z["opcje"]))
    return "\n".join(parts)


def extract_odpowiedz(text: str) -> str:
    m = re.search(r"<odpowiedz>\s*(.*?)\s*</odpowiedz>", text, re.S | re.I)
    return m.group(1).strip() if m else ""


def run_sheet(model_id: str, adapter_path: str | None, zadania_path: Path, out_path: Path) -> None:
    print(f"Ładuję {model_id}" + (f" + adapter {adapter_path}" if adapter_path else ""))
    model, tokenizer = load(model_id, adapter_path=adapter_path) if adapter_path else load(model_id)
    zadania = json.loads(Path(zadania_path).read_text(encoding="utf-8"))
    sampler = make_sampler(temp=0.0)
    wyniki = []
    for z in zadania:
        sys_prompt = SYS_ZAMK if z["typ"] == "zamkniete" else SYS_OTW
        messages = [{"role": "system", "content": sys_prompt},
                    {"role": "user", "content": build_user_prompt(z)}]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        t0 = time.perf_counter()
        text = generate(model, tokenizer, prompt=prompt, max_tokens=1500, sampler=sampler, verbose=False)
        dt = time.perf_counter() - t0
        odp = extract_odpowiedz(text)
        print(f"  z{z['id']:02d}: {dt:5.1f}s odp={odp[:30]!r}")
        wyniki.append({"id": z["id"], "typ": z["typ"], "punkty_max": z["punkty_max"],
                       "czas_s": round(dt, 2), "raw": text, "odpowiedz": odp})
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(wyniki, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"OK: {out_path}")
