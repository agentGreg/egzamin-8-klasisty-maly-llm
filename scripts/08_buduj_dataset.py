"""Faza 8: budowa zbioru treningowego (MLX chat format) z data/archiwum/*.json.

Każda linia: {"messages": [system, user, assistant]}.
Cel asystenta = oficjalne rozwiązanie + <odpowiedz>...</odpowiedz>.
Split train/valid 90/10 deterministyczny (seed 42).
"""
from __future__ import annotations

import glob
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYS_ZAMK = (ROOT / "prompts" / "system_zamkniete.txt").read_text(encoding="utf-8")
SYS_OTW = (ROOT / "prompts" / "system_otwarte.txt").read_text(encoding="utf-8")
ARCHIWUM = ROOT / "data" / "archiwum"
OUT_DIR = ROOT / "data" / "finetune"


def format_opcje(opcje: dict[str, str]) -> str:
    if not opcje:
        return ""
    return "\n".join(f"  {k}. {v}" for k, v in opcje.items())


def build_user_prompt(z: dict) -> str:
    parts = [z["tresc"]]
    if z.get("opis_obrazka"):
        parts.append(f"\nOpis rysunku do zadania:\n{z['opis_obrazka']}")
    if z["typ"] == "zamkniete" and z.get("opcje"):
        parts.append("\nOpcje odpowiedzi:")
        parts.append(format_opcje(z["opcje"]))
    return "\n".join(parts)


def make_example(z: dict, sys_zamk: str, sys_otw: str) -> dict:
    sys_prompt = sys_zamk if z["typ"] == "zamkniete" else sys_otw
    target = f"{z['rozwiazanie'].strip()}\n<odpowiedz>{z['odpowiedz']}</odpowiedz>"
    return {
        "messages": [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": build_user_prompt(z)},
            {"role": "assistant", "content": target},
        ]
    }


def main() -> None:
    files = sorted(glob.glob(str(ARCHIWUM / "*.json")))
    if not files:
        raise SystemExit(f"Brak plików w {ARCHIWUM}")

    examples = []
    for f in files:
        for z in json.load(open(f, encoding="utf-8")):
            examples.append(make_example(z, SYS_ZAMK, SYS_OTW))

    random.Random(42).shuffle(examples)
    cut = max(1, int(len(examples) * 0.1))
    valid, train = examples[:cut], examples[cut:]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, rows in [("train", train), ("valid", valid)]:
        path = OUT_DIR / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name}: {len(rows)} -> {path}")


if __name__ == "__main__":
    main()
