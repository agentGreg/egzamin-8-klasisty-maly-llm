"""
Faza 2: klucz odpowiedzi (wzorzec) generowany przez Claude Opus 4.7.

Wejście: data/zadania.json (treści, opcje, opisy obrazków).
Wyjście: data/klucz_odpowiedzi.json — dla każdego zadania:
  - odpowiedz (litera dla zamkniętych, krótki finalny wynik dla otwartych)
  - rozwiazanie (krok po kroku, służy potem jako WZORZEC do oceny)

Używamy modelu claude-opus-4-7 z 1M kontekstu (alias 1M jest w model ID).
PDF arkusza wrzucamy jako dokument (cache 1h), żeby Claude widział oryginalne
rysunki / wzory — to istotne dla zadań 17, 18, 20.
"""

from __future__ import annotations

import base64
import json
import os
import re
import sys
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

if not os.environ.get("ANTHROPIC_API_KEY"):
    sys.exit("Brak ANTHROPIC_API_KEY w .env")

ZADANIA = ROOT / "data" / "zadania.json"
PDF = ROOT / "matematyka-2026-egzamin-osmoklasisty.pdf"
OUT = ROOT / "data" / "klucz_odpowiedzi.json"

MODEL = "claude-opus-4-7"  # 1M context wariant

PROMPT_SYSTEM = """Jesteś doświadczonym nauczycielem matematyki przygotowującym wzorcowy klucz
odpowiedzi do egzaminu ósmoklasisty z matematyki. Załączony PDF zawiera oficjalny arkusz CKE.

Rozwiążesz wszystkie 20 zadań. Dla każdego podaj:
- "id": numer zadania
- "odpowiedz": dla zadań zamkniętych (1-14) — wybrana opcja, np. "C" lub "AD" lub "PF"
              dla zadań otwartych (15-20) — krótkie pełne zdanie z wynikiem końcowym
- "rozwiazanie": rozwiązanie krok po kroku (kilka linii, jasne obliczenia)

Bądź bezbłędny — to klucz, według którego ocenimy modele open-source.
"""

PROMPT_USER = """Rozwiąż wszystkie 20 zadań z załączonego arkusza egzaminacyjnego.

Zwróć WYŁĄCZNIE JSON, listę 20 obiektów, każdy o strukturze:
{"id": int, "odpowiedz": str, "rozwiazanie": str}

Bez żadnego tekstu wprowadzającego, bez bloków kodu markdown."""


def main() -> None:
    client = Anthropic()

    pdf_bytes = PDF.read_bytes()
    pdf_b64 = base64.standard_b64encode(pdf_bytes).decode()

    print(f"Wysyłam zapytanie do {MODEL} z PDF ({len(pdf_bytes) / 1024:.0f} KB)...")
    msg = client.messages.create(
        model=MODEL,
        max_tokens=16000,
        system=PROMPT_SYSTEM,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": pdf_b64,
                        },
                        "cache_control": {"type": "ephemeral"},
                    },
                    {"type": "text", "text": PROMPT_USER},
                ],
            }
        ],
    )

    raw = msg.content[0].text
    print(f"Tokeny: input={msg.usage.input_tokens}, output={msg.usage.output_tokens}")

    # Czasem model owinie odpowiedź w ```json ... ```; spróbujmy wyłuskać
    m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", raw, re.S)
    payload = m.group(1) if m else raw.strip()
    klucz = json.loads(payload)

    if len(klucz) != 20:
        sys.exit(f"Klucz ma {len(klucz)} zadań, oczekiwano 20")

    OUT.write_text(json.dumps(klucz, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"OK: zapisano {OUT}")

    # podsumowanie
    for k in klucz:
        odp = k["odpowiedz"]
        if len(odp) > 80:
            odp = odp[:77] + "..."
        print(f"  z{k['id']:02d}: {odp}")


if __name__ == "__main__":
    main()
