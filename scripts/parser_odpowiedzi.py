"""Współdzielony parser odpowiedzi (przeniesiony z 05_ocen.py).

- normalize: normalizacja do porównań (usuwa whitespace, upper).
- extract_odpowiedz: ścisłe wyłuskanie z tagu <odpowiedz>...</odpowiedz>.
- wylusk_zamkniete: hojne wyłuskanie A/B/C/D (i PF/kombo) z dowolnego tekstu.
"""
from __future__ import annotations

import re


def normalize(s: str) -> str:
    return re.sub(r"\s+", "", s.strip().upper())


def extract_odpowiedz(text: str) -> str:
    m = re.search(r"<odpowiedz>\s*(.*?)\s*</odpowiedz>", text, re.S | re.I)
    return m.group(1).strip() if m else ""


def wylusk_zamkniete(raw: str, odp: str) -> str:
    """Wyłuska odpowiedź A/B/C/D (lub kombo AC/BD/PF) z tekstu.

    Celowo hojny: poza polem odpowiedz → \\boxed{} → "Odpowiedź: X" / **X" łapie też
    format GSM8K (`#### X`) oraz odpowiedzi opisowe ("D. 0,6x+0,8y"), które niektóre
    modele (PLLuM) zwracają, ignorując instrukcję <odpowiedz>. Zwalidowane na 8 modelach
    (2026-05-27): monotoniczny — żaden model nie traci punktów; jedyny realny zysk to
    PLLuM-12B zad. 5. Niski wynik PLLuM (3-4/30) NIE jest artefaktem parsowania.
    """
    if odp and re.fullmatch(r"[A-DPF]{1,2}", odp.strip().upper()):
        return odp.strip().upper()

    boxed = re.findall(r"\\boxed\{([^{}]+)\}", raw)
    if boxed:
        cand = boxed[-1].strip().upper()
        m = re.search(r"[A-DPF]{1,2}", cand)
        if m:
            return m.group(0)

    for pat in [
        r"(?:Odpowied[zź]|Final|Wynik)[^\w]*\**\s*([A-DPF]{1,2})\b",
        r"\*\*([A-DPF]{1,2})\*\*\s*$",
        r"\b([A-DPF]{1,2})\b\s*\.?\s*$",
    ]:
        m = re.search(pat, raw, re.I | re.M)
        if m:
            return m.group(1).upper()

    # Hojne fallbacki — gdy model olał format <odpowiedz> (np. styl GSM8K).
    for pat in [
        r"####\s*([A-DPF]{1,2})\b",
        r"(?:odpowied[zź]\s*to|wynik|final\w*)\s*[:\-]?\s*\**\s*([A-DPF]{1,2})\b",
    ]:
        m = re.search(pat, raw, re.I)
        if m:
            return m.group(1).upper()
    enum = re.findall(r"(?m)^\s*([A-DPF]{1,2})[.):]", raw)  # ostatnia linia typu "D. ..."
    if enum:
        return enum[-1].upper()

    return (odp or "").strip().upper()
