"""Metryki rozdzielające formę od matematyki.

format_compliance — czy model odpowiada w wymaganym kształcie (bez klucza).
conditional_accuracy — poprawność WARUNKOWA na sparsowanych odpowiedziach.
"""
from __future__ import annotations

import re

from scripts.parser_odpowiedzi import normalize, extract_odpowiedz, wylusk_zamkniete

_VALID = re.compile(r"^[A-DPF]{1,2}$")


def strict_compliant(raw: str, odp: str) -> bool:
    tag = extract_odpowiedz(raw) or odp
    return bool(_VALID.match(normalize(tag))) if tag else False


def generous_parsed(raw: str, odp: str) -> bool:
    return bool(wylusk_zamkniete(raw, odp))


def format_compliance(rows: list[dict]) -> float:
    """Udział zadań zamkniętych odpowiedzianych ściśle (tag + poprawny token)."""
    if not rows:
        return 0.0
    n = sum(1 for r in rows if strict_compliant(r["raw"], r.get("odpowiedz", "")))
    return n / len(rows)


def conditional_accuracy(rows: list[dict], ids: list[int], keys: dict[int, str]) -> float:
    """Poprawność wśród zadań, które dało się hojnie sparsować."""
    parsed = correct = 0
    for r, qid in zip(rows, ids):
        guess = wylusk_zamkniete(r["raw"], r.get("odpowiedz", ""))
        if not guess:
            continue
        parsed += 1
        if guess == normalize(keys[qid]):
            correct += 1
    return correct / parsed if parsed else 0.0
