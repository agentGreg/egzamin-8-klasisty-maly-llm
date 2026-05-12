"""
Faza 1: ekstrakcja zadań z PDF.

Tworzy:
- data/obrazki/zNN.png  — render obszaru każdego zadania (300 DPI)
- data/zadania_raw.json — surowa baza (treść, punkty_max) wyciągnięta z PDF
                          (opcje i opis_obrazka uzupełniane ręcznie w 02)
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pymupdf

PDF = Path("matematyka-2026-egzamin-osmoklasisty.pdf")
OUT_JSON = Path("data/zadania_raw.json")
OUT_IMG_DIR = Path("data/obrazki")

# Mapowanie: nr zadania -> 1-indexed numer strony PDF (z czytania arkusza)
ZADANIE_NA_STRONE = {
    1: 3, 2: 3, 3: 3,
    4: 4, 5: 4, 6: 4, 7: 4,
    8: 5, 9: 5, 10: 5,
    11: 6, 12: 6,
    13: 7, 14: 7,
    15: 9, 16: 10, 17: 11, 18: 12, 19: 13, 20: 14,
}

ZADANIE_RE = re.compile(r"Zadanie\s+(\d+)\.\s*\((\d)\s*[-–]\s*(\d)\)")


def main() -> None:
    OUT_IMG_DIR.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(PDF)

    # 1) Znajdź bboxy nagłówków "Zadanie N. (0-X)" per strona
    pozycje: dict[int, tuple[int, int, int, pymupdf.Rect]] = {}
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        for inst in page.search_for("Zadanie", quads=False):
            # weź pełną linię (klamra zadania)
            line_text = page.get_textbox(
                pymupdf.Rect(inst.x0 - 2, inst.y0 - 2, page.rect.x1, inst.y1 + 2)
            )
            m = ZADANIE_RE.search(line_text)
            if not m:
                continue
            n = int(m.group(1))
            pkt_max = int(m.group(3))
            # Niektóre zadania (15-20) są wymienione TAKŻE na stronie 8 jako referencje.
            # Bierzemy tylko jedną — z faktycznej strony zadania (ZADANIE_NA_STRONE).
            expected_page = ZADANIE_NA_STRONE.get(n)
            if expected_page is None or expected_page != page_idx + 1:
                continue
            pozycje[n] = (page_idx, pkt_max, inst.y0, inst)

    if len(pozycje) != 20:
        missing = sorted(set(range(1, 21)) - set(pozycje))
        raise RuntimeError(f"Brak zadań w PDF: {missing}")

    # 2) Dla każdej strony zbierz nagłówki posortowane wg Y, by ustalić koniec zadania.
    per_page: dict[int, list[tuple[int, float]]] = {}
    for n, (pidx, _, y0, _) in pozycje.items():
        per_page.setdefault(pidx, []).append((n, y0))
    for pidx in per_page:
        per_page[pidx].sort(key=lambda t: t[1])

    # 3) Crop + render + tekst per zadanie
    zadania = []
    for n in range(1, 21):
        pidx, pkt_max, y0, header_rect = pozycje[n]
        page = doc[pidx]
        # koniec: y0 następnego zadania na tej samej stronie, inaczej dół strony
        same_page = [y for (nn, y) in per_page[pidx] if y > y0]
        # zostaw drobny margines na "PRZENIEŚ ROZWIĄZANIA…" itp.
        y_bottom = (min(same_page) - 4) if same_page else page.rect.y1 - 30
        # cropuj cały obszar w poziomie strony (poza skrajnymi marginesami)
        clip = pymupdf.Rect(40, max(0, y0 - 6), page.rect.x1 - 40, y_bottom)
        pix = page.get_pixmap(clip=clip, dpi=300)
        out_png = OUT_IMG_DIR / f"z{n:02d}.png"
        pix.save(out_png)

        # tekst
        tresc = page.get_textbox(clip).strip()

        zadania.append({
            "id": n,
            "typ": "zamkniete" if n <= 14 else "otwarte",
            "punkty_max": pkt_max,
            "tresc": tresc,
            "obrazek": str(out_png),
            "opis_obrazka": "",  # uzupełniane ręcznie
            "opcje": {},          # uzupełniane ręcznie dla zamkniętych
        })

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(zadania, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"OK: {len(zadania)} zadań, JSON={OUT_JSON}, PNGs={OUT_IMG_DIR}")


if __name__ == "__main__":
    main()
