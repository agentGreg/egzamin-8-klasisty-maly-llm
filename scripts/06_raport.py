"""
Faza 5: raport.md z wynikami benchmarku.

Obsługuje listę modeli zdefiniowaną w MODELE (poniżej).
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean

from tabulate import tabulate

ROOT = Path(__file__).resolve().parents[1]
OCENA = ROOT / "results" / "ocena_szczegolowa.json"
RAPORT = ROOT / "results" / "raport.md"

# Konfiguracja kolejności i etykiet modeli w raporcie
MODELE = [
    ("bielik", "Bielik 4.5B v3 (8-bit, text-only)", ROOT / "results" / "bielik_odpowiedzi.json"),
    ("gemma_text", "Gemma 3 4B IT (4-bit, text-only)", ROOT / "results" / "gemma_text_odpowiedzi.json"),
    ("gemma", "Gemma 3 4B IT (4-bit, multimodal)", ROOT / "results" / "gemma_odpowiedzi.json"),
]


def main() -> None:
    data = json.loads(OCENA.read_text(encoding="utf-8"))
    raw_per_model = {
        nazwa: {r["id"]: r for r in json.loads(p.read_text(encoding="utf-8"))}
        for nazwa, _, p in MODELE
    }

    # ===== Tabela szczegółowa =====
    rows = []
    for z in data["zadania"]:
        nid = z["id"]
        row = [
            f"z{nid:02d}",
            "zamk" if z["typ"] == "zamkniete" else "otw",
            f"{z['punkty_max']}",
            (z["klucz_odp"][:25] + ("…" if len(z["klucz_odp"]) > 25 else "")),
        ]
        for nazwa, _, _ in MODELE:
            odp = (z[nazwa]["odp"] or "—")[:22]
            row.append(odp)
            row.append(f"{z[nazwa]['punkty']}/{z['punkty_max']}")
        rows.append(row)

    headers = ["zad", "typ", "max", "klucz"]
    for _, etykieta, _ in MODELE:
        krotka = etykieta.split()[0]  # "Bielik" / "Gemma"
        sufix = "txt" if "text-only" in etykieta and "multi" not in etykieta else ("MM" if "multimodal" in etykieta else "")
        label = f"{krotka} {sufix}".strip()
        headers.extend([f"{label} odp", f"{label} pkt"])

    tabela = tabulate(rows, headers=headers, tablefmt="github")

    # ===== Tabela podsumowania =====
    podsum_rows = []
    for nazwa, etykieta, _ in MODELE:
        s = data[nazwa]
        podsum_rows.append([
            etykieta,
            f"**{s['suma']} / 30**",
            f"**{s['procent']}%**",
            f"{s['zamk_correct']}/{s['zamk_total']}",
            f"{s['otw_pkt']}/{s['otw_max']} pkt",
        ])
    podsum = tabulate(
        podsum_rows,
        headers=["Model", "Wynik", "Procent", "Zamknięte", "Otwarte"],
        tablefmt="github",
    )

    # ===== Wydajność =====
    perf_rows = []
    for nazwa, etykieta, _ in MODELE:
        times = [v["czas_s"] for v in raw_per_model[nazwa].values()]
        tps = [v["tokens_per_s"] for v in raw_per_model[nazwa].values() if v.get("tokens_per_s")]
        perf_rows.append([
            etykieta,
            f"{sum(times):.0f} s",
            f"{mean(times):.1f} s",
            f"{mean(tps):.0f} tok/s" if tps else "—",
        ])
    perf = tabulate(
        perf_rows,
        headers=["Model", "Łączny czas", "Średni / zadanie", "Throughput"],
        tablefmt="github",
    )

    # ===== Raport =====
    raport_md = f"""# Raport — Egzamin Ósmoklasisty z Matematyki, 12 maja 2026

Benchmark trzech konfiguracji modeli ~4-5B parametrów uruchamianych lokalnie przez MLX na Apple M5 Max.

## Wyniki

{podsum}

(Próg zdawalności egzaminu nie jest formalnie ustanowiony — wynik to liczba zdobytych punktów na 30 możliwych.)

## Tabela szczegółowa

{tabela}

## Wydajność

{perf}

## Metodyka

- **Arkusz**: oficjalny PDF CKE z 12 maja 2026, 20 zadań (1–14 zamknięte ABCD/PF, 15–20 otwarte), max 30 pkt.
- **Klucz odpowiedzi**: wygenerowany przez Claude Opus 4.7 z PDF jako kontekst, następnie **ręcznie zweryfikowany** (Claude pomylił się w 5 zadaniach — głównie copy-paste między rozumowaniem a polem `odpowiedz`).
- **Runtime**: `mlx-lm` / `mlx-vlm` na Apple M5 Max, 128 GB unified memory.
- **Gemma multimodal**: przez `mlx-vlm` — model widzi obrazki zadań bezpośrednio.
- **Gemma text-only**: ten sam model przez `mlx-lm`, dostaje tekstowe opisy rysunków zamiast obrazków (uczciwy compare z Bielikiem).
- **Bielik 4.5B v3** (text-only): dostaje te same opisy rysunków.
- **Ocena zadań otwartych**: Claude Opus 4.7 wg kryteriów CKE (pełna metoda + wynik, błąd rachunkowy, brak postępu).
- **Temperatura**: 0 dla wszystkich modeli (deterministyczne odpowiedzi).
- **Parser odpowiedzi**: preferuje `<odpowiedz>` → `\\boxed{{}}` → ostatnie „Odpowiedź: X" → fallback.
"""
    RAPORT.parent.mkdir(parents=True, exist_ok=True)
    RAPORT.write_text(raport_md, encoding="utf-8")
    print(f"OK: {RAPORT}")


if __name__ == "__main__":
    main()
