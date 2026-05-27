"""
Faza 4: ocena odpowiedzi modeli vs klucz.

- Zadania zamknięte (1-14): proste porównanie tekstowe (z normalizacją).
- Zadania otwarte (15-20): Claude Opus 4.7 jako egzaminator wg kryteriów CKE.

Obsługuje listę modeli (dynamicznie). Wynik: results/ocena_szczegolowa.json
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.parser_odpowiedzi import normalize, wylusk_zamkniete  # noqa: E402

load_dotenv(ROOT / ".env")
if not os.environ.get("ANTHROPIC_API_KEY"):
    sys.exit("Brak ANTHROPIC_API_KEY")

ZADANIA = ROOT / "data" / "zadania.json"
KLUCZ = ROOT / "data" / "klucz_odpowiedzi.json"
PROMPT_OCENA = (ROOT / "prompts" / "ocena_otwartych.txt").read_text(encoding="utf-8")
OUT = ROOT / "results" / "ocena_szczegolowa.json"

# Lista modeli do oceny — (klucz_w_json, ścieżka_do_results)
MODELE: list[tuple[str, Path]] = [
    ("gemma", ROOT / "results" / "gemma_odpowiedzi.json"),
    ("gemma_text", ROOT / "results" / "gemma_text_odpowiedzi.json"),
    ("gemma4_text", ROOT / "results" / "gemma4_text_odpowiedzi.json"),
    ("gemma4_mm", ROOT / "results" / "gemma4_mm_odpowiedzi.json"),
    ("bielik", ROOT / "results" / "bielik_odpowiedzi.json"),
    ("bielik_minitron", ROOT / "results" / "bielik_minitron_odpowiedzi.json"),
    ("pllum", ROOT / "results" / "pllum_odpowiedzi.json"),
    ("pllum12", ROOT / "results" / "pllum12_odpowiedzi.json"),
    # nowa generacja 2512 (CYFRAGOVPL) — dopisywane przez 04e/04f/04g
    ("llama_pllum8_2512", ROOT / "results" / "pllum8_2512_odpowiedzi.json"),
    ("pllum12_2512", ROOT / "results" / "pllum12_2512_odpowiedzi.json"),
    ("pllum4_2512", ROOT / "results" / "pllum4_2512_odpowiedzi.json"),
]

SEDZIA = "claude-opus-4-7"


def porownaj_zamkniete(odp_model: str, raw: str, odp_klucz: str) -> tuple[bool, str]:
    a = wylusk_zamkniete(raw, odp_model)
    b = normalize(odp_klucz)
    return (a == b), f"{a} vs {b}"


def ocen_otwarte(client: Anthropic, zadanie: dict, klucz: dict, odp_model: dict) -> dict:
    prompt = PROMPT_OCENA.format(
        tresc=zadanie["tresc"],
        wzorzec=f"{klucz['odpowiedz']}\n\nKrok po kroku:\n{klucz['rozwiazanie']}",
        pkt_max=zadanie["punkty_max"],
        odp_modelu=odp_model["raw"],
    )

    msg = client.messages.create(
        model=SEDZIA,
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = msg.content[0].text.strip()
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        return {"punkty": 0, "uzasadnienie": f"PARSE ERROR: {raw[:200]}"}
    return json.loads(m.group(0))


def main() -> None:
    zadania = {z["id"]: z for z in json.loads(ZADANIA.read_text(encoding="utf-8"))}
    klucz = {k["id"]: k for k in json.loads(KLUCZ.read_text(encoding="utf-8"))}

    # pomiń modele bez pliku wyników (np. nieukończona konwersja MLX)
    global MODELE
    brakujace = [n for n, s in MODELE if not s.exists()]
    if brakujace:
        print(f"Pomijam (brak pliku wyników): {', '.join(brakujace)}")
    MODELE = [(n, s) for n, s in MODELE if s.exists()]

    # załaduj dostępne modele
    odpowiedzi: dict[str, dict[int, dict]] = {}
    for nazwa, sciezka in MODELE:
        odpowiedzi[nazwa] = {r["id"]: r for r in json.loads(sciezka.read_text(encoding="utf-8"))}

    client = Anthropic()
    wyniki = []

    for nid in range(1, 21):
        z = zadania[nid]
        k = klucz[nid]
        wynik = {
            "id": nid,
            "typ": z["typ"],
            "punkty_max": z["punkty_max"],
            "klucz_odp": k["odpowiedz"],
        }

        if z["typ"] == "zamkniete":
            for nazwa, _ in MODELE:
                o = odpowiedzi[nazwa][nid]
                ok, cmp_ = porownaj_zamkniete(o["odpowiedz"], o["raw"], k["odpowiedz"])
                wynik[nazwa] = {
                    "odp": cmp_.split(" vs ")[0],
                    "punkty": z["punkty_max"] if ok else 0,
                    "porownanie": cmp_,
                }
        else:
            for nazwa, _ in MODELE:
                o = odpowiedzi[nazwa][nid]
                print(f"  oceniam z{nid:02d} {nazwa}...")
                ocena = ocen_otwarte(client, z, k, o)
                print(f"    -> {ocena['punkty']}/{z['punkty_max']}")
                wynik[nazwa] = {
                    "odp": o["odpowiedz"],
                    "punkty": ocena["punkty"],
                    "uzasadnienie": ocena["uzasadnienie"],
                }

        wyniki.append(wynik)

    # podsumowania per model
    podsumowania = {}
    for nazwa, _ in MODELE:
        suma = sum(w[nazwa]["punkty"] for w in wyniki)
        zamk_corr = sum(
            1 for w in wyniki if w["typ"] == "zamkniete" and w[nazwa]["punkty"] > 0
        )
        zamk_total = sum(1 for w in wyniki if w["typ"] == "zamkniete")
        otw_pkt = sum(w[nazwa]["punkty"] for w in wyniki if w["typ"] == "otwarte")
        otw_max = sum(w["punkty_max"] for w in wyniki if w["typ"] == "otwarte")
        podsumowania[nazwa] = {
            "suma": suma,
            "procent": round(100 * suma / 30, 1),
            "zamk_correct": zamk_corr,
            "zamk_total": zamk_total,
            "otw_pkt": otw_pkt,
            "otw_max": otw_max,
        }

    summary = {
        "sedzia": SEDZIA,
        "max_pkt": 30,
        "modele": [m[0] for m in MODELE],
        **podsumowania,
        "zadania": wyniki,
    }

    OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    for nazwa, _ in MODELE:
        s = podsumowania[nazwa]
        print(f"  {nazwa:12}: {s['suma']:2}/30 ({s['procent']}%) — zamk {s['zamk_correct']}/{s['zamk_total']}, otw {s['otw_pkt']}/{s['otw_max']}")
    print(f"OK: {OUT}")


if __name__ == "__main__":
    main()
