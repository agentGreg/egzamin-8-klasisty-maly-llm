# Egzamin ósmoklasisty z matematyki 2026 — benchmark małych LLM-ów

Czy otwarty model językowy ~4-5B parametrów uruchomiony lokalnie na MacBooku
poradzi sobie z egzaminem ósmoklasisty z matematyki?

Sprawdzam to na oficjalnym arkuszu CKE z 12 maja 2026 (20 zadań, 30 pkt max).

## Wyniki

| Model | Wynik | Zamknięte | Otwarte |
|---|---|---|---|
| 🥇 **Bielik 4.5B v3** (8-bit, text-only) | **24 / 30 (80%)** | 12/14 | 12/16 pkt |
| 🥈 Gemma 3 4B IT (4-bit, text-only) | 18 / 30 (60%) | 9/14 | 9/16 pkt |
| 🥉 Gemma 3 4B IT (4-bit, multimodal) | 14 / 30 (47%) | 8/14 | 6/16 pkt |

Pełna tabela per-zadaniowa + wydajność: [`results/raport.md`](results/raport.md).

**TL;DR:**
- Polski **Bielik 4.5B** (SpeakLeash) wygrywa z Gemmą 3 nawet w fair compare (text-only).
- **Multimodalność Gemmy zaszkodziła** o 4 pkt vs. ta sama Gemma w text-only.
- Wszystko offline, na Apple M5 Max przez MLX, ~3 min inferencji.

## Architektura

```
01_przygotuj_zadania.py  →  data/zadania.json + obrazki/zNN.png
02_klucz_claude.py       →  data/klucz_odpowiedzi.json  (Claude Opus 4.7)
03_run_gemma.py          →  results/gemma_odpowiedzi.json       (multimodal, mlx-vlm)
03b_run_gemma_text.py    →  results/gemma_text_odpowiedzi.json  (text-only, mlx-lm)
04_run_bielik.py         →  results/bielik_odpowiedzi.json      (text-only, mlx-lm)
05_ocen.py               →  results/ocena_szczegolowa.json      (Claude jako sędzia otwartych)
06_raport.py             →  results/raport.md
```

Dla każdego modelu prompt jest taki sam (`prompts/system_*.txt`), temperatura 0,
parsowanie odpowiedzi tolerancyjne (`<odpowiedz>`, `\boxed{}`, fallback regex).

## Uruchomienie

Wymagania: macOS na Apple Silicon, [`uv`](https://github.com/astral-sh/uv),
klucz Anthropic API (Claude pełni rolę klucza odpowiedzi + sędziego otwartych).

```bash
# 1. Pobierz arkusz CKE
curl -o matematyka-2026-egzamin-osmoklasisty.pdf \
  https://arkusze.pl/osmoklasisty/matematyka-2026-egzamin-osmoklasisty.pdf

# 2. Klucz API
echo 'ANTHROPIC_API_KEY=sk-ant-...' > .env

# 3. Dependencies (mlx-lm, mlx-vlm, anthropic, pymupdf, pillow, dotenv, tabulate)
uv sync

# 4. Pełen pipeline
uv run python scripts/01_przygotuj_zadania.py    # wyciąga PNG per zadanie
uv run python scripts/02_klucz_claude.py         # klucz odpowiedzi
uv run python scripts/03_run_gemma.py            # Gemma multimodal
uv run python scripts/03b_run_gemma_text.py     # Gemma text-only
uv run python scripts/04_run_bielik.py           # Bielik
uv run python scripts/05_ocen.py                 # ocena
uv run python scripts/06_raport.py               # raport
```

Modele MLX pobiorą się automatycznie z HF przy pierwszym uruchomieniu
(`mlx-community/gemma-3-4b-it-4bit` ~3 GB, `speakleash/Bielik-4.5B-v3.0-Instruct-MLX-8bit` ~5 GB).

## Uwagi metodyczne

- **Klucz odpowiedzi** generowany przez Claude Opus 4.7 z PDF jako kontekst,
  następnie **ręcznie zweryfikowany** — Claude pomylił się w 5 zadaniach (głównie
  niespójność między rozumowaniem a finalną literą). Po weryfikacji klucz
  zgadza się z moim obliczeniem matematycznym.
- **Opisy rysunków dla modeli text-only** są zwięzłe i neutralne (np. *„sześcian
  ABCDEFGH, punkt S w środku krawędzi DH"*) — bez podpowiadania odpowiedzi.
  Bielik i Gemma text-only dostają dokładnie te same opisy.
- **Ocena zadań otwartych** wg standardowych kryteriów CKE: pełna metoda +
  wynik = max pkt, błąd rachunkowy = N-1, istotny postęp = częściowe pkt.

## Arkusz CKE

PDF arkusza nie jest w repo (3.3 MB). Pobierz z:
https://arkusze.pl/osmoklasisty/matematyka-2026-egzamin-osmoklasisty.pdf

Lub bezpośrednio z [CKE](https://cke.gov.pl/egzamin-osmoklasisty/arkusze/).

## Stack

- **Runtime**: MLX (`mlx-lm` / `mlx-vlm`) — natywne pod Apple Silicon
- **Sędzia**: Claude Opus 4.7 (1M context, anthropic SDK)
- **Sprzęt referencyjny**: Apple M5 Max + 128 GB unified memory

## Licencja

Kod: MIT. Treść arkusza i klucz odpowiedzi: © CKE 2026.

---

Made with ❤️ at **[Prosit AS](https://prosit.no)** — norwesko-polski software house
(integracje systemów, data science, API & apps, AI/LLM, cloud, cybersecurity).
Autor: **Grzegorz Brzezinka** · [greg@prosit.no](mailto:greg@prosit.no)
