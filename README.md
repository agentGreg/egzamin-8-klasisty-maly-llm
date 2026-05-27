# Egzamin ósmoklasisty z matematyki 2026 — benchmark małych LLM-ów

Czy otwarty model językowy ~4-12B parametrów uruchomiony lokalnie na MacBooku
poradzi sobie z egzaminem ósmoklasisty z matematyki?

Sprawdzam to na oficjalnym arkuszu CKE z 12 maja 2026 (20 zadań, 30 pkt max).

## Wyniki

| Model | Wynik | Zamknięte | Otwarte |
|---|---|---|---|
| 🥇 **Bielik-Minitron 7B v3** (SpeakLeash, kompresja Bielika 11B v3) | **25 / 30 (83%)** | 11/14 | 14/16 pkt |
| 🥈 **Gemma 4 E4B** (Google, text-only) | **24 / 30 (80%)** | 10/14 | 14/16 pkt |
| 🥉 **Bielik 4.5B v3** (SpeakLeash) | **23 / 30 (77%)** | 12/14 | 11/16 pkt |
| 🥉 **Gemma 4 E4B** (Google, multimodal) | **23 / 30 (77%)** | 11/14 | 12/16 pkt |
| 5. Gemma 3 4B IT (text-only) | 18 / 30 (60%) | 9/14 | 9/16 pkt |
| 6. Gemma 3 4B IT (multimodal) | 14 / 30 (47%) | 8/14 | 6/16 pkt |
| 7. PLLuM 12B Instruct **2412** (CYFRAGOVPL) | 4 / 30 (13%) | 2/14 | 2/16 pkt |
| 8. Llama-PLLuM 8B Instruct **2412** (CYFRAGOVPL) | 3 / 30 (10%) | 1/14 | 2/16 pkt |
| 8. Llama-PLLuM 8B Instruct **2512** (CYFRAGOVPL) | 3 / 30 (10%) | 2/14 | 1/16 pkt |
| 10. PLLuM 12B Instruct **2512** (CYFRAGOVPL) | 2 / 30 (7%) | 1/14 | 1/16 pkt |

> **PLLuM 4B 2512** (najnowszy, najmniejszy) nie wszedł do tabeli: architektonicznie to `Gemma3ForConditionalGeneration` (zbudowany na Gemmie 3 4B Google'a + polski post-training), wydany jako text-only — nie udało się go uruchomić na pipelinie MLX (niezgodność `mlx-lm`/`mlx-vlm`, szczegóły w [`results/raport.md`](results/raport.md)).

Pełna tabela per-zadaniowa + wydajność: [`results/raport.md`](results/raport.md).

**TL;DR:**
- **Bielik-Minitron 7B** (skompresowana wersja Bielika 11B v3, -33% parametrów przez structured pruning + distillation) wygrywa jako pierwszy model w serii powyżej 80%.
- **Gemma 4 zrobiła ogromny skok** vs Gemma 3 na tej samej ilości parametrów (+6 pkt) i wyrównała z Bielikiem 4.5B.
- **Vision tower naprawiony w Gemma 4.** Multimodalność w Gemma 3 zabierała 4 pkt vs text-only. W Gemma 4 zabiera tylko 1 pkt (w granicach szumu sędziego). Małe modele edge dorastają do multimodalności „za darmo".
- **Nowa generacja PLLuM (2512) nie naprawiła wyniku.** PLLuM 12B 2512 = 2/30 (spadek z 4/30 w 2412), Llama-PLLuM 8B 2512 = 3/30 — najniżej w całym benchmarku. Nowa wersja *trzyma teraz format* (12B-2512 zwraca `<odpowiedz>`, czego 2412 nie robił), ale odpowiedzi dalej błędne: manier się nauczyła, matematyki nie.
- **To nie artefakt parsowania.** Parser jest celowo hojny (łapie też styl GSM8K `#### X` i odpowiedzi opisowe „D. ..."); PLLuM 2412 ignoruje strukturę zadań zamkniętych i liczy free-form, a i tak po maksymalnie fair wyłuskaniu osiąga tylko 3–4/30. Niski wynik jest realny.
- **Nowy PLLuM 4B = Gemma 3 Google'a** (`Gemma3ForConditionalGeneration` + polski post-training) — ciekawy zwrot architektoniczny, ale nie udało się go uruchomić na MLX.
- Wszystko offline, na Apple Silicon przez MLX, ~1–3 minuty inferencji per model.

## 🥇 Wagi MLX dla Bielik-Minitron 7B

Skoro Bielik-Minitron 7B wygrał benchmark, a nie miał oficjalnych wag MLX, opublikowałem zestaw kwantyzacji dla społeczności Apple Silicon (za zgodą zespołu SpeakLeash):

| Wariant | Rozmiar | Use case |
|---|---|---|
| [`agentGreg/Bielik-Minitron-7B-v3.0-Instruct-MLX-4bit`](https://huggingface.co/agentGreg/Bielik-Minitron-7B-v3.0-Instruct-MLX-4bit) | ~4 GB | edge / MacBook Air, ograniczona pamięć |
| [`agentGreg/Bielik-Minitron-7B-v3.0-Instruct-MLX-6bit`](https://huggingface.co/agentGreg/Bielik-Minitron-7B-v3.0-Instruct-MLX-6bit) | ~5.5 GB | sweet spot quality/size |
| [`agentGreg/Bielik-Minitron-7B-v3.0-Instruct-MLX-8bit`](https://huggingface.co/agentGreg/Bielik-Minitron-7B-v3.0-Instruct-MLX-8bit) | ~7.5 GB | high quality, używana w benchmarku |
| [`agentGreg/Bielik-Minitron-7B-v3.0-Instruct-MLX-bf16`](https://huggingface.co/agentGreg/Bielik-Minitron-7B-v3.0-Instruct-MLX-bf16) | ~14 GB | pełna precyzja, źródło |

```python
from mlx_lm import load, generate

model, tokenizer = load("agentGreg/Bielik-Minitron-7B-v3.0-Instruct-MLX-6bit")

messages = [{"role": "user", "content": "Wyjaśnij prosto twierdzenie Pitagorasa."}]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
print(generate(model, tokenizer, prompt=prompt, max_tokens=400, verbose=True))
```

Konwersja, README per repo, smoke test i upload: zob. [`publish/`](publish/).

## Architektura

```
01_przygotuj_zadania.py    →  data/zadania.json + obrazki/zNN.png
02_klucz_claude.py         →  data/klucz_odpowiedzi.json  (Claude Opus 4.7)
03_run_gemma.py            →  results/gemma_odpowiedzi.json          (Gemma 3 multimodal, mlx-vlm)
03b_run_gemma_text.py      →  results/gemma_text_odpowiedzi.json     (Gemma 3 text-only, mlx-lm)
03c_run_gemma4_text.py     →  results/gemma4_text_odpowiedzi.json    (Gemma 4 text-only, mlx-vlm bez image)
03d_run_gemma4_mm.py       →  results/gemma4_mm_odpowiedzi.json      (Gemma 4 multimodal, mlx-vlm)
04_run_bielik.py           →  results/bielik_odpowiedzi.json         (Bielik 4.5B, mlx-lm)
04b_run_bielik_minitron.py →  results/bielik_minitron_odpowiedzi.json (Bielik-Minitron 7B, mlx-lm)
04c_run_pllum.py           →  results/pllum_odpowiedzi.json          (Llama-PLLuM 8B, mlx-lm)
04d_run_pllum12.py         →  results/pllum12_odpowiedzi.json        (PLLuM 12B 2412, mlx-lm)
04e_run_pllum8_2512.py     →  results/pllum8_2512_odpowiedzi.json    (Llama-PLLuM 8B 2512, mlx-lm)
04f_run_pllum12_2512.py    →  results/pllum12_2512_odpowiedzi.json   (PLLuM 12B 2512, mlx-lm)
04g_run_pllum4_2512.py     →  results/pllum4_2512_odpowiedzi.json    (PLLuM 4B 2512 = Gemma3; nie wszedł na MLX)
05_ocen.py                 →  results/ocena_szczegolowa.json         (Claude sędzią otwartych)
06_raport.py               →  results/raport.md
```

Dla każdego modelu identyczny prompt (`prompts/system_*.txt`), temperatura 0,
parsowanie odpowiedzi tolerancyjne (`<odpowiedz>`, `\boxed{}`, fallback regex).

## Uruchomienie

Wymagania: macOS na Apple Silicon, [`uv`](https://github.com/astral-sh/uv),
klucz Anthropic API (Claude pełni rolę klucza odpowiedzi + sędziego otwartych),
HF token (Bielik-Minitron 7B jest gated).

```bash
git clone https://github.com/agentGreg/egzamin-8-klasisty-maly-llm.git
cd egzamin-8-klasisty-maly-llm

# 1. Pobierz arkusz CKE (nie jest w repo)
curl -o matematyka-2026-egzamin-osmoklasisty.pdf \
  https://arkusze.pl/osmoklasisty/matematyka-2026-egzamin-osmoklasisty.pdf

# 2. Klucze API w .env
cat > .env <<EOF
ANTHROPIC_API_KEY=sk-ant-...
HF_TOKEN=hf_...      # potrzebny tylko dla Bielik-Minitron 7B (gated repo)
EOF

# 3. Dependencies (mlx-lm, mlx-vlm, anthropic, pymupdf, pillow, dotenv, tabulate)
uv sync

# 4. Konwersja MLX dla modeli bez gotowych wag MLX
#    Bielik 4.5B v3, Bielik-Minitron 7B (agentGreg/...-MLX-*) i Gemma 3/4
#    mają gotowe MLX. Konwertujemy tylko PLLuM-y:
uv run python -m mlx_lm convert \
  --hf-path CYFRAGOVPL/Llama-PLLuM-8B-instruct \
  --mlx-path ~/.cache/huggingface/local-mlx/Llama-PLLuM-8B-instruct-mlx-8bit \
  -q --q-bits 8

# 5. Pełen pipeline
uv run python scripts/01_przygotuj_zadania.py     # wyciąga PNG per zadanie
uv run python scripts/02_klucz_claude.py          # klucz odpowiedzi
uv run python scripts/03_run_gemma.py             # Gemma 3 multimodal
uv run python scripts/03b_run_gemma_text.py      # Gemma 3 text-only
uv run python scripts/03c_run_gemma4_text.py     # Gemma 4 text-only
uv run python scripts/03d_run_gemma4_mm.py        # Gemma 4 multimodal
uv run python scripts/04_run_bielik.py            # Bielik 4.5B v3
uv run python scripts/04b_run_bielik_minitron.py  # Bielik-Minitron 7B
uv run python scripts/04c_run_pllum.py            # Llama-PLLuM 8B 2412
uv run python scripts/04d_run_pllum12.py          # PLLuM 12B 2412
uv run python scripts/04e_run_pllum8_2512.py      # Llama-PLLuM 8B 2512 (konwersja MLX jak 2412)
uv run python scripts/04f_run_pllum12_2512.py     # PLLuM 12B 2512 (konwersja MLX jak 2412)
# 04g_run_pllum4_2512.py — PLLuM 4B 2512 (Gemma3); nie startuje na MLX, patrz raport
uv run python scripts/05_ocen.py                  # ocena (Claude sędzia)
uv run python scripts/06_raport.py                # raport
```

Modele MLX pobiorą się automatycznie z HF przy pierwszym uruchomieniu:
- `mlx-community/gemma-3-4b-it-4bit` (~3 GB)
- `mlx-community/gemma-4-e4b-it-4bit` (~5 GB, model multimodalny ładowany przez mlx-vlm)
- `speakleash/Bielik-4.5B-v3.0-Instruct-MLX-8bit` (~5 GB)

Modele do lokalnej konwersji:
- `speakleash/Bielik-Minitron-7B-v3.0-Instruct` (~14 GB safetensors, gated, ~8 GB po Q8)
- `CYFRAGOVPL/Llama-PLLuM-8B-instruct` (~16 GB safetensors, ~8 GB po Q8)

## Uruchamiane modele

- **Bielik 4.5B v3 Instruct** (SpeakLeash): polski LLM ogólnego zastosowania, 8-bit MLX (oficjalne wagi).
- **Bielik-Minitron 7B v3 Instruct** (SpeakLeash): skompresowana wersja **Bielika-11B-v3.0** (-33% parametrów, z 11.04B do 7.35B) przez structured pruning + knowledge distillation z użyciem NVIDIA Model Optimizer i NeMo Framework (podejście inspirowane techniką Minitron). [Paper: arxiv.org/abs/2603.11881](https://arxiv.org/abs/2603.11881). Gated na HF, lokalna konwersja do MLX 8-bit.
- **Llama-PLLuM 8B Instruct** (CYFRAGOVPL): polski instruction tuning na bazie Llama 3.1 8B, lokalna konwersja do MLX 8-bit.
- **PLLuM 12B Instruct** (CYFRAGOVPL): natywny dense PLLuM (nie Llama-based, według tokenizera bazuje na Mistral Small), gotowa wersja MLX Q6 z `lukagra/PLLuM-12B-instruct-Q6-mlx`.
- **Gemma 3 4B IT** (Google): 4-bit MLX, dwa warianty: multimodalny (mlx-vlm z obrazkami) i text-only (mlx-lm z opisami rysunków).
- **Gemma 4 E4B IT** (Google): nowsza edycja edge, 4-bit MLX, dwa warianty: multimodalny (mlx-vlm z obrazkami) i text-only (mlx-vlm bez obrazków).

### Nowa generacja PLLuM 2512 (grudzień 2025)

- **Llama-PLLuM 8B Instruct 2512** (CYFRAGOVPL): następca 8B z 2412, `LlamaForCausalLM`, 8-bit MLX po konwersji.
- **PLLuM 12B Instruct 2512** (CYFRAGOVPL): następca natywnego 12B, `MistralForCausalLM`, 8-bit MLX po konwersji. W 2512 model **zaczął trzymać format `<odpowiedz>`** (2412 liczył free-form, styl GSM8K), ale wynik spadł do 2/30.
- **PLLuM 4B Instruct 2512** (CYFRAGOVPL): nowy najmniejszy wariant — architektonicznie `Gemma3ForConditionalGeneration`, zbudowany **na Gemmie 3 4B Google'a** + polski post-training, wydany text-only. Nie udało się uruchomić na MLX (`mlx-vlm` wymaga procesora obrazu, którego repo nie ma; `mlx-lm` ma niezgodność rozmiaru embeddingu w `gemma3`).

## Uwagi metodyczne

- **Klucz odpowiedzi** generowany przez Claude Opus 4.7 z PDF jako kontekst, następnie ręcznie zweryfikowany. Claude pomylił się w 5 zadaniach (głównie niespójność między rozumowaniem a finalną literą). Po weryfikacji klucz zgadza się z obliczeniem matematycznym.
- **Opisy rysunków dla modeli text-only** są zwięzłe i neutralne (np. *„sześcian ABCDEFGH, punkt S w środku krawędzi DH"*) — bez podpowiadania odpowiedzi. Wszystkie 5 modeli text-only dostaje dokładnie te same opisy.
- **Ocena zadań otwartych** wg standardowych kryteriów CKE: pełna metoda + wynik = max pkt, błąd rachunkowy = N-1, istotny postęp = częściowe pkt.

## Arkusz CKE

PDF arkusza nie jest w repo (3.3 MB). Pobierz z:
https://arkusze.pl/osmoklasisty/matematyka-2026-egzamin-osmoklasisty.pdf

Lub bezpośrednio z [CKE](https://cke.gov.pl/egzamin-osmoklasisty/arkusze/).

## Stack

- **Runtime**: MLX (`mlx-lm` / `mlx-vlm`) — natywne pod Apple Silicon.
- **Sędzia**: Claude Opus 4.7 (1M context, Anthropic SDK).
- **Sprzęt referencyjny**: Apple Silicon (testowane na M-series).

## Licencja

Kod: MIT. Treść arkusza i klucz odpowiedzi: © CKE 2026.

---

Made with ❤️ at **[Prosit AS](https://prosit.no)** — norwesko-polski software house
(integracje systemów, data science, API & apps, AI/LLM, cloud, cybersecurity).
Autor: **Grzegorz Brzezinka** · [greg@prosit.no](mailto:greg@prosit.no)
