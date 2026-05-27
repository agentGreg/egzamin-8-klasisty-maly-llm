# Fine-tune Full Study (PLLuM + Bielik, early-stopped) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the PLLuM-only spike into the publishable study: train an **early-stopped** LoRA on a larger past-exam set for BOTH a weak model (Llama-PLLuM 8B 2512) and the strong control (Bielik-Minitron 7B), then measure the format-vs-math decomposition across a real chronological held-out test set (2024 + 2025 + 2026 main sheets).

**Architecture:** Reuse the spike's proven modules (`parser_odpowiedzi`, `metryki_format`, `08_buduj_dataset`). Add (1) a parametrized inference runner that works on any test sheet + key, (2) an early-stopping checkpoint-selection step, (3) a multi-sheet, multi-model decomposition aggregator, (4) a 2-model comparison graph. The **headline metrics (format-compliance, conditional-accuracy) are closed-task and key-only**, so they need NO Claude judge — the judge stays an optional secondary step for open-task totals.

**Tech Stack:** `mlx_lm` 0.31.3 (LoRA train + adapter load), `pytest`, `uv`, matplotlib. Base weights: existing 8-bit PLLuM MLX; Bielik-Minitron bf16 (`agentGreg/...-MLX-bf16`). Apple M5 Max / 128 GB.

**Builds on:** `docs/superpowers/plans/2026-05-27-finetune-spike.md` (spike, GO). Reuses `scripts/parser_odpowiedzi.py`, `scripts/metryki_format.py`, `scripts/08_buduj_dataset.py`, `tests/`.

**Carries forward two spike findings:**
1. Val loss bottomed at ~iter 60 then overfit on 52 examples → **early stopping is mandatory** (Task 5).
2. The 2023 sheet's statements were partly reconstructed from a corrupted PDF font → **re-transcribe 2023 cleanly** (Task 1).

---

## File structure

| Path | Responsibility |
|---|---|
| `data/archiwum/*.json` | **Expand.** Training sheets: 2019–2023 main + termin dodatkowy + trial exams (re-do 2023). Same schema as spike. |
| `data/testy/2024.json`, `2025.json` | **New.** Held-out test sheets (questions + `odpowiedz` key), zadania.json schema. 2026 reuses `data/zadania.json` + `data/klucz_odpowiedzi.json`. |
| `scripts/lib_inferencja.py` | **New.** Parametrized inference: `run_sheet(model_id, adapter_path, zadania_path, out_path)`. The single source of run logic (replaces per-model copies). |
| `scripts/12_wybierz_checkpoint.py` | **New.** Parse a LoRA training log, pick min-val-loss checkpoint, copy it to `<adapter_dir>/adapters.safetensors`. |
| `scripts/13_ocen_studium.py` | **New.** Multi-sheet × multi-model decomposition aggregator → `results/studium_decomposition.json`. |
| `scripts/14_wykresy_studium.py` | **New.** 2-model grouped decomposition graph + per-sheet breakdown. |
| `configs/lora_pllum_full.yaml`, `configs/lora_bielik_full.yaml` | **New.** Early-stopping configs (frequent eval + save). |
| `tests/test_lib_inferencja.py`, `tests/test_wybierz_checkpoint.py`, `tests/test_ocen_studium.py` | **New.** Unit tests for the new logic. |

---

## Task 1: Expand the training archive (and re-transcribe 2023)

**Files:**
- Create/replace: `data/archiwum/2019_glowny.json`, `2020_glowny.json`, `2023_glowny.json` (re-do), plus available `*_dodatkowy.json` and trial (`*_probny.json`) sheets from ≤2023.

PDFs are already in `temp/archiwum/` (full 2017–2025 archive). Test years (2024, 2025) are EXCLUDED from this task.

- [ ] **Step 1: List the ≤2023 source PDFs to transcribe**

Run:
```bash
ls temp/archiwum/ | grep -vE "2024|2025|2026" | grep -v odpowiedzi
```
Expected: the arkusz PDFs for 2017–2023 (main, dodatkowy, próbny). These are the training pool.

- [ ] **Step 2: Transcribe each missing sheet (subagent per sheet)**

For each sheet not already in `data/archiwum/`, dispatch a transcription subagent using the EXACT prompt template from the spike plan (Task 5, Step 2) — same schema, same rules, neutral `opis_obrazka`, official key for `odpowiedz`, worked solution for `rozwiazanie`. Re-transcribe 2023 with `pdftotext -layout` (the spike's 2021 agent confirmed this avoids the font-corruption issue) instead of reconstructing from memory.

- [ ] **Step 3: Validate the expanded archive**

Run:
```bash
uv run python -c "
import json,glob
tot=0
for f in sorted(glob.glob('data/archiwum/*.json')):
    a=json.load(open(f,encoding='utf-8'))
    assert all('odpowiedz' in z and z['rozwiazanie'].strip() for z in a), f
    tot+=len(a); print(f, len(a))
print('TOTAL', tot); assert tot>=120, 'want a larger train set than the spike'
"
```
Expected: each file with its count, `TOTAL >= 120`, no assertion error.

- [ ] **Step 4: Commit**

```bash
git add data/archiwum/
git commit -m "data: expand training archive to 2019-2023 (re-transcribe 2023)"
```

---

## Task 2: Transcribe the held-out test sheets (2024, 2025)

**Files:**
- Create: `data/testy/2024.json`, `data/testy/2025.json`

These use the `data/zadania.json` schema (NO `rozwiazanie` needed — they are test inputs) plus an `odpowiedz` field per task as the answer key. We keep questions + key together for closed-task scoring.

- [ ] **Step 1: Transcribe 2024 and 2025 main sheets (subagent per sheet)**

Dispatch a subagent for each, reading `temp/archiwum/matematyka-2024-egzamin-osmoklasisty.pdf` (+ `-odpowiedzi.pdf`) and the 2025 equivalents. Output `data/testy/<year>.json`: array of objects with fields `id, typ, punkty_max, tresc, opcje, obrazek (""), opis_obrazka, podtyp, odpowiedz`. `odpowiedz` = official key (letters for closed). No `rozwiazanie`.

- [ ] **Step 2: Validate test sheets**

Run:
```bash
uv run python -c "
import json
for y in ('2024','2025'):
    a=json.load(open(f'data/testy/{y}.json',encoding='utf-8'))
    nz=sum(1 for z in a if z['typ']=='zamkniete')
    assert all('odpowiedz' in z and z['odpowiedz'].strip() for z in a), y
    print(y, len(a), 'tasks,', nz, 'closed')
"
```
Expected: both years print with non-zero closed counts, no assertion error.

- [ ] **Step 3: Commit**

```bash
git add data/testy/
git commit -m "data: transcribe 2024+2025 held-out test sheets with keys"
```

---

## Task 3: Parametrized inference runner

**Files:**
- Create: `scripts/lib_inferencja.py`
- Create: `tests/test_lib_inferencja.py`

Extracts the run loop (currently duplicated across `04*_run_*.py`) into one function. Pure-logic helpers (`build_user_prompt`, `extract_odpowiedz`) are unit-tested; the model-loading `run_sheet` is integration-run later.

- [ ] **Step 1: Write failing tests for the pure helpers**

Create `tests/test_lib_inferencja.py`:
```python
from scripts.lib_inferencja import build_user_prompt, extract_odpowiedz


def test_user_prompt_closed_has_options():
    z = {"tresc": "T.", "typ": "zamkniete", "opcje": {"A": "1", "B": "2"}, "opis_obrazka": ""}
    p = build_user_prompt(z)
    assert "Opcje odpowiedzi:" in p and "A. 1" in p


def test_user_prompt_open_has_figure_desc():
    z = {"tresc": "Oblicz.", "typ": "otwarte", "opcje": {}, "opis_obrazka": "trójkąt ABC"}
    p = build_user_prompt(z)
    assert "Opis rysunku" in p and "trójkąt ABC" in p


def test_extract_tag():
    assert extract_odpowiedz("blah <odpowiedz>BD</odpowiedz>") == "BD"


def test_extract_tag_absent():
    assert extract_odpowiedz("no tag here") == ""
```

- [ ] **Step 2: Run, verify FAIL**

Run: `uv run pytest tests/test_lib_inferencja.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.lib_inferencja'`.

- [ ] **Step 3: Implement the runner**

Create `scripts/lib_inferencja.py`:
```python
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
```

- [ ] **Step 4: Run tests, verify PASS**

Run: `uv run pytest tests/test_lib_inferencja.py -v`
Expected: all 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/lib_inferencja.py tests/test_lib_inferencja.py
git commit -m "feat: parametrized inference runner for arbitrary test sheets"
```

---

## Task 4: Build the expanded dataset

**Files:**
- Output: `data/finetune/train.jsonl`, `valid.jsonl` (regenerated from the larger archive)

`08_buduj_dataset.py` already globs `data/archiwum/*.json`, so no code change — just regenerate.

- [ ] **Step 1: Regenerate the dataset**

Run: `uv run python scripts/08_buduj_dataset.py`
Expected: `train: N` with N noticeably larger than the spike's 52 (≥ ~110), `valid: M`.

- [ ] **Step 2: Verify the held-out years did NOT leak in**

Run:
```bash
uv run python -c "
import json
rows=[json.loads(l) for l in open('data/finetune/train.jsonl',encoding='utf-8')]
rows+=[json.loads(l) for l in open('data/finetune/valid.jsonl',encoding='utf-8')]
blob=json.dumps(rows, ensure_ascii=False)
for bad in ('2024','2025','2026'):
    pass  # years alone are not reliable markers; rely on file provenance instead
print('examples:', len(rows))
"
```
Expected: prints the example count. (Leakage is guaranteed structurally: Task 1 only transcribes ≤2023 into `data/archiwum/`, and the builder reads only that dir.)

- [ ] **Step 3: Commit** (data is gitignored; nothing to commit unless counts file added — skip if clean)

---

## Task 5: Early-stopping checkpoint selection

**Files:**
- Create: `scripts/12_wybierz_checkpoint.py`
- Create: `tests/test_wybierz_checkpoint.py`

mlx_lm.lora has no "save best". Strategy: train with `steps_per_eval == save_every` (e.g. 25) so every checkpoint has a paired val loss, then select the min-val-loss checkpoint and copy it to `adapters.safetensors` (what `load(adapter_path=...)` reads).

- [ ] **Step 1: Write failing test for the log parser + selector**

Create `tests/test_wybierz_checkpoint.py`:
```python
from scripts.wybierz_checkpoint import parse_val_losses, best_iter

LOG = """
Iter 1: Val loss 1.327, Val took 2.5s
Iter 25: Train loss 0.6
Iter 25: Val loss 0.80, Val took 1.0s
Iter 50: Val loss 0.74, Val took 1.0s
Iter 75: Val loss 0.91, Val took 1.0s
"""


def test_parse_val_losses():
    assert parse_val_losses(LOG) == {1: 1.327, 25: 0.80, 50: 0.74, 75: 0.91}


def test_best_iter_ignores_iter1_warmup():
    # iter 1 is the pre-training baseline; exclude it from selection
    assert best_iter({1: 1.327, 25: 0.80, 50: 0.74, 75: 0.91}) == 50
```
Note: import path is `scripts.wybierz_checkpoint` — give the module a digit-free importable name by also creating it WITHOUT the `12_` prefix is not allowed (convention). Instead the test loads via the digit-prefixed file using importlib:
```python
import importlib.util
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
def _load():
    spec = importlib.util.spec_from_file_location("wc12", ROOT / "scripts" / "12_wybierz_checkpoint.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
```
Replace the top-level `from scripts.wybierz_checkpoint import ...` with `m = _load()` and call `m.parse_val_losses(...)` / `m.best_iter(...)` in each test.

- [ ] **Step 2: Run, verify FAIL**

Run: `uv run pytest tests/test_wybierz_checkpoint.py -q`
Expected: FAIL — module file does not exist.

- [ ] **Step 3: Implement the selector**

Create `scripts/12_wybierz_checkpoint.py`:
```python
"""Faza 12: wybór checkpointu LoRA o najniższej stracie walidacyjnej (early stopping).

Parsuje log treningowy, znajduje iterację z min val loss (pomijając iter 1 baseline),
kopiuje NNNNNNN_adapters.safetensors -> adapters.safetensors w katalogu adaptera.
Użycie: python scripts/12_wybierz_checkpoint.py <adapter_dir> <log_file>
"""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

VAL_RE = re.compile(r"Iter\s+(\d+):\s*Val loss\s+([0-9.]+)")


def parse_val_losses(log_text: str) -> dict[int, float]:
    return {int(i): float(v) for i, v in VAL_RE.findall(log_text)}


def best_iter(vals: dict[int, float]) -> int:
    candidates = {i: v for i, v in vals.items() if i > 1}
    if not candidates:
        raise ValueError("no validation checkpoints after iter 1")
    return min(candidates, key=candidates.get)


def main() -> None:
    adapter_dir = Path(sys.argv[1])
    log_file = Path(sys.argv[2])
    vals = parse_val_losses(log_file.read_text(encoding="utf-8"))
    bi = best_iter(vals)
    ckpt = adapter_dir / f"{bi:07d}_adapters.safetensors"
    if not ckpt.exists():
        raise SystemExit(f"checkpoint {ckpt} not found — set save_every == steps_per_eval")
    shutil.copy(ckpt, adapter_dir / "adapters.safetensors")
    print(f"best val iter={bi} (loss {vals[bi]}); copied {ckpt.name} -> adapters.safetensors")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests, verify PASS**

Run: `uv run pytest tests/test_wybierz_checkpoint.py -v`
Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/12_wybierz_checkpoint.py tests/test_wybierz_checkpoint.py
git commit -m "feat: early-stopping checkpoint selection by min val loss"
```

---

## Task 6: Train both models (early-stopped)

**Files:**
- Create: `configs/lora_pllum_full.yaml`, `configs/lora_bielik_full.yaml`
- Output: `adapters/pllum8_full/`, `adapters/bielik_full/`

- [ ] **Step 1: Confirm Bielik-Minitron bf16 base is available**

Run:
```bash
uv run python -c "from huggingface_hub import snapshot_download; print(snapshot_download('agentGreg/Bielik-Minitron-7B-v3.0-Instruct-MLX-bf16'))"
```
Expected: prints a local cache path with model files. (PLLuM 8-bit base already confirmed in the spike.)

- [ ] **Step 2: Write the two configs**

Create `configs/lora_pllum_full.yaml`:
```yaml
model: "/Users/greg/.cache/huggingface/local-mlx/Llama-PLLuM-8B-instruct-2512-mlx-8bit"
train: true
data: "data/finetune"
adapter_path: "adapters/pllum8_full"
fine_tune_type: lora
num_layers: 16
iters: 400
batch_size: 1
learning_rate: 1.0e-4
steps_per_report: 25
steps_per_eval: 25
save_every: 25
max_seq_length: 2048
seed: 42
lora_parameters:
  rank: 8
  scale: 20.0
  dropout: 0.05
```
Create `configs/lora_bielik_full.yaml` — identical EXCEPT:
```yaml
model: "agentGreg/Bielik-Minitron-7B-v3.0-Instruct-MLX-bf16"
adapter_path: "adapters/bielik_full"
```
(`steps_per_eval == save_every == 25` is what makes Task 5's selection work. `dropout: 0.05` adds mild regularization vs the spike's 0.0.)

- [ ] **Step 3: Train PLLuM, then select best checkpoint**

Run:
```bash
uv run python -m mlx_lm lora --config configs/lora_pllum_full.yaml 2>&1 | tee temp/lora_pllum_full.log
uv run python scripts/12_wybierz_checkpoint.py adapters/pllum8_full temp/lora_pllum_full.log
```
Expected: training logs show val loss every 25 iters; selector prints the chosen best iter and copies it to `adapters.safetensors`.

- [ ] **Step 4: Train Bielik, then select best checkpoint**

Run:
```bash
uv run python -m mlx_lm lora --config configs/lora_bielik_full.yaml 2>&1 | tee temp/lora_bielik_full.log
uv run python scripts/12_wybierz_checkpoint.py adapters/bielik_full temp/lora_bielik_full.log
```
Expected: same — best checkpoint copied. (Bielik bf16 ~14 GB; comfortable on 128 GB.)

- [ ] **Step 5: Commit configs** (adapters are gitignored)

```bash
git add configs/lora_pllum_full.yaml configs/lora_bielik_full.yaml
git commit -m "feat: early-stopping LoRA configs for PLLuM + Bielik full study"
```

---

## Task 7: Run base vs fine-tuned on all three test sheets

**Files:**
- Output: `results/studium/<model>_<sheet>.json` for model ∈ {pllum_base, pllum_ft, bielik_base, bielik_ft}, sheet ∈ {2024, 2025, 2026}

Uses `lib_inferencja.run_sheet`. 2026 reads `data/zadania.json`; 2024/2025 read `data/testy/<year>.json`.

- [ ] **Step 1: Write a small driver that runs all 12 (model × sheet) combos**

Create `scripts/07b_run_studium.py`:
```python
"""Faza 7b: base vs fine-tune, oba modele, trzy arkusze testowe (2024/2025/2026)."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.lib_inferencja import run_sheet  # noqa: E402

PLLUM = "/Users/greg/.cache/huggingface/local-mlx/Llama-PLLuM-8B-instruct-2512-mlx-8bit"
BIELIK = "agentGreg/Bielik-Minitron-7B-v3.0-Instruct-MLX-bf16"
SHEETS = {"2024": ROOT/"data/testy/2024.json", "2025": ROOT/"data/testy/2025.json",
          "2026": ROOT/"data/zadania.json"}
JOBS = [
    ("pllum_base", PLLUM, None), ("pllum_ft", PLLUM, str(ROOT/"adapters/pllum8_full")),
    ("bielik_base", BIELIK, None), ("bielik_ft", BIELIK, str(ROOT/"adapters/bielik_full")),
]
OUT = ROOT/"results"/"studium"; OUT.mkdir(parents=True, exist_ok=True)
for name, model, adapter in JOBS:
    for sheet, path in SHEETS.items():
        run_sheet(model, adapter, path, OUT/f"{name}_{sheet}.json")
```

- [ ] **Step 2: Run it**

Run: `uv run python scripts/07b_run_studium.py`
Expected: 12 result files written under `results/studium/`, each with 19–20 task entries.

- [ ] **Step 3: Commit**

```bash
git add scripts/07b_run_studium.py results/studium/
git commit -m "feat: run base vs fine-tune for both models across 2024-2026"
```

---

## Task 8: Multi-sheet decomposition aggregator

**Files:**
- Create: `scripts/13_ocen_studium.py`
- Create: `tests/test_ocen_studium.py`
- Output: `results/studium_decomposition.json`

Aggregates closed-task format-compliance + conditional-accuracy across the 3 sheets, per model, base vs ft. Keys come from each sheet's `odpowiedz` field (2024/2025 in `data/testy/`, 2026 in `data/klucz_odpowiedzi.json`).

- [ ] **Step 1: Write failing test for the aggregation helper**

Create `tests/test_ocen_studium.py`:
```python
import importlib.util
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("os13", ROOT / "scripts" / "13_ocen_studium.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def test_aggregate_pools_closed_across_sheets():
    m = _load()
    per_sheet = [
        ([{"id": 1, "typ": "zamkniete", "raw": "<odpowiedz>A</odpowiedz>", "odpowiedz": "A"}], {1: "A"}),
        ([{"id": 1, "typ": "zamkniete", "raw": "<odpowiedz>B</odpowiedz>", "odpowiedz": "B"}], {1: "C"}),
    ]
    fmt, acc, n = m.aggregate(per_sheet)
    assert n == 2            # 2 closed tasks pooled
    assert fmt == 1.0        # both strict-compliant
    assert acc == 0.5        # 1 of 2 correct
```

- [ ] **Step 2: Run, verify FAIL**

Run: `uv run pytest tests/test_ocen_studium.py -q`
Expected: FAIL — module missing.

- [ ] **Step 3: Implement the aggregator**

Create `scripts/13_ocen_studium.py`:
```python
"""Faza 13: dekompozycja forma-vs-matematyka, oba modele, pula 2024+2025+2026."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.metryki_format import strict_compliant, conditional_accuracy  # noqa: E402

SHEETS = {
    "2024": ROOT / "data/testy/2024.json",
    "2025": ROOT / "data/testy/2025.json",
    "2026": ROOT / "data/zadania.json",
}
KEY_2026 = ROOT / "data/klucz_odpowiedzi.json"
STUD = ROOT / "results/studium"
OUT = ROOT / "results/studium_decomposition.json"
MODELS = [("Llama-PLLuM 8B", "pllum_base", "pllum_ft"),
          ("Bielik-Minitron 7B", "bielik_base", "bielik_ft")]


def sheet_key(sheet: str) -> dict[int, str]:
    if sheet == "2026":
        return {k["id"]: k["odpowiedz"] for k in json.load(open(KEY_2026, encoding="utf-8"))}
    return {z["id"]: z["odpowiedz"] for z in json.load(open(SHEETS[sheet], encoding="utf-8"))}


def aggregate(per_sheet: list[tuple[list[dict], dict[int, str]]]):
    """per_sheet: list of (closed_rows, key). Returns (format_compliance, cond_acc, n_closed)."""
    rows_all, ids_all, keys_all = [], [], {}
    compliant = 0
    for rows, key in per_sheet:
        for r in rows:
            rows_all.append(r); ids_all.append(r["id"])
            keys_all[r["id"]] = key[r["id"]]
            if strict_compliant(r["raw"], r.get("odpowiedz", "")):
                compliant += 1
    n = len(rows_all)
    fmt = compliant / n if n else 0.0
    acc = conditional_accuracy(rows_all, ids_all, keys_all)
    return fmt, acc, n


def closed_rows(model_tag: str, sheet: str) -> list[dict]:
    data = json.load(open(STUD / f"{model_tag}_{sheet}.json", encoding="utf-8"))
    return [r for r in data if r["typ"] == "zamkniete"]


def main() -> None:
    out = []
    for label, base_tag, ft_tag in MODELS:
        for variant, tag in [("base", base_tag), ("+LoRA", ft_tag)]:
            per_sheet = [(closed_rows(tag, s), sheet_key(s)) for s in SHEETS]
            fmt, acc, n = aggregate(per_sheet)
            out.append({"model": label, "variant": variant,
                        "format_compliance": round(fmt, 3),
                        "conditional_accuracy": round(acc, 3), "n_closed": n})
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{'model':20} {'variant':7} {'format':>7} {'cond.acc':>9} {'n':>4}")
    for r in out:
        print(f"{r['model']:20} {r['variant']:7} {r['format_compliance']:>7} {r['conditional_accuracy']:>9} {r['n_closed']:>4}")
    print(f"OK: {OUT}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests, then the aggregator**

Run:
```bash
uv run pytest tests/test_ocen_studium.py -v
uv run python scripts/13_ocen_studium.py
```
Expected: test PASS; then a 4-row table (PLLuM base/+LoRA, Bielik base/+LoRA). **Read this:** PLLuM should show format up + accuracy ~flat; Bielik should show format already high + accuracy ~flat. That two-model pattern IS the publishable result.

- [ ] **Step 5: Commit**

```bash
git add scripts/13_ocen_studium.py tests/test_ocen_studium.py results/studium_decomposition.json
git commit -m "feat: multi-sheet two-model format/accuracy decomposition"
```

---

## Task 9: Two-model comparison graph

**Files:**
- Create: `scripts/14_wykresy_studium.py`
- Output: `results/wykresy/03_studium.png`

- [ ] **Step 1: Implement the graph (style from `scripts/11_wykresy.py`)**

Create `scripts/14_wykresy_studium.py`: read `results/studium_decomposition.json`; render a 1080×1350 grouped bar chart with two panels (format-compliance, conditional-accuracy), each showing four bars: PLLuM base, PLLuM +LoRA, Bielik base, Bielik +LoRA. Reuse colors/fonts/footer from `11_wykresy.py` (DejaVu Sans, PLLuM purple `#9333ea`, Bielik green `#16a34a`, grey `#94a3b8` for base, footer attribution string). Title: "Fine-tuning uczy formy, nie matematyki — oba modele". Save to `results/wykresy/03_studium.png`.

- [ ] **Step 2: Render and eyeball**

Run: `uv run python scripts/14_wykresy_studium.py`
Expected: PNG written; visually, the format panel shows PLLuM jumping and Bielik already-high, while the accuracy panel shows little movement for either.

- [ ] **Step 3: Commit**

```bash
git add scripts/14_wykresy_studium.py results/wykresy/03_studium.png
git commit -m "feat: two-model decomposition chart for the full study"
```

---

## Task 10: Report + README

**Files:**
- Modify: `results/raport.md` (new "Fine-tuning: forma vs matematyka" section)
- Modify: `README.md` (short subsection + link, in the existing voice)

- [ ] **Step 1: Write the report section**

Append to `results/raport.md` a section with: the study design (chronological split, early stopping), the two-model decomposition table from `results/studium_decomposition.json`, and the honest framing — format-compliance is what moves; conditional accuracy does not; the strong model (Bielik) had no format headroom to begin with. Note the spike's overfitting finding and how early stopping addressed it.

- [ ] **Step 2: Update README**

Add a brief subsection under the existing results, linking to the report and the `03_studium.png` chart, in Polish, matching the README's measured tone. Do NOT overclaim — scope the conclusion to "small models, this exam, this method."

- [ ] **Step 3: Commit**

```bash
git add results/raport.md README.md
git commit -m "docs: report + README for the fine-tuning study"
```

---

## Task 11 (optional): Publish adapters to Hugging Face

Only after the user approves. Mirror the existing Bielik-Minitron MLX release pattern in `publish/`: write an adapter card noting the format-not-math finding and the held-out test, then upload `adapters/pllum8_full` and `adapters/bielik_full`. Decide with the user first.

---

## Self-review notes

- **Spec coverage:** both models (Tasks 6–8), chronological held-out test 2024+2025+2026 (Tasks 2, 7), early stopping (Tasks 5–6), decomposition + graphs (Tasks 8–9), report/README (Task 10), optional HF (Task 11). Carries both spike findings (overfitting → early stopping; 2023 re-transcribe → Task 1).
- **Placeholder scan:** transcription tasks (1–2) are genuine data entry from source PDFs; graph styling (Task 9) and report prose (Task 10) reference the concrete source file `11_wykresy.py` and the concrete data file `studium_decomposition.json` rather than leaving content vague.
- **Type/name consistency:** `run_sheet(model_id, adapter_path, zadania_path, out_path)`, `build_user_prompt(z)`, `extract_odpowiedz(text)`, `parse_val_losses`, `best_iter`, `aggregate(per_sheet) -> (fmt, acc, n)` used identically across tasks; result paths `results/studium/<model>_<sheet>.json` and model tags (`pllum_base/pllum_ft/bielik_base/bielik_ft`) match between Tasks 7 and 8.
- **Judge dependency removed from the headline:** Tasks 8–9 use only closed-task keys, so no `05_ocen.py` multi-sheet refactor is required; open-task totals remain an optional later add-on.
