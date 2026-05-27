# Fine-tune Spike (PLLuM, format-not-math) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove the full train→eval loop end-to-end on Llama-PLLuM 8B 2512 — LoRA-fine-tune it on 2–3 transcribed past CKE exams, then measure format-compliance vs conditional-accuracy on the held-out 2026 sheet — before investing in the full-archive extractor.

**Architecture:** Reuse the existing flat numbered-script + `ROOT/results/*.json` pattern. Extract the answer parser from `05_ocen.py` into a shared module so it can be consumed by a new metrics module. Build an MLX chat-format dataset from hand-transcribed past sheets, run `mlx_lm.lora` on the existing 8-bit PLLuM base, evaluate the base model + adapter through a copy of the existing run script, and score base-vs-fine-tuned with the format/accuracy decomposition.

**Tech Stack:** Python 3, `mlx_lm` 0.31.3 (LoRA CLI + adapter loading), `pytest` (new dev dep), `uv` for running, Anthropic SDK (existing judge in `05_ocen.py`). Apple M5 Max / 128 GB.

**Non-goals (deferred to Plan 2):** robust multi-year PDF extraction, Bielik-Minitron arm, full 2024/2025 held-out test set, HF publication of adapters, graphs.

---

## File structure

| Path | Responsibility |
|---|---|
| `scripts/parser_odpowiedzi.py` | **New.** Shared answer parser: `normalize`, `extract_odpowiedz` (strict `<odpowiedz>`), `wylusk_zamkniete` (generous). Moved verbatim from `05_ocen.py`. |
| `scripts/05_ocen.py` | **Modify.** Import parser from `scripts.parser_odpowiedzi` instead of defining it inline. Add the fine-tuned PLLuM entry to `MODELE`. |
| `scripts/metryki_format.py` | **New.** Format-compliance + conditional-accuracy computation over a results JSON + key. No model loading. |
| `scripts/08_buduj_dataset.py` | **New.** Build `data/finetune/{train,valid}.jsonl` (MLX chat format) from `data/archiwum/*.json`. |
| `scripts/04e_ft_run_pllum8.py` | **New.** Copy of `04e_run_pllum8_2512.py` that loads base + LoRA adapter and writes `results/pllum8_2512_ft_odpowiedzi.json`. |
| `scripts/10_ocen_finetune.py` | **New.** Compute base-vs-fine-tuned decomposition (format-compliance, conditional-accuracy, total score) → `results/finetune_decomposition.json` + printed table. |
| `configs/lora_pllum_spike.yaml` | **New.** `mlx_lm.lora` config (rank, layers, iters, lr, data path, adapter path). |
| `data/archiwum/<year>.json` | **New (data).** Transcribed past sheets: `zadania.json` schema + `odpowiedz` + `rozwiazanie`. |
| `tests/test_parser_odpowiedzi.py` | **New.** Characterization tests locking current parser behavior before the move. |
| `tests/test_metryki_format.py` | **New.** Unit tests for compliance/accuracy math. |
| `tests/test_buduj_dataset.py` | **New.** Shape tests for the emitted JSONL. |

---

## Task 1: Add pytest and a tests package

**Files:**
- Modify: `pyproject.toml`
- Create: `tests/__init__.py` (empty)
- Create: `tests/conftest.py`

- [ ] **Step 1: Add pytest as a dev dependency**

Run:
```bash
uv add --dev pytest
```
Expected: `pyproject.toml` gains a `[dependency-groups]`/`dev` entry with `pytest`; `uv.lock` updates; exit 0.

- [ ] **Step 2: Create the tests package marker**

Create `tests/__init__.py` with no content (empty file).

- [ ] **Step 3: Make `scripts/` importable from tests**

Create `tests/conftest.py`:
```python
import sys
from pathlib import Path

# Allow `import scripts.<module>` from the repo root in tests.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
```

- [ ] **Step 4: Create the scripts package marker**

Create `scripts/__init__.py` with no content (empty file). This lets `scripts.parser_odpowiedzi` import cleanly. (Existing numbered scripts are run as files, so an `__init__.py` does not affect them.)

- [ ] **Step 5: Verify pytest collects nothing yet (no error)**

Run: `uv run pytest -q`
Expected: `no tests ran` (exit code 5 is acceptable here — it means "no tests collected", not a failure).

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock tests/__init__.py tests/conftest.py scripts/__init__.py
git commit -m "test: add pytest dev dependency and tests scaffold"
```

---

## Task 2: Characterize the existing parser (lock behavior before moving it)

**Files:**
- Create: `tests/test_parser_odpowiedzi.py`

These tests assert the CURRENT behavior of `wylusk_zamkniete` / `extract_odpowiedz` as defined in `05_ocen.py` and `04e_run_pllum8_2512.py`. We import them from `05_ocen.py` via `importlib` because the module name starts with a digit. The tests must pass against the unmodified code first.

- [ ] **Step 1: Write characterization tests**

Create `tests/test_parser_odpowiedzi.py`:
```python
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load_05():
    spec = importlib.util.spec_from_file_location("ocen05", ROOT / "scripts" / "05_ocen.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_strict_tag_letter():
    m = _load_05()
    assert m.wylusk_zamkniete("blah <odpowiedz>C</odpowiedz>", "C") == "C"


def test_clean_field_short_circuits():
    m = _load_05()
    assert m.wylusk_zamkniete("irrelevant text", "BD") == "BD"


def test_boxed_fallback():
    m = _load_05()
    assert m.wylusk_zamkniete(r"final \boxed{A}", "") == "A"


def test_gsm8k_hash_fallback():
    m = _load_05()
    assert m.wylusk_zamkniete("rozumowanie...\n#### D", "") == "D"


def test_enumerated_last_line():
    m = _load_05()
    assert m.wylusk_zamkniete("A. zle\nB. tez zle\nC. dobrze", "") == "C"


def test_normalize_strips_ws():
    m = _load_05()
    assert m.normalize(" a c ") == "AC"


def test_no_match_returns_field_upper():
    m = _load_05()
    assert m.wylusk_zamkniete("kompletnie bez litery", "") == ""
```

- [ ] **Step 2: Run against unmodified code, verify PASS**

Run: `uv run pytest tests/test_parser_odpowiedzi.py -v`
Expected: all 7 tests PASS (they describe the code as it is today).

- [ ] **Step 3: Commit**

```bash
git add tests/test_parser_odpowiedzi.py
git commit -m "test: characterize answer parser before extraction"
```

---

## Task 3: Extract the parser into a shared module (DRY)

**Files:**
- Create: `scripts/parser_odpowiedzi.py`
- Modify: `scripts/05_ocen.py:50-94` (remove inline `normalize` + `wylusk_zamkniete`, import instead)
- Modify: `tests/test_parser_odpowiedzi.py` (point at the new module)

- [ ] **Step 1: Create the shared parser module**

Create `scripts/parser_odpowiedzi.py` (copy the functions verbatim from `05_ocen.py` and add `extract_odpowiedz` from `04e_run_pllum8_2512.py`):
```python
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

    for pat in [
        r"####\s*([A-DPF]{1,2})\b",
        r"(?:odpowied[zź]\s*to|wynik|final\w*)\s*[:\-]?\s*\**\s*([A-DPF]{1,2})\b",
    ]:
        m = re.search(pat, raw, re.I)
        if m:
            return m.group(1).upper()
    enum = re.findall(r"(?m)^\s*([A-DPF]{1,2})[.):]", raw)
    if enum:
        return enum[-1].upper()

    return (odp or "").strip().upper()
```

- [ ] **Step 2: Replace the inline parser in `05_ocen.py`**

In `scripts/05_ocen.py`, delete the inline `normalize` and `wylusk_zamkniete` definitions (lines 50-94) and add this import near the other imports (after line 19):
```python
from scripts.parser_odpowiedzi import normalize, wylusk_zamkniete
```
Note: `05_ocen.py` is run as `uv run python scripts/05_ocen.py` from repo root, so `import scripts.parser_odpowiedzi` resolves via the repo-root entry on `sys.path[0]`. If running as a file does not put repo root on the path, add at the top of `05_ocen.py` (after `from pathlib import Path`):
```python
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
```

- [ ] **Step 3: Update tests to target the shared module**

Replace the body of `tests/test_parser_odpowiedzi.py` so every test imports from the shared module instead of loading `05_ocen.py`:
```python
from scripts.parser_odpowiedzi import normalize, wylusk_zamkniete


def test_strict_tag_letter():
    assert wylusk_zamkniete("blah <odpowiedz>C</odpowiedz>", "C") == "C"


def test_clean_field_short_circuits():
    assert wylusk_zamkniete("irrelevant text", "BD") == "BD"


def test_boxed_fallback():
    assert wylusk_zamkniete(r"final \boxed{A}", "") == "A"


def test_gsm8k_hash_fallback():
    assert wylusk_zamkniete("rozumowanie...\n#### D", "") == "D"


def test_enumerated_last_line():
    assert wylusk_zamkniete("A. zle\nB. tez zle\nC. dobrze", "") == "C"


def test_normalize_strips_ws():
    assert normalize(" a c ") == "AC"


def test_no_match_returns_field_upper():
    assert wylusk_zamkniete("kompletnie bez litery", "") == ""
```

- [ ] **Step 4: Run parser tests, verify PASS**

Run: `uv run pytest tests/test_parser_odpowiedzi.py -v`
Expected: all 7 PASS against the shared module.

- [ ] **Step 5: Smoke-test that `05_ocen.py` still imports**

Run: `uv run python -c "import importlib.util,pathlib; s=importlib.util.spec_from_file_location('o', pathlib.Path('scripts/05_ocen.py')); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); print('import ok', m.wylusk_zamkniete('x <odpowiedz>A</odpowiedz>','A'))"`
Expected: prints `import ok A` (confirms the refactored module loads and the imported parser works).

- [ ] **Step 6: Commit**

```bash
git add scripts/parser_odpowiedzi.py scripts/05_ocen.py tests/test_parser_odpowiedzi.py
git commit -m "refactor: extract shared answer parser from 05_ocen"
```

---

## Task 4: Format-compliance + conditional-accuracy metrics

**Files:**
- Create: `scripts/metryki_format.py`
- Create: `tests/test_metryki_format.py`

Definitions (closed tasks only for compliance/strict; open tasks handled in the eval script via the judge):
- **strict-compliant** = `extract_odpowiedz(raw)` is non-empty AND matches `^[A-DPF]{1,2}$` after `normalize`.
- **generous-parsed** = `wylusk_zamkniete(raw, odp)` is non-empty.
- **conditional-accuracy** = (# closed where generous-parsed answer == key) / (# closed generous-parsed).

- [ ] **Step 1: Write failing tests**

Create `tests/test_metryki_format.py`:
```python
from scripts.metryki_format import (
    strict_compliant,
    generous_parsed,
    format_compliance,
    conditional_accuracy,
)


def test_strict_compliant_true():
    assert strict_compliant("rozumowanie <odpowiedz>C</odpowiedz>", "C") is True


def test_strict_compliant_false_when_no_tag():
    assert strict_compliant("po prostu C na końcu", "") is False


def test_strict_compliant_false_when_tag_garbage():
    assert strict_compliant("<odpowiedz>nie wiem</odpowiedz>", "") is False


def test_generous_parsed_true_via_fallback():
    assert generous_parsed("...\n#### D", "") is True


def test_format_compliance_counts_strict():
    rows = [
        {"raw": "<odpowiedz>A</odpowiedz>", "odpowiedz": "A"},
        {"raw": "luzem B na końcu B", "odpowiedz": ""},
    ]
    # 1 of 2 strict-compliant
    assert format_compliance(rows) == 0.5


def test_conditional_accuracy_ignores_unparseable():
    rows = [
        {"raw": "<odpowiedz>A</odpowiedz>", "odpowiedz": "A"},  # parsed, correct
        {"raw": "<odpowiedz>B</odpowiedz>", "odpowiedz": "B"},  # parsed, wrong
        {"raw": "zupelnie bez litery", "odpowiedz": ""},        # unparseable -> excluded
    ]
    keys = {1: "A", 2: "A", 3: "C"}
    ids = [1, 2, 3]
    # parsed: rows 1,2 -> 1 correct of 2 = 0.5
    assert conditional_accuracy(rows, ids, keys) == 0.5
```

- [ ] **Step 2: Run, verify FAIL**

Run: `uv run pytest tests/test_metryki_format.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts.metryki_format'`.

- [ ] **Step 3: Implement the metrics module**

Create `scripts/metryki_format.py`:
```python
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
```

- [ ] **Step 4: Run, verify PASS**

Run: `uv run pytest tests/test_metryki_format.py -v`
Expected: all 6 PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/metryki_format.py tests/test_metryki_format.py
git commit -m "feat: format-compliance and conditional-accuracy metrics"
```

---

## Task 5: Transcribe 2–3 past sheets into the archive schema

**Files:**
- Create: `data/archiwum/2023_glowny.json`
- Create: `data/archiwum/2022_glowny.json`
- (Optional third) `data/archiwum/2021_glowny.json`

Each file is a JSON array using the EXACT `data/zadania.json` schema plus two fields per item: `odpowiedz` (official key) and `rozwiazanie` (official worked solution from *zasady oceniania*, plain text). These are the held-OUT-of-test training years (2026 is the test sheet, never transcribed here).

- [ ] **Step 1: Obtain the source PDFs**

Run (URLs follow the arkusze.pl pattern already in the README; if a URL 404s, the user will drop the PDF into `temp/` manually):
```bash
mkdir -p temp/archiwum
curl -fL -o temp/archiwum/2023_glowny.pdf "https://arkusze.pl/osmoklasisty/matematyka-2023-egzamin-osmoklasisty.pdf" || echo "FETCH FAILED 2023 — ask user to provide"
curl -fL -o temp/archiwum/2022_glowny.pdf "https://arkusze.pl/osmoklasisty/matematyka-2022-egzamin-osmoklasisty.pdf" || echo "FETCH FAILED 2022 — ask user to provide"
```
Expected: two PDFs in `temp/archiwum/`. If either FETCH FAILED, STOP and ask the user to supply that year's *arkusz* + *zasady oceniania* PDF before continuing.

- [ ] **Step 2: Transcribe each sheet (Claude-assisted) into the archive schema**

For each PDF, read the *arkusz* and *zasady oceniania* and produce `data/archiwum/<year>_glowny.json`. One element per task; figure tasks get a neutral, non-leading `opis_obrazka` (same style as `data/zadania.json`). Example element shape (values illustrative — fill from the actual sheet):
```json
{
  "id": 1,
  "typ": "zamkniete",
  "punkty_max": 1,
  "tresc": "<dokładna treść zadania z arkusza>",
  "opcje": {"A": "...", "B": "...", "C": "...", "D": "..."},
  "obrazek": "",
  "opis_obrazka": "",
  "podtyp": "ABCD",
  "odpowiedz": "C",
  "rozwiazanie": "<oficjalne rozwiązanie krok po kroku z zasad oceniania>"
}
```
Prioritize text-only tasks; for the spike, figure-heavy tasks may be omitted if the figure can't be described neutrally in one or two sentences. Target ≥ 30 usable tasks across the sheets.

- [ ] **Step 3: Validate the archive files load and carry both fields**

Run:
```bash
uv run python -c "import json,glob; tot=0
for f in sorted(glob.glob('data/archiwum/*.json')):
    a=json.load(open(f,encoding='utf-8'))
    assert all('odpowiedz' in z and 'rozwiazanie' in z and z['rozwiazanie'].strip() for z in a), f
    tot+=len(a); print(f, len(a))
print('TOTAL', tot); assert tot>=30, 'need >=30 examples for the spike'"
```
Expected: prints each file with its count and `TOTAL >= 30`, no assertion error.

- [ ] **Step 4: Commit**

```bash
git add data/archiwum/
git commit -m "data: transcribe 2-3 past CKE sheets for fine-tune spike"
```

---

## Task 6: Build the MLX chat-format dataset

**Files:**
- Create: `scripts/08_buduj_dataset.py`
- Create: `tests/test_buduj_dataset.py`
- Output: `data/finetune/train.jsonl`, `data/finetune/valid.jsonl`

Each JSONL line is `{"messages": [{"role":"system",...},{"role":"user",...},{"role":"assistant",...}]}`. System prompt comes from `prompts/system_{zamkniete,otwarte}.txt`; user prompt reuses the existing `build_user_prompt` logic; assistant target = `f"{rozwiazanie}\n<odpowiedz>{odpowiedz}</odpowiedz>"`.

- [ ] **Step 1: Write failing tests**

Create `tests/test_buduj_dataset.py`:
```python
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("bd08", ROOT / "scripts" / "08_buduj_dataset.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_build_user_prompt_includes_options():
    m = _load()
    z = {"tresc": "Treść.", "typ": "zamkniete", "opcje": {"A": "1", "B": "2"}, "opis_obrazka": ""}
    p = m.build_user_prompt(z)
    assert "Opcje odpowiedzi:" in p and "A. 1" in p


def test_make_example_shape():
    m = _load()
    z = {
        "id": 1, "typ": "zamkniete", "punkty_max": 1, "tresc": "Treść.",
        "opcje": {"A": "1", "B": "2"}, "opis_obrazka": "",
        "odpowiedz": "A", "rozwiazanie": "Bo 1 < 2.",
    }
    ex = m.make_example(z, sys_zamk="SYS-Z", sys_otw="SYS-O")
    roles = [msg["role"] for msg in ex["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert ex["messages"][0]["content"] == "SYS-Z"
    assert ex["messages"][2]["content"].endswith("<odpowiedz>A</odpowiedz>")
    assert "Bo 1 < 2." in ex["messages"][2]["content"]


def test_make_example_uses_open_system_prompt():
    m = _load()
    z = {"id": 15, "typ": "otwarte", "punkty_max": 2, "tresc": "Oblicz.",
         "opcje": {}, "opis_obrazka": "", "odpowiedz": "42", "rozwiazanie": "Liczymy."}
    ex = m.make_example(z, sys_zamk="SYS-Z", sys_otw="SYS-O")
    assert ex["messages"][0]["content"] == "SYS-O"
```

- [ ] **Step 2: Run, verify FAIL**

Run: `uv run pytest tests/test_buduj_dataset.py -v`
Expected: FAIL — module file does not exist yet.

- [ ] **Step 3: Implement the dataset builder**

Create `scripts/08_buduj_dataset.py`:
```python
"""Faza 8: budowa zbioru treningowego (MLX chat format) z data/archiwum/*.json.

Każda linia: {"messages": [system, user, assistant]}.
Cel asystenta = oficjalne rozwiązanie + <odpowiedz>...</odpowiedz>.
Split train/valid 90/10 deterministyczny (sort po (plik, id)).
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
```

- [ ] **Step 4: Run tests, verify PASS**

Run: `uv run pytest tests/test_buduj_dataset.py -v`
Expected: all 3 PASS.

- [ ] **Step 5: Generate the dataset and sanity-check it**

Run:
```bash
uv run python scripts/08_buduj_dataset.py
uv run python -c "import json; L=[json.loads(l) for l in open('data/finetune/train.jsonl',encoding='utf-8')]; print('train', len(L)); print('roles', [m['role'] for m in L[0]['messages']]); assert all(len(x['messages'])==3 for x in L)"
```
Expected: prints `train N`, `roles ['system','user','assistant']`, no assertion error; both `train.jsonl` and `valid.jsonl` exist.

- [ ] **Step 6: Commit**

```bash
git add scripts/08_buduj_dataset.py tests/test_buduj_dataset.py data/finetune/.gitignore
git commit -m "feat: build MLX chat dataset from past-exam archive"
```
(Create `data/finetune/.gitignore` containing `*.jsonl` first — generated artifacts stay out of git.)

---

## Task 7: LoRA-fine-tune Llama-PLLuM 8B on the 8-bit base

**Files:**
- Create: `configs/lora_pllum_spike.yaml`
- Output: `adapters/pllum8_spike/` (adapter weights + `adapter_config.json`)

Prerequisite: the 8-bit MLX PLLuM exists at `~/.cache/huggingface/local-mlx/Llama-PLLuM-8B-instruct-2512-mlx-8bit` (produced by the existing `temp/run_pllum_2512.sh`). If absent, run that conversion first.

- [ ] **Step 1: Confirm the base model is present**

Run:
```bash
ls ~/.cache/huggingface/local-mlx/Llama-PLLuM-8B-instruct-2512-mlx-8bit/ | head
```
Expected: lists `config.json`, `*.safetensors`, tokenizer files. If empty, run the existing conversion (`bash temp/run_pllum_2512.sh`) before continuing.

- [ ] **Step 2: Write the LoRA config**

Create `configs/lora_pllum_spike.yaml`:
```yaml
model: "~/.cache/huggingface/local-mlx/Llama-PLLuM-8B-instruct-2512-mlx-8bit"
train: true
data: "data/finetune"
adapter_path: "adapters/pllum8_spike"
fine_tune_type: lora
num_layers: 16
iters: 300
batch_size: 1
learning_rate: 1.0e-4
steps_per_report: 20
steps_per_eval: 60
save_every: 100
max_seq_length: 2048
seed: 42
lora_parameters:
  rank: 8
  scale: 20.0
  dropout: 0.0
```

- [ ] **Step 3: Run LoRA training**

Run:
```bash
uv run python -m mlx_lm lora --config configs/lora_pllum_spike.yaml 2>&1 | tee temp/lora_pllum_spike.log
```
Expected: training logs show `Iter` lines with train + validation loss; validation loss trends DOWN over iterations; ends writing adapters to `adapters/pllum8_spike/`. If validation loss does not decrease at all, lower `learning_rate` to `5.0e-5` or raise `iters`, and note it in the spike summary.

- [ ] **Step 4: Confirm the adapter was written**

Run: `ls adapters/pllum8_spike/`
Expected: contains `adapters.safetensors` (or numbered checkpoints) and `adapter_config.json`.

- [ ] **Step 5: Commit config (adapters stay out of git)**

```bash
echo "adapters/" >> .gitignore
git add configs/lora_pllum_spike.yaml .gitignore
git commit -m "feat: LoRA config for PLLuM spike; ignore adapters/"
```

---

## Task 8: Evaluate base + adapter on the held-out 2026 sheet

**Files:**
- Create: `scripts/04e_ft_run_pllum8.py`
- Output: `results/pllum8_2512_ft_odpowiedzi.json`

This is `04e_run_pllum8_2512.py` with two changes: load the adapter, and write to the `_ft_` results file.

- [ ] **Step 1: Create the fine-tuned run script**

Create `scripts/04e_ft_run_pllum8.py` as a copy of `scripts/04e_run_pllum8_2512.py` with these exact edits:
- Change the `OUT` line to:
```python
OUT = ROOT / "results" / "pllum8_2512_ft_odpowiedzi.json"
```
- Add below the `MODEL_ID` line:
```python
ADAPTER = str(ROOT / "adapters" / "pllum8_spike")
```
- Change the model load call from `model, tokenizer = load(MODEL_ID)` to:
```python
model, tokenizer = load(MODEL_ID, adapter_path=ADAPTER)
```

- [ ] **Step 2: Run inference on the 2026 test sheet**

Run:
```bash
uv run python scripts/04e_ft_run_pllum8.py
```
Expected: per-task log lines `z01..z20`, then `OK: .../pllum8_2512_ft_odpowiedzi.json`. Crucially, `odp=` values should now mostly be clean letters in `<odpowiedz>` (format learned), versus the base model's free-form output.

- [ ] **Step 3: Quick eyeball — did format compliance move?**

Run:
```bash
uv run python -c "
from scripts.metryki_format import format_compliance
import json
base=json.load(open('results/pllum8_2512_odpowiedzi.json',encoding='utf-8'))
ft=json.load(open('results/pllum8_2512_ft_odpowiedzi.json',encoding='utf-8'))
clz=lambda d:[r for r in d if r['typ']=='zamkniete']
print('strict format-compliance (closed):')
print('  base', round(format_compliance(clz(base)),3))
print('  ft  ', round(format_compliance(clz(ft)),3))
"
```
Expected: `ft` compliance noticeably higher than `base` (the core spike signal). Record both numbers.

- [ ] **Step 4: Commit the run script and results**

```bash
git add scripts/04e_ft_run_pllum8.py results/pllum8_2512_ft_odpowiedzi.json
git commit -m "feat: eval PLLuM 8B + LoRA adapter on held-out 2026 sheet"
```

---

## Task 9: Decomposition report (base vs fine-tuned)

**Files:**
- Create: `scripts/10_ocen_finetune.py`
- Output: `results/finetune_decomposition.json`

Reuses the closed-task key from `data/klucz_odpowiedzi.json` for compliance + conditional accuracy. Total score (incl. open tasks via the judge) is pulled from `results/ocena_szczegolowa.json` if present, otherwise reported as closed-only.

- [ ] **Step 1: Register the fine-tuned model in the judge pipeline**

In `scripts/05_ocen.py`, add to the `MODELE` list (after the `pllum4_2512` entry):
```python
    ("llama_pllum8_2512_ft", ROOT / "results" / "pllum8_2512_ft_odpowiedzi.json"),
```

- [ ] **Step 2: Run the full judge so open-task scores exist for base + ft**

Run:
```bash
uv run python scripts/05_ocen.py
```
Expected: prints a per-model line including `llama_pllum8_2512` and `llama_pllum8_2512_ft` with `/30` totals; writes `results/ocena_szczegolowa.json`.

- [ ] **Step 3: Implement the decomposition script**

Create `scripts/10_ocen_finetune.py`:
```python
"""Faza 10: dekompozycja forma-vs-matematyka, baza vs fine-tune.

Liczy dla pary (base, ft):
- format_compliance (zamknięte, ścisłe)
- conditional_accuracy (zamknięte, na sparsowanych)
- total_score (z ocena_szczegolowa.json jeśli jest)
Wynik: results/finetune_decomposition.json + tabela na stdout.
"""
from __future__ import annotations

import json
from pathlib import Path

from scripts.metryki_format import format_compliance, conditional_accuracy

ROOT = Path(__file__).resolve().parents[1]
KLUCZ = ROOT / "data" / "klucz_odpowiedzi.json"
OCENA = ROOT / "results" / "ocena_szczegolowa.json"
ZADANIA = ROOT / "data" / "zadania.json"
OUT = ROOT / "results" / "finetune_decomposition.json"

PARY = [
    ("PLLuM-8B base", "results/pllum8_2512_odpowiedzi.json", "llama_pllum8_2512"),
    ("PLLuM-8B +LoRA", "results/pllum8_2512_ft_odpowiedzi.json", "llama_pllum8_2512_ft"),
]


def closed_rows_and_ids(path: Path):
    data = json.load(open(path, encoding="utf-8"))
    rows = [r for r in data if r["typ"] == "zamkniete"]
    ids = [r["id"] for r in rows]
    return rows, ids


def main() -> None:
    klucz = {k["id"]: k["odpowiedz"] for k in json.load(open(KLUCZ, encoding="utf-8"))}
    totals = {}
    if OCENA.exists():
        oc = json.load(open(OCENA, encoding="utf-8"))
        totals = {m: oc[m]["suma"] for m in oc.get("modele", []) if m in oc}

    out = []
    for label, rel, judge_key in PARY:
        rows, ids = closed_rows_and_ids(ROOT / rel)
        out.append({
            "model": label,
            "format_compliance": round(format_compliance(rows), 3),
            "conditional_accuracy": round(conditional_accuracy(rows, ids, klucz), 3),
            "total_score": totals.get(judge_key),
        })

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{'model':16} {'format':>8} {'cond.acc':>9} {'score/30':>9}")
    for r in out:
        s = "-" if r["total_score"] is None else r["total_score"]
        print(f"{r['model']:16} {r['format_compliance']:>8} {r['conditional_accuracy']:>9} {str(s):>9}")
    print(f"OK: {OUT}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the decomposition**

Run:
```bash
uv run python scripts/10_ocen_finetune.py
```
Expected: a two-row table. The spike CONFIRMS the hypothesis if `format_compliance` rises substantially from base→ft while `conditional_accuracy` stays roughly flat (and `total_score` barely moves). Record the four numbers.

- [ ] **Step 5: Commit**

```bash
git add scripts/05_ocen.py scripts/10_ocen_finetune.py results/finetune_decomposition.json results/ocena_szczegolowa.json
git commit -m "feat: base-vs-finetune format/accuracy decomposition report"
```

---

## Task 10: Spike decision gate

- [ ] **Step 1: Run the whole suite once**

Run: `uv run pytest -q`
Expected: all tests PASS.

- [ ] **Step 2: Summarize the spike outcome**

Write a short note (in the chat, and append to the spec's "First step" section) stating the four numbers from Task 9 Step 4 and one of:
- **GO:** format-compliance rose markedly, conditional-accuracy ~flat → hypothesis holds; proceed to Plan 2 (full archive extractor, Bielik arm, 2024/2025 test, graphs).
- **GO (flipped):** conditional-accuracy also rose materially → honest finding changes; Plan 2 keeps the same machinery but reframes the narrative.
- **NO-GO:** loss didn't converge / compliance didn't move → debug training (lr, iters, data quality) before any scale-up.

- [ ] **Step 3: Decide with the user whether to write Plan 2**

Do not start Plan 2 work in this plan. Present the spike numbers and the GO/NO-GO recommendation, and let the user decide.

---

## Self-review notes

- **Spec coverage (spike scope):** hypothesis (Task 9–10), PLLuM target (Tasks 7–8), chronological no-leak split — 2026 is test only, training years transcribed in Task 5 (Task 5), LoRA via `mlx_lm.lora` (Task 7), format/conditional decomposition metrics (Tasks 4, 9), reuse of existing eval pipeline (Task 9 Step 2). Bielik arm, full archive, graphs, HF publication are explicitly deferred to Plan 2 (header non-goals).
- **Placeholder scan:** transcription content in Task 5 is genuine human/Claude data entry from the source PDFs, not a code placeholder; LoRA hyperparameters are concrete (config in Task 7) with a documented fallback if loss stalls.
- **Type/name consistency:** `format_compliance(rows)`, `conditional_accuracy(rows, ids, keys)`, `strict_compliant(raw, odp)`, `generous_parsed(raw, odp)`, `make_example(z, sys_zamk, sys_otw)`, `build_user_prompt(z)` are used identically wherever referenced; results filenames (`pllum8_2512_ft_odpowiedzi.json`) and judge key (`llama_pllum8_2512_ft`) match across Tasks 8–9.
