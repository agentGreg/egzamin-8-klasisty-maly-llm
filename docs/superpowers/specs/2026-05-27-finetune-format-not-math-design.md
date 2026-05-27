# Fine-tuning small LLMs on past CKE exams: "The Uniform, Not the Math"

**Date:** 2026-05-27
**Author:** Grzegorz Brzezinka (with Claude)
**Status:** Design approved, pending implementation plan
**Repo:** egzamin-8-klasisty

## Summary

We LoRA-fine-tune two small Polish-capable LLMs on the archive of past CKE
8th-grade math exams, then re-evaluate them on a held-out set of recent exams
(including the 2026 sheet already used in the benchmark). The goal is **not** to
top the leaderboard. It is to measure, cleanly, what fine-tuning on past exams
actually buys a small model.

## Hypothesis

> LoRA fine-tuning on past CKE exams raises a model's **format compliance**
> sharply, but leaves its **mathematical ability** roughly unchanged. The model
> learns the uniform of the exam, not the math.

- **PLLuM** demonstrates the effect: it currently ignores the answer format and
  computes free-form (GSM8K style), scoring 3/30. We expect format compliance to
  jump from low to high, while conditional accuracy (correctness *given* a
  parseable answer) stays low.
- **Bielik-Minitron 7B** is the control: already format-compliant at 25/30, so it
  has little compliance headroom. We expect a small or null change — evidence
  that for the strong model this was never a format problem.

If conditional accuracy *does* rise materially, the finding flips honestly to
"fine-tuning helped the math too" — that is still a publishable result. We report
whatever the data shows.

## Models (2 targets)

| Model | Base score (2026) | Role | Tractability |
|---|---|---|---|
| **Llama-PLLuM 8B Instruct 2512** (CYFRAGOVPL) | 3/30 | "Rescue" subject | `LlamaForCausalLM`, standard Llama 3.1 arch → straightforward `mlx_lm.lora` |
| **Bielik-Minitron 7B v3 Instruct** (SpeakLeash) | 25/30 | Control | bf16 MLX weights already published at `agentGreg/Bielik-Minitron-7B-v3.0-Instruct-MLX-bf16` |

**Out of scope:** Gemma 4 E4B (multimodal, loaded via `mlx-vlm`; LoRA support
there is immature). Noted as a future stretch, not part of this study.

## Data and split (chronological, leak-free)

**Source.** For each exam: CKE *arkusz* (questions + answer key) and *zasady
oceniania rozwiązań zadań* (marking scheme with worked solutions for open tasks).

**Sheets in scope:** egzamin ósmoklasisty z matematyki, terminy główne and
terminy dodatkowe 2019–2025, plus CKE trial exams (*egzaminy próbne*) and the
*informator* example tasks.

**Split:**

- **Training pool:** all sheets from 2019–2023 (main + dodatkowy) + trial exams
  + informator. Target ~200+ question/solution pairs.
- **Held-out test:** 2024 main + 2025 main + **2026 main** (~60 questions). The
  2026 sheet is kept in the test set so results stay directly comparable to the
  published leaderboard; 2024 + 2025 add statistical power against the n=20
  single-sheet noise.
- **Excluded entirely:** termin dodatkowy for 2024 and 2025 (same-year stylistic
  proximity to test → leakage risk). Not used for training or test.

**Contamination guard:** the model never sees any 2024/2025/2026 material during
training. Split is strictly chronological (train on the past, predict the
future), which is the most defensible framing.

## Data pipeline

Reuse and generalize the existing extraction scripts (`01_przygotuj_zadania.py`,
`02_klucz_claude.py`), which already do this for the 2026 sheet.

For each archived sheet:
1. Extract per-question text + per-question figure PNG from the PDF.
2. Generate a **neutral text description** of each figure (`opis_obrazka`),
   matching the benchmark's text-only convention — concise, non-leading, never
   hinting the answer. Claude-assisted, same style as the 2026 descriptions.
3. Capture the official answer key (closed tasks) and the official worked
   solution (open tasks) from *zasady oceniania*.
4. Reformat each official solution into the **scored output shape**: brief
   CKE-style reasoning ending in `<odpowiedz>…</odpowiedz>` (a letter / PP pair /
   two-choice for closed; method + result for open).

The training example is then a chat record using the **exact system prompt from
the benchmark** (`prompts/system_*.txt`):

```
system:    <identical benchmark system prompt>
user:      <question text> + <figure description if any> + <options if closed>
assistant: <reformatted official solution ending in <odpowiedz>…</odpowiedz>>
```

Output: `data/finetune/train.jsonl` and `data/finetune/valid.jsonl`
(MLX chat format), plus a manifest recording which sheet each example came from
(for auditability and the contamination guard).

## Training (identical recipe per model)

- **Method:** LoRA SFT via `mlx_lm.lora` (`mlx_lm` 0.31.3, LoRA CLI confirmed
  present).
- **Base weights:** bf16 MLX. Bielik-Minitron bf16 is already published;
  Llama-PLLuM 8B 2512 is converted locally to bf16 MLX (same conversion path as
  the existing benchmark scripts).
- **Config:** low LoRA rank (~8–16), attention + MLP projections, a few hundred
  iterations, small batch, validation split for early stopping. Tune iters to
  the validation loss, not to the test set.
- **Eval-time:** temperature 0, identical to the benchmark.
- **Post-training:** fuse adapter → quantize to 8-bit → evaluate through the
  **exact same benchmark eval scripts** so base-vs-fine-tuned is apples-to-apples.
- Two adapters total (one per model), same data, same recipe.

Hardware: Apple M5 Max, 128 GB. Training is expected to take minutes per model.

## Evaluation and metrics

Run **base vs fine-tuned** for each model through the existing eval pipeline
(Claude Opus 4.7 as judge, `05_ocen.py` logic) on the held-out test set. The same
judge and prompts are used for base and fine-tuned to keep the comparison fair.

Metrics, per model, base vs fine-tuned:

- **Format compliance** (no answer key required): fraction of questions answered
  in the required shape. Reported in two flavors:
  - *strict*: clean `<odpowiedz>` tag containing a valid token;
  - the *strict-vs-generous parser gap* (how much the generous fallback parser is
    doing the work) as a secondary compliance signal.
- **Conditional accuracy:** correct / parseable. Math ability with format effects
  removed. **This is the key honesty metric.**
- **Total score** (points / 30): reported for continuity with the leaderboard,
  but explicitly *not* the headline.
- **Open-task structure** (optional): did the model produce a method+result
  worked solution in CKE style.

**Decomposition (the headline).** Show that `ΔScore` is driven by `ΔFormat`, not
by `ΔConditionalAccuracy`. This is a large, robust effect that survives the small
test set — unlike a raw ±2-point score swing.

## Deliverables

New scripts continuing the existing numbering convention:

| Script | Output |
|---|---|
| `07_zbierz_archiwum.py` | extract past exams → per-question text + figures + official solutions |
| `08_buduj_dataset.py` | build `data/finetune/{train,valid}.jsonl` + provenance manifest |
| `09_trenuj_lora.py` | run LoRA SFT per model (or a documented `mlx_lm.lora` CLI invocation) |
| `10_ocen_finetune.py` | eval base-vs-fine-tuned on held-out test; compute decomposition metrics |
| `11_wykresy.py` | render the three graphs |

Other deliverables:
- **Three graphs:** (1) format compliance before/after (grouped bars, both
  models); (2) total score before/after; (3) the conditional-accuracy
  decomposition.
- New section in `results/raport.md` and a README update with the before/after
  table and the "uniform not the math" framing.
- **Optional:** publish the LoRA adapters / fused 8-bit models to Hugging Face,
  mirroring the existing Bielik-Minitron MLX weight releases.

**Out of this spec (separate follow-on):** the LinkedIn post — written later via
the `linkedin-post` skill once results are in.

## Methodological guards

- Strict chronological split; 2026 unseen; same-year *dodatkowy* excluded.
- Same judge, same prompts, temperature 0 for base and fine-tuned alike.
- Conditional accuracy reported as the robust signal; total-score deltas
  explicitly framed as noise-bound at this sample size.
- Low LoRA rank + few epochs + validation early-stop to limit catastrophic
  forgetting.
- If conditional accuracy rises materially, report the flipped finding honestly.

## Success criteria

The study succeeds if it produces a clean, defensible answer to: *what does
fine-tuning a small model on past CKE exams actually change?* — decomposed into
format vs. math, for both a weak and a strong model, on a leak-free held-out test
set. A null result on math is a success, not a failure.

## Open implementation details (for the plan)

- Exact CKE source URLs / availability per year (user is gathering the archive).
- Final LoRA hyperparameters (set via validation loss during the spike).
- Whether `09_trenuj_lora.py` wraps the CLI or is a thin documented runbook.
- Whether to publish adapters to HF (decide after results).

## First step: de-risking spike

Before the full archive prep, run the entire loop end-to-end on **PLLuM with 2–3
sheets only**: extract → build a tiny dataset → LoRA → fuse → eval on the 2026
sheet. Goal: confirm the training and eval plumbing works and that format
compliance moves in the predicted direction, before investing in the full
archive extraction.
