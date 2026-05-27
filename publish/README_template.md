---
license: apache-2.0
base_model: speakleash/Bielik-Minitron-7B-v3.0-Instruct
language:
  - pl
  - en
tags:
  - mlx
  - apple-silicon
  - bielik
  - speakleash
  - polish
  - text-generation
library_name: mlx
pipeline_tag: text-generation
---

# Bielik-Minitron-7B-v3.0-Instruct — MLX {QUANT_LABEL}

Konwersja [`speakleash/Bielik-Minitron-7B-v3.0-Instruct`](https://huggingface.co/speakleash/Bielik-Minitron-7B-v3.0-Instruct) do formatu **MLX** (Apple Silicon), kwantyzacja **{QUANT_LABEL}**.

Oryginalny model to skompresowana wersja **Bielika-11B-v3.0** (z 11.04B do 7.35B parametrów, -33%) przez structured pruning + knowledge distillation z użyciem NVIDIA Model Optimizer i NeMo Framework. Podejście inspirowane techniką Minitron.

Paper: [arxiv.org/abs/2603.11881](https://arxiv.org/abs/2603.11881)

## Warianty kwantyzacji

| Wariant | Rozmiar | Use case |
|---|---|---|
| [MLX-4bit](https://huggingface.co/agentGreg/Bielik-Minitron-7B-v3.0-Instruct-MLX-4bit) | ~4 GB | Edge / MacBook Air, ograniczona pamięć |
| [MLX-6bit](https://huggingface.co/agentGreg/Bielik-Minitron-7B-v3.0-Instruct-MLX-6bit) | ~5.5 GB | Sweet spot quality/size |
| [MLX-8bit](https://huggingface.co/agentGreg/Bielik-Minitron-7B-v3.0-Instruct-MLX-8bit) | ~8 GB | Wysoka jakość, blisko bf16 |
| [MLX-bf16](https://huggingface.co/agentGreg/Bielik-Minitron-7B-v3.0-Instruct-MLX-bf16) | ~15 GB | Pełna precyzja, źródło do dalszych konwersji |

**Aktualne repo: {QUANT_LABEL}** ({SIZE_GB} GB)

## Użycie

Wymagane: macOS na Apple Silicon, Python 3.10+.

```bash
pip install mlx-lm
```

```python
from mlx_lm import load, generate

model, tokenizer = load("agentGreg/Bielik-Minitron-7B-v3.0-Instruct-MLX-{QUANT_LABEL}")

messages = [
    {"role": "user", "content": "Wyjaśnij prosto czym różni się prędkość od przyspieszenia."},
]
prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
print(generate(model, tokenizer, prompt=prompt, max_tokens=400, verbose=True))
```

## Wydajność benchmarkowa

Bielik-Minitron 7B został przetestowany na oficjalnym arkuszu CKE z **egzaminu ósmoklasisty z matematyki 2026** (12 maja 2026, 20 zadań, 30 pkt maksimum), w porównaniu z 7 innymi konfiguracjami modeli ~4-12B parametrów.

**Wynik: 🥇 25/30 (83%)** — pierwsze miejsce w stawce, jedyny model który przekroczył 80%.

| Pozycja | Model | Wynik |
|---|---|---|
| 🥇 | **Bielik-Minitron 7B v3 (MLX 8-bit)** | **25/30 (83%)** |
| 🥈 | Bielik 4.5B v3 | 24/30 (80%) |
| 🥈 | Gemma 4 E4B (text-only) | 24/30 (80%) |
| 4 | Gemma 4 E4B (multimodal) | 23/30 (77%) |
| 5 | Gemma 3 4B (text-only) | 18/30 (60%) |
| 6 | Gemma 3 4B (multimodal) | 14/30 (47%) |
| 7 | Llama-PLLuM 8B | 3/30 (10%) |
| 7 | PLLuM 12B | 3/30 (10%) |

Pełna metodyka, kod, klucz odpowiedzi i analiza per-zadaniowa: [github.com/agentGreg/egzamin-8-klasisty-maly-llm](https://github.com/agentGreg/egzamin-8-klasisty-maly-llm)

## Atrybucja

- **Model bazowy:** [`speakleash/Bielik-Minitron-7B-v3.0-Instruct`](https://huggingface.co/speakleash/Bielik-Minitron-7B-v3.0-Instruct) — © SpeakLeash team
- **Paper:** [_Compressing Polish LLMs with Hybrid Pruning and Distillation_](https://arxiv.org/abs/2603.11881)
- **Konwersja do MLX:** [Grzegorz Brzezinka](mailto:greg@prosit.no) ([Prosit AS](https://prosit.no)), opublikowane za zgodą zespołu SpeakLeash
- **Narzędzie konwersji:** [`mlx-lm`](https://github.com/ml-explore/mlx-lm) ({MLX_LM_VERSION})

## Licencja

Apache 2.0 — zgodnie z licencją oryginalnego modelu.

## Cytowanie

Jeśli używasz tego modelu w pracy naukowej lub komercyjnej, cytuj oryginalny paper SpeakLeash:

```
@article{bielik_minitron_2026,
  title={Compressing Polish LLMs with Hybrid Pruning and Distillation},
  author={SpeakLeash team},
  journal={arXiv preprint arXiv:2603.11881},
  year={2026}
}
```

---

*Wagi MLX przygotowane w ramach benchmarku [Egzamin ósmoklasisty z matematyki 2026 — benchmark małych LLM-ów](https://github.com/agentGreg/egzamin-8-klasisty-maly-llm) by [Prosit AS](https://prosit.no).*
