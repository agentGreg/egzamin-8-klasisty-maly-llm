# publish/ — narzędzie do publikacji wag MLX dla Bielik-Minitron 7B

Skrypty do skonwertowania `speakleash/Bielik-Minitron-7B-v3.0-Instruct` do formatu MLX
(4 kwantyzacje) i upload na HF pod namespace `agentGreg/`.

Wymaga: HF token z **write scope** w `.env` jako `HF_TOKEN`, oraz zaakceptowanej
licencji bazowego modelu na HF.

## Pipeline

```bash
uv run python publish/verify_token.py   # 1. sanity check (role=write)
uv run python publish/convert.py        # 2. konwersje 4/6/8-bit + bf16 do ~/.cache/huggingface/local-mlx/
uv run python publish/smoke_test.py     # 3. każdy wariant ładuje + odpowiada na zadanie 1
uv run python publish/upload.py         # 4. tworzy repo + upload do agentGreg/...
```

## Zakres publikacji

| Repo (po uploadzie) | Rozmiar | Use case |
|---|---|---|
| `agentGreg/Bielik-Minitron-7B-v3.0-Instruct-MLX-4bit` | ~4 GB | edge / MacBook Air |
| `agentGreg/Bielik-Minitron-7B-v3.0-Instruct-MLX-6bit` | ~5.5 GB | sweet spot |
| `agentGreg/Bielik-Minitron-7B-v3.0-Instruct-MLX-8bit` | ~8 GB | high quality |
| `agentGreg/Bielik-Minitron-7B-v3.0-Instruct-MLX-bf16` | ~15 GB | full precision, źródło |

Wszystkie z `README_template.md` wypełnionym dla danej kwantyzacji, Apache 2.0,
z atrybucją SpeakLeash i linkiem do benchmarku.

## Atrybucja i zgoda

Publikacja za zgodą zespołu SpeakLeash (Paweł Kiszczak). Skontaktować się dla rebrandingu
do `speakleash/` namespace, gdy SpeakLeash będzie chciał oficjalne przejęcie wagi.
