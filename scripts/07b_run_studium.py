"""Faza 7b: base vs fine-tune, oba modele, trzy arkusze testowe (2024/2025/2026).

Pisze results/studium/<tag>_<rok>.json dla tag w {pllum_base, pllum_ft,
bielik_base, bielik_ft} i roku w {2024, 2025, 2026}.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.lib_inferencja import run_sheet  # noqa: E402

PLLUM = "/Users/greg/.cache/huggingface/local-mlx/Llama-PLLuM-8B-instruct-2512-mlx-8bit"
BIELIK = "agentGreg/Bielik-Minitron-7B-v3.0-Instruct-MLX-bf16"

SHEETS = {
    "2024": ROOT / "data/testy/2024.json",
    "2025": ROOT / "data/testy/2025.json",
    "2026": ROOT / "data/zadania.json",
}
JOBS = [
    ("pllum_base", PLLUM, None),
    ("pllum_ft", PLLUM, str(ROOT / "adapters/pllum8_full")),
    ("bielik_base", BIELIK, None),
    ("bielik_ft", BIELIK, str(ROOT / "adapters/bielik_full")),
]
OUT = ROOT / "results" / "studium"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    for name, model, adapter in JOBS:
        for sheet, path in SHEETS.items():
            print(f"\n=== {name} · {sheet} ===")
            run_sheet(model, adapter, path, OUT / f"{name}_{sheet}.json")


if __name__ == "__main__":
    main()
