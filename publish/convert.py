"""
Konwertuje speakleash/Bielik-Minitron-7B-v3.0-Instruct do MLX dla 4 kwantyzacji:
4-bit, 6-bit, 8-bit, bf16. Wynik w ~/.cache/huggingface/local-mlx/.

8-bit może już istnieć z poprzedniej sesji benchmarkowej (skip).
bf16 to pełna precyzja (bez kwantyzacji).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

if not os.environ.get("HF_TOKEN"):
    sys.exit("Brak HF_TOKEN w .env (potrzebny by ściągnąć gated bazowy model)")

HF_SOURCE = "speakleash/Bielik-Minitron-7B-v3.0-Instruct"
OUT_BASE = Path.home() / ".cache" / "huggingface" / "local-mlx"

# (q-bits, sufix-w-nazwie-katalogu, label-w-README)
WARIANTY = [
    (4,    "Bielik-Minitron-7B-mlx-4bit",  "4bit"),
    (6,    "Bielik-Minitron-7B-mlx-6bit",  "6bit"),
    (8,    "Bielik-Minitron-7B-mlx-8bit",  "8bit"),
    (None, "Bielik-Minitron-7B-mlx-bf16", "bf16"),  # None = bez kwantyzacji
]


def convert(q_bits: int | None, sufix: str) -> Path:
    out = OUT_BASE / sufix
    if out.exists() and any(out.glob("*.safetensors")):
        size_gb = sum(f.stat().st_size for f in out.rglob("*")) / 1024**3
        print(f"  SKIP {sufix}: istnieje ({size_gb:.1f} GB)")
        return out

    # mlx_lm convert chce żeby --mlx-path NIE istniał; usuń jeśli pusty
    if out.exists():
        shutil.rmtree(out)

    cmd = [
        "uv", "run", "python", "-m", "mlx_lm", "convert",
        "--hf-path", HF_SOURCE,
        "--mlx-path", str(out),
    ]
    if q_bits is not None:
        cmd += ["-q", "--q-bits", str(q_bits)]
    print(f"  RUN  {sufix}: {' '.join(cmd[-6:])}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  FAIL {sufix}:\n{result.stderr[-500:]}")
        shutil.rmtree(out, ignore_errors=True)
        return None
    size_gb = sum(f.stat().st_size for f in out.rglob("*")) / 1024**3
    print(f"  DONE {sufix}: {size_gb:.1f} GB")
    return out


def main() -> None:
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    print(f"Konwertuje {HF_SOURCE} do {OUT_BASE}")
    print(f"(8-bit zostanie pominięte jeśli już istnieje z poprzedniej sesji)\n")
    for q_bits, sufix, label in WARIANTY:
        convert(q_bits, sufix)

    print("\n=== Wynik ===")
    for q_bits, sufix, label in WARIANTY:
        p = OUT_BASE / sufix
        if p.exists():
            size_gb = sum(f.stat().st_size for f in p.rglob("*")) / 1024**3
            print(f"  {label:5} -> {p}  ({size_gb:.1f} GB)")
        else:
            print(f"  {label:5} -> BRAK")


if __name__ == "__main__":
    main()
