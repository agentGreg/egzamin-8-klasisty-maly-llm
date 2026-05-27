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
