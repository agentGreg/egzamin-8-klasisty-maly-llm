"""
Krótki status uploadu — dla /loop co 5 min.

Drukuje:
- Czy proces upload.py wciąż żyje (PID)
- Aktualne rozmiary na HF per repo (z porównaniem do lokalnych)
- Ostatnie 5 linii loga
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

LOG_FILE = ROOT / "publish" / "upload.log"
OUT_BASE = Path.home() / ".cache" / "huggingface" / "local-mlx"

NAMESPACE = "agentGreg"
BASE = "Bielik-Minitron-7B-v3.0-Instruct-MLX"
WARIANTY = [
    ("4bit", "Bielik-Minitron-7B-mlx-4bit"),
    ("6bit", "Bielik-Minitron-7B-mlx-6bit"),
    ("8bit", "Bielik-Minitron-7B-mlx-8bit"),
    ("bf16", "Bielik-Minitron-7B-mlx-bf16"),
]


def proces_zywy() -> tuple[bool, str]:
    r = subprocess.run(["pgrep", "-fla", "publish/upload.py"], capture_output=True, text=True)
    lines = [l for l in r.stdout.strip().split("\n") if l and "monitor.py" not in l]
    if lines:
        pids = [l.split()[0] for l in lines]
        return True, f"PID: {','.join(pids)}"
    return False, "brak procesu"


def lokalne_rozmiary() -> dict[str, float]:
    out = {}
    for q, sufix in WARIANTY:
        p = OUT_BASE / sufix
        if p.exists():
            out[q] = sum(f.stat().st_size for f in p.rglob("*")) / 1024**3
        else:
            out[q] = 0.0
    return out


def hf_rozmiary() -> dict[str, tuple[bool, float, int]]:
    """zwraca: {quant: (istnieje, gb, liczba_plikow)}"""
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    out = {}
    for q, _ in WARIANTY:
        repo_id = f"{NAMESPACE}/{BASE}-{q}"
        try:
            info = api.repo_info(repo_id=repo_id, repo_type="model", files_metadata=True)
            files = info.siblings or []
            total = sum(getattr(f, "size", None) or 0 for f in files)
            out[q] = (True, total / 1024**3, len(files))
        except Exception:
            out[q] = (False, 0.0, 0)
    return out


def main() -> None:
    alive, proc_info = proces_zywy()
    print(f"Proces upload.py: {'ŻYJE' if alive else 'NIE DZIAŁA'} ({proc_info})")

    lokal = lokalne_rozmiary()
    hf = hf_rozmiary()

    print(f"\n{'wariant':6} {'lokal':>8} {'HF':>8} {'plików':>8} {'%':>6}")
    print("-" * 42)
    done = 0
    for q, _ in WARIANTY:
        l = lokal.get(q, 0.0)
        exists, h, n = hf.get(q, (False, 0.0, 0))
        pct = (h / l * 100) if l > 0 else 0
        status = "✓" if exists and h > 0.95 * l else ("…" if exists else "—")
        if exists and h > 0.95 * l:
            done += 1
        print(f"{q:6} {l:>6.1f}GB {h:>6.1f}GB {n:>8} {pct:>5.0f}% {status}")
    print(f"\nGotowych: {done}/4")

    if LOG_FILE.exists():
        print(f"\n--- ostatnie 8 linii loga ---")
        lines = LOG_FILE.read_text(encoding="utf-8").splitlines()[-8:]
        for l in lines:
            print(l)

    # Exit code dla /loop: 0 = wciąż w robocie, 1 = wszystko done
    if done == 4 and not alive:
        print("\n✓ WSZYSTKIE 4 REPO ZAKOŃCZONE")
        sys.exit(1)


if __name__ == "__main__":
    main()
