"""Sprawdza stan 4 repo na HF: czy istnieją, jakie pliki, total size."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

if not os.environ.get("HF_TOKEN"):
    sys.exit("Brak HF_TOKEN")

NAMESPACE = "agentGreg"
BASE = "Bielik-Minitron-7B-v3.0-Instruct-MLX"
WARIANTY = ["4bit", "6bit", "8bit", "bf16"]

api = HfApi(token=os.environ["HF_TOKEN"])

for q in WARIANTY:
    repo_id = f"{NAMESPACE}/{BASE}-{q}"
    print(f"\n=== {repo_id} ===")
    try:
        info = api.repo_info(repo_id=repo_id, repo_type="model", files_metadata=True)
        print(f"  istnieje: TAK")
        print(f"  created: {info.created_at}")
        files = info.siblings or []
        total = 0
        for f in files:
            size = getattr(f, "size", None) or 0
            total += size
            marker = "✓" if size > 0 else "—"
            print(f"  {marker} {f.rfilename}: {size / 1024**2:.1f} MB" if size else f"  — {f.rfilename}")
        print(f"  TOTAL: {total / 1024**3:.2f} GB")
    except Exception as e:
        print(f"  istnieje: NIE ({type(e).__name__})")
