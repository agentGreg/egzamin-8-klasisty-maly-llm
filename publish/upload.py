"""
Upload 4 wariantów MLX do agentGreg/Bielik-Minitron-7B-v3.0-Instruct-MLX-{quant}.

Używa `huggingface-cli upload` przez subprocess (resume, robust pod duże transfery,
lepszy niż `HfApi.upload_large_folder` który ma buga z CLOSE_WAIT).

Loguje do publish/upload.log z timestampami. Skip wariantu jeśli HF już ma
wszystkie pliki tej samej wielkości co lokalne.
"""

from __future__ import annotations

import importlib.metadata
import logging
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi, create_repo

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

LOG_FILE = ROOT / "publish" / "upload.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("upload")

if not os.environ.get("HF_TOKEN"):
    log.error("Brak HF_TOKEN w .env")
    sys.exit(1)

OUT_BASE = Path.home() / ".cache" / "huggingface" / "local-mlx"
TEMPLATE = (ROOT / "publish" / "README_template.md").read_text(encoding="utf-8")

try:
    MLX_LM_VERSION = importlib.metadata.version("mlx-lm")
except Exception:
    MLX_LM_VERSION = "unknown"

NAMESPACE = "agentGreg"
BASE_NAME = "Bielik-Minitron-7B-v3.0-Instruct-MLX"

WARIANTY = [
    ("Bielik-Minitron-7B-mlx-4bit", "4bit", "4bit"),
    ("Bielik-Minitron-7B-mlx-6bit", "6bit", "6bit"),
    ("Bielik-Minitron-7B-mlx-8bit", "8bit", "8bit"),
    ("Bielik-Minitron-7B-mlx-bf16", "bf16", "bf16"),
]


def fill_readme(quant_label: str, size_gb: float) -> str:
    return (
        TEMPLATE
        .replace("{QUANT_LABEL}", quant_label)
        .replace("{SIZE_GB}", f"{size_gb:.1f}")
        .replace("{MLX_LM_VERSION}", MLX_LM_VERSION)
    )


def juz_uploadowane(repo_id: str, local_dir: Path) -> bool:
    """Zwraca True jeśli HF ma już komplet plików o tych samych rozmiarach co lokalne."""
    api = HfApi(token=os.environ["HF_TOKEN"])
    try:
        info = api.repo_info(repo_id=repo_id, repo_type="model", files_metadata=True)
    except Exception:
        return False
    hf_files = {f.rfilename: (getattr(f, "size", None) or 0) for f in (info.siblings or [])}
    local_files = {
        f.relative_to(local_dir).as_posix(): f.stat().st_size
        for f in local_dir.rglob("*")
        if f.is_file()
    }
    # Sprawdź czy każdy lokalny plik jest na HF z tym samym rozmiarem
    for name, lsize in local_files.items():
        if name == "README.md":
            continue  # README może się drobnie różnić (whitespace)
        if hf_files.get(name) != lsize:
            return False
    return True


def upload_jeden(local_sufix: str, quant_label: str, repo_suffix: str) -> str | None:
    local_dir = OUT_BASE / local_sufix
    if not local_dir.exists() or not any(local_dir.glob("*.safetensors")):
        log.warning(f"SKIP {quant_label}: brak {local_dir}")
        return None

    repo_id = f"{NAMESPACE}/{BASE_NAME}-{repo_suffix}"
    size_gb = sum(f.stat().st_size for f in local_dir.rglob("*")) / 1024**3
    log.info(f"=== {repo_id} ({size_gb:.1f} GB) ===")

    # README
    readme = fill_readme(quant_label, size_gb)
    (local_dir / "README.md").write_text(readme, encoding="utf-8")

    # create_repo (idempotent)
    create_repo(
        repo_id=repo_id,
        repo_type="model",
        exist_ok=True,
        private=False,
        token=os.environ["HF_TOKEN"],
    )

    # skip jeśli już uploadowane
    if juz_uploadowane(repo_id, local_dir):
        url = f"https://huggingface.co/{repo_id}"
        log.info(f"  SKIP — już na HF: {url}")
        return url

    # `hf upload <repo_id> <local_path> <remote_path>` (huggingface-cli is deprecated)
    log.info(f"  Start `hf upload`...")
    cmd = [
        "uv", "run", "hf", "upload",
        repo_id,
        str(local_dir),
        ".",  # remote path = root
        "--repo-type", "model",
        "--commit-message", f"Upload Bielik-Minitron 7B MLX {quant_label}",
        "--token", os.environ["HF_TOKEN"],
    ]
    # Stdout cli pójdzie do logfile (nasz stdout handler złapie)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        log.error(f"  FAIL {quant_label} (exit {result.returncode})")
        log.error(f"  stdout: {result.stdout[-300:]}")
        log.error(f"  stderr: {result.stderr[-300:]}")
        return None

    url = f"https://huggingface.co/{repo_id}"
    log.info(f"  ✓ {url}")
    return url


def main() -> None:
    log.info("=" * 60)
    log.info(f"START upload, namespace={NAMESPACE}, mlx-lm={MLX_LM_VERSION}")

    uploaded: list[str] = []
    for local_sufix, quant_label, repo_suffix in WARIANTY:
        try:
            url = upload_jeden(local_sufix, quant_label, repo_suffix)
            if url:
                uploaded.append(url)
        except Exception as e:
            log.error(f"FAIL {quant_label}: {type(e).__name__}: {e}")
            continue

    log.info("=" * 60)
    log.info(f"DONE — uploaded {len(uploaded)}/{len(WARIANTY)}:")
    for u in uploaded:
        log.info(f"  {u}")


if __name__ == "__main__":
    main()
