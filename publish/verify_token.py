"""
Sprawdza HF token załadowany z .env. Nie wypisuje samego tokena.
Drukuje tylko: user, rola (read/write), namespace info.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

if not os.environ.get("HF_TOKEN"):
    sys.exit("Brak HF_TOKEN w .env")

api = HfApi(token=os.environ["HF_TOKEN"])
info = api.whoami()

print(f"User:       {info.get('name')}")
print(f"Email:      {info.get('email')}")
auth = info.get("auth", {}).get("accessToken", {})
print(f"Token role: {auth.get('role', '?')}")
print(f"Token type: {auth.get('type', '?')}")
orgs = [o.get("name") for o in info.get("orgs", [])]
print(f"Orgs:       {orgs}")

if auth.get("role") != "write":
    sys.exit("\nERROR: token ma rolę != write. Wygeneruj nowy write token na https://huggingface.co/settings/tokens")

print("\nOK: write token gotowy, można publikować.")
