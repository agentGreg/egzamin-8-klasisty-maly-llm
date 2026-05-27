"""Faza 14: wykres studium — oba modele, base vs +LoRA, pod LinkedIn (1080x1350).

Dwa panele (zgodność z formatem, trafność) × dwie grupy (PLLuM, Bielik) × dwa słupki
(bazowy, +LoRA). Dane z results/studium_decomposition.json. Styl jak 11_wykresy.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DEC = json.loads((ROOT / "results" / "studium_decomposition.json").read_text(encoding="utf-8"))
OUT_DIR = ROOT / "results" / "wykresy"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GREY = "#94a3b8"
PURPLE = "#9333ea"
INK = "#0f172a"
SLATE = "#475569"
GRID = "#e2e8f0"

plt.rcParams["font.family"] = "DejaVu Sans"

MODELS = ["Llama-PLLuM 8B", "Bielik-Minitron 7B"]
FOOTER1 = "Test: arkusze CKE 2024+2025+2026 (held-out) · LoRA z early stopping · zadania zamknięte"
FOOTER2 = "benchmark: Grzegorz Brzezinka / Prosit AS · github.com/agentGreg/egzamin-8-klasisty-maly-llm"


def val(model: str, variant: str, metric: str) -> float:
    r = next(r for r in DEC if r["model"] == model and r["variant"] == variant)
    return r[metric] * 100


def panel(ax, metric: str, title: str):
    x = [0, 1]
    w = 0.36
    base = [val(m, "base", metric) for m in MODELS]
    ft = [val(m, "+LoRA", metric) for m in MODELS]
    b1 = ax.bar([xi - w / 2 for xi in x], base, width=w, color=GREY, zorder=3, label="bazowy")
    b2 = ax.bar([xi + w / 2 for xi in x], ft, width=w, color=PURPLE, zorder=3, label="+LoRA")
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 50, 100])
    ax.set_yticklabels(["0", "50", "100%"], fontsize=9, color=SLATE)
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS, fontsize=10.5, color=INK)
    ax.set_title(title, fontsize=11.5, fontweight="bold", color=INK, loc="left", pad=8)
    ax.grid(axis="y", color=GRID, zorder=0)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                    f"{bar.get_height():.0f}%", ha="center", va="bottom",
                    fontsize=10, fontweight="bold", color=INK)
    return b1, b2


def main() -> None:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.4, 6.75), dpi=200)
    fig.patch.set_facecolor("white")
    for ax in (ax1, ax2):
        ax.set_facecolor("white")

    panel(ax1, "format_compliance", "Zgodność z formatem")
    b1, _ = panel(ax2, "conditional_accuracy", "Trafność (na sparsowanych)")
    ax1.legend(loc="upper right", frameon=False, fontsize=9.5)

    fig.text(0.045, 0.965, "Fine-tuning uczy formy, nie matematyki",
             fontsize=13.5, fontweight="bold", color=INK, ha="left")
    fig.text(0.045, 0.940,
             "Słaby model (PLLuM) i mocny (Bielik) — LoRA rusza formę, nie trafność.",
             fontsize=8.6, color=SLATE, ha="left")
    fig.text(0.045, 0.038, FOOTER1, fontsize=7.0, color=GREY, ha="left")
    fig.text(0.045, 0.020, FOOTER2, fontsize=7.0, color=GREY, ha="left")

    plt.subplots_adjust(left=0.10, right=0.965, top=0.88, bottom=0.115, hspace=0.42)
    out = OUT_DIR / "03_studium.png"
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"OK: {out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
