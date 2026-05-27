"""Faza 11: wykresy fine-tune (spike) pod LinkedIn — 1080x1350 (4:5).

Dwa panele:
  01_dekompozycja.png — forma vs matematyka, bazowy vs +LoRA (z finetune_decomposition.json)
  02_overfitting.png  — krzywe straty train/val (z finetune_loss_spike.json)

Dane czytane z results/ (single source of truth). Styl spójny z temp/make_chart.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DEC = json.loads((ROOT / "results" / "finetune_decomposition.json").read_text(encoding="utf-8"))
LOSS = json.loads((ROOT / "results" / "finetune_loss_spike.json").read_text(encoding="utf-8"))
OUT_DIR = ROOT / "results" / "wykresy"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GREY = "#94a3b8"
PURPLE = "#9333ea"
PURPLE_DARK = "#581c87"
INK = "#0f172a"
SLATE = "#475569"
FAINT = "#94a3b8"
GRID = "#e2e8f0"

plt.rcParams["font.family"] = "DejaVu Sans"

FOOTER1 = "Llama-PLLuM 8B (2512) · LoRA na arkuszach CKE 2021–2023 · test: arkusz 2026 (held-out)"
FOOTER2 = "benchmark: Grzegorz Brzezinka / Prosit AS · github.com/agentGreg/egzamin-8-klasisty-maly-llm"


def footer(fig):
    fig.text(0.045, 0.038, FOOTER1, fontsize=7.3, color=FAINT, ha="left")
    fig.text(0.045, 0.020, FOOTER2, fontsize=7.3, color=FAINT, ha="left")


def base_ft(metric):
    b = next(r for r in DEC if r["model"].endswith("base"))
    f = next(r for r in DEC if "LoRA" in r["model"])
    return b[metric], f[metric]


def wykres_dekompozycja():
    fmt_b, fmt_f = base_ft("format_compliance")
    acc_b, acc_f = base_ft("conditional_accuracy")

    groups = ["Zgodność\nz formatem", "Trafność\n(na sparsowanych)"]
    base_vals = [fmt_b * 100, acc_b * 100]
    ft_vals = [fmt_f * 100, acc_f * 100]

    fig, ax = plt.subplots(figsize=(5.4, 6.75), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    x = [0, 1]
    w = 0.36
    b1 = ax.bar([xi - w / 2 for xi in x], base_vals, width=w, color=GREY, zorder=3, label="bazowy")
    b2 = ax.bar([xi + w / 2 for xi in x], ft_vals, width=w, color=PURPLE, zorder=3, label="+LoRA")

    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0", "25", "50", "75", "100%"], fontsize=9.5, color=SLATE)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=11, color=INK)
    ax.grid(axis="y", color=GRID, zorder=0)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)

    for bars in (b1, b2):
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                    f"{bar.get_height():.0f}%", ha="center", va="bottom",
                    fontsize=11, fontweight="bold", color=INK)

    ax.legend(loc="upper right", frameon=False, fontsize=10, bbox_to_anchor=(1.0, 0.92))

    ax.annotate("", xy=(-0.18, fmt_f * 100), xytext=(-0.18, fmt_b * 100),
                arrowprops=dict(arrowstyle="-|>", color=PURPLE_DARK, lw=2))
    ax.text(-0.30, (fmt_b + fmt_f) / 2 * 100, "+50\npkt proc.", ha="right", va="center",
            fontsize=9.5, fontweight="bold", color=PURPLE_DARK, linespacing=1.3)
    ax.text(1.0, acc_f * 100 + 6, "bez zmian", ha="center", va="bottom",
            fontsize=9.5, fontweight="bold", color=SLATE)

    fig.text(0.045, 0.962, "Fine-tuning uczy formy, nie matematyki",
             fontsize=13.5, fontweight="bold", color=INK, ha="left")
    fig.text(0.045, 0.935,
             "Forma odpowiedzi: 43% → 93%. Trafność: 14% → 14%. Wynik: 3 → 4 / 30 (szum).",
             fontsize=8.6, color=SLATE, ha="left")
    footer(fig)

    plt.subplots_adjust(left=0.16, right=0.965, top=0.90, bottom=0.135)
    out = OUT_DIR / "01_dekompozycja.png"
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    return out


def wykres_overfitting():
    ti, tl = LOSS["train"]["iter"], LOSS["train"]["loss"]
    vi, vl = LOSS["val"]["iter"], LOSS["val"]["loss"]
    best = LOSS["best_val_iter"]

    fig, ax = plt.subplots(figsize=(5.4, 6.75), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.plot(ti, tl, color=PURPLE, lw=2.4, marker="o", ms=4, zorder=3, label="strata treningowa")
    ax.plot(vi, vl, color="#dc2626", lw=2.4, marker="s", ms=4, zorder=3, label="strata walidacyjna")

    ax.axvline(best, color=PURPLE_DARK, ls="--", lw=1.3, zorder=2)
    ax.text(best + 6, 1.18, f"najlepsza\nwalidacja\n(~iter {best})", ha="left", va="top",
            fontsize=9, color=PURPLE_DARK, fontweight="bold", linespacing=1.3)

    ax.set_xlim(0, 310)
    ax.set_ylim(0, 1.4)
    ax.set_xlabel("iteracja", fontsize=9.5, color=SLATE)
    ax.set_ylabel("strata", fontsize=9.5, color=SLATE)
    ax.grid(color=GRID, zorder=0)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0, labelcolor=SLATE)
    ax.legend(loc="upper right", frameon=False, fontsize=10)

    ax.annotate("trening spada do 0,07\nwalidacja zawraca w górę",
                xy=(300, 0.91), xytext=(150, 0.55),
                fontsize=9.5, color=INK, fontweight="bold", linespacing=1.4,
                arrowprops=dict(arrowstyle="-|>", color=SLATE, lw=1.4),
                bbox=dict(boxstyle="round,pad=0.5", fc="#f5f3ff", ec="#ddd6fe"))

    fig.text(0.045, 0.962, "Dlaczego LoRA to nie rozwiązanie: overfitting",
             fontsize=13.5, fontweight="bold", color=INK, ha="left")
    fig.text(0.045, 0.935,
             "52 przykłady treningowe. Model zapamiętuje formę, walidacja rośnie po ~60 iteracji.",
             fontsize=8.6, color=SLATE, ha="left")
    footer(fig)

    plt.subplots_adjust(left=0.12, right=0.965, top=0.90, bottom=0.135)
    out = OUT_DIR / "02_overfitting.png"
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    return out


def main() -> None:
    for fn in (wykres_dekompozycja, wykres_overfitting):
        out = fn()
        print(f"OK: {out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
