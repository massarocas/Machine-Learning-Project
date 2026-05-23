"""
Generate presentation figures for professors:
  1. focal_loss_mechanism.png  — how focal loss down-weights easy examples
  2. focal_loss_presentation.png — full story (imbalance + mechanism + results)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker

from dust_particles_focal import SCENARIOS, make_data

np.random.seed(42)

PURPLE = "#7F77DD"
GRAY = "#9A9893"
RED = "#E24B4A"
GREEN = "#1D9E75"
AMBER = "#EF9F27"
BG = "#FAFAF8"
BG2 = "#F2F1EF"
TEXT = "#1A1A18"
LIGHT = "#888780"
OUT_DIR = Path(__file__).resolve().parent


def cross_entropy_scalar(p):
    return -np.log(p)


def focal_scalar(p, gamma):
    return -((1 - p) ** gamma) * np.log(p)


def plot_mechanism(path: Path) -> None:
    """Panel A: FL vs CE as function of confidence p (correct class)."""
    p = np.linspace(0.01, 0.99, 500)
    ce = cross_entropy_scalar(p)

    fig, (ax_main, ax_ratio) = plt.subplots(
        2,
        1,
        figsize=(9, 7),
        facecolor=BG,
        gridspec_kw={"height_ratios": [3, 1.4], "hspace": 0.08},
    )
    for ax in (ax_main, ax_ratio):
        ax.set_facecolor(BG2)
        ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
        ax.tick_params(colors=LIGHT, labelsize=9)
        ax.grid(axis="y", color="white", linewidth=1.4, zorder=2)

    gammas = [
        (0.0, GRAY, "γ = 0  (= cross-entropy)", "--"),
        (1.0, GREEN, "γ = 1", "-"),
        (2.0, PURPLE, "γ = 2  (default)", "-"),
        (5.0, RED, "γ = 5", "-"),
    ]
    for gamma, color, label, ls in gammas:
        fl = focal_scalar(p, gamma)
        lw = 2.4 if gamma == 2.0 else 1.6
        ax_main.plot(p, fl, color=color, lw=lw, ls=ls, label=label, zorder=3)

    ax_main.axvspan(0.01, 0.30, color=RED, alpha=0.06, zorder=1)
    ax_main.axvspan(0.70, 0.99, color=GREEN, alpha=0.06, zorder=1)
    ax_main.text(0.15, 3.5, "Hard\n(wrong / unsure)", ha="center", fontsize=8, color=RED, style="italic")
    ax_main.text(0.85, 3.5, "Easy\n(confident & correct)", ha="center", fontsize=8, color=GREEN, style="italic")
    ax_main.set_xlim(0, 1)
    ax_main.set_ylim(0, 3.8)
    ax_main.set_ylabel("Loss (per example)", fontsize=10, color=TEXT)
    ax_main.set_xticklabels([])
    ax_main.set_title(
        "How focal loss works: down-weight easy examples",
        fontsize=13,
        fontweight="600",
        color=TEXT,
        pad=12,
    )
    ax_main.legend(loc="upper right", fontsize=9, frameon=True, facecolor=BG, edgecolor=LIGHT)

    for gamma, color, _label, ls in gammas[1:]:
        ratio = focal_scalar(p, gamma) / ce
        lw = 2.4 if gamma == 2.0 else 1.6
        ax_ratio.plot(p, ratio, color=color, lw=lw, ls=ls, zorder=3)
    ax_ratio.axhline(1.0, color=GRAY, lw=1.2, ls="--")
    ax_ratio.set_xlim(0, 1)
    ax_ratio.set_ylim(-0.05, 1.05)
    ax_ratio.set_xlabel("p = predicted probability for the true class", fontsize=10, color=TEXT)
    ax_ratio.set_ylabel("Focal / CE", fontsize=10, color=TEXT)
    ax_ratio.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))
    ax_ratio.axvspan(0.70, 0.99, color=GREEN, alpha=0.06, zorder=1)

    p_mark = 0.9
    r_mark = focal_scalar(p_mark, 2.0) / cross_entropy_scalar(p_mark)
    ax_ratio.annotate(
        f"γ=2, p=0.9 → only {r_mark:.0%} of CE loss",
        xy=(p_mark, r_mark),
        xytext=(0.55, 0.35),
        arrowprops=dict(arrowstyle="->", color=PURPLE, lw=1.3),
        fontsize=9,
        color=PURPLE,
    )

    fig.text(
        0.5,
        0.01,
        "FL = −(1−p)^γ · log(p)   ·   When p→1 (easy), (1−p)^γ → 0 so gradient focuses on hard/minority-like mistakes",
        ha="center",
        fontsize=8,
        color=LIGHT,
        style="italic",
    )
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close()


def plot_setup(path: Path) -> None:
    """Setup slide only: context, imbalance, mechanism, feature space (no BCE vs focal scores)."""
    extreme = SCENARIOS[2]
    X, y = make_data(extreme["n_dust"], extreme["n_core"])

    fig = plt.figure(figsize=(14, 8), facecolor=BG)
    fig.suptitle(
        "Focal loss for imbalanced classification — synthetic cleanroom particles",
        fontsize=15,
        fontweight="600",
        color=TEXT,
        y=0.98,
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.32, height_ratios=[1.1, 0.9])

    # ── (0,0) What are dust vs core grit? ──
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.axis("off")
    ax0.text(
        0.5,
        0.92,
        "Particle classes (synthetic)",
        ha="center",
        fontsize=12,
        fontweight="600",
        color=TEXT,
        transform=ax0.transAxes,
    )
    story = (
        "Optical particle counter in a cleanroom:\n\n"
        "• Dust (label 0, majority)\n"
        "  Fine mineral or textile debris.\n"
        "  Large, low fluorescence, common.\n\n"
        "• Core grit (label 1, minority)\n"
        "  Rare abrasive SiC grit or solder-\n"
        "  flux residue from tooling.\n"
        "  Smaller, brighter scatter — yield risk.\n\n"
        "“Core grit” names the rare class we\n"
        "must not miss when data are imbalanced."
    )
    ax0.text(
        0.08,
        0.82,
        story,
        va="top",
        fontsize=9.5,
        color=TEXT,
        linespacing=1.45,
        transform=ax0.transAxes,
        family="sans-serif",
    )

    # ── (0,1) Class imbalance ──
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.set_facecolor(BG2)
    counts = [extreme["n_dust"], extreme["n_core"]]
    bars = ax1.bar(
        ["Dust\n(majority)", "Core grit\n(minority)"],
        counts,
        color=[GRAY, PURPLE],
        width=0.55,
        zorder=3,
    )
    for b, c in zip(bars, counts):
        ax1.text(
            b.get_x() + b.get_width() / 2,
            c + 15,
            str(c),
            ha="center",
            fontsize=10,
            fontweight="600",
            color=TEXT,
        )
    ax1.set_title("Extreme imbalance (96% / 4%)", fontsize=11, fontweight="600", color=TEXT)
    ax1.set_ylabel("Training particles", fontsize=9, color=TEXT)
    ax1.set_ylim(0, 1100)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.grid(axis="y", color="white", linewidth=1.2, zorder=2)
    ax1.text(
        0.5,
        -0.22,
        "Rare class is hard to learn with standard BCE under strong imbalance",
        ha="center",
        fontsize=8,
        color=LIGHT,
        transform=ax1.transAxes,
        style="italic",
    )

    # ── (0,2) Mechanism mini-chart ──
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor(BG2)
    p = np.linspace(0.05, 0.99, 200)
    ax2.plot(p, cross_entropy_scalar(p), color=GRAY, lw=2, ls="--", label="Cross-entropy")
    ax2.plot(p, focal_scalar(p, 2.0), color=PURPLE, lw=2.4, label="Focal (γ=2)")
    ax2.fill_between(p, 0, focal_scalar(p, 2.0), where=(p > 0.75), alpha=0.12, color=GREEN)
    ax2.text(0.88, 0.35, "easy\n↓ low loss", ha="center", fontsize=8, color=GREEN)
    ax2.text(0.2, 2.2, "hard\n↑ kept", ha="center", fontsize=8, color=RED)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 3.2)
    ax2.set_xlabel("p(correct class)", fontsize=9)
    ax2.set_ylabel("Loss", fontsize=9)
    ax2.set_title("Focal loss focuses on hard examples", fontsize=11, fontweight="600", color=TEXT)
    ax2.legend(fontsize=8, frameon=False)
    ax2.spines[["top", "right"]].set_visible(False)

    # ── bottom row: feature space (full width) ──
    ax3 = fig.add_subplot(gs[1, :])
    ax3.set_facecolor(BG2)
    ax3.scatter(X[y == 0, 0], X[y == 0, 1], s=8, alpha=0.25, c=GRAY, label="Dust")
    ax3.scatter(
        X[y == 1, 0],
        X[y == 1, 1],
        s=40,
        alpha=0.9,
        c=PURPLE,
        edgecolors="white",
        linewidths=0.3,
        label="Core grit",
    )
    ax3.set_xlabel("Sensor feature 1", fontsize=9)
    ax3.set_ylabel("Sensor feature 2", fontsize=9)
    ax3.set_title("Synthetic feature space (overlapping)", fontsize=11, fontweight="600", color=TEXT)
    ax3.legend(fontsize=8, frameon=False, loc="upper left")
    ax3.spines[["top", "right"]].set_visible(False)

    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close()


def main():
    p1 = OUT_DIR / "focal_loss_mechanism.png"
    p2 = OUT_DIR / "focal_loss_setup.png"
    plot_mechanism(p1)
    plot_setup(p2)
    print(f"Saved → {p1}")
    print(f"Saved → {p2}")
    print("(No BCE vs focal metric outputs. Use run_evaluation() in dust_particles_focal when ready.)")


if __name__ == "__main__":
    main()
