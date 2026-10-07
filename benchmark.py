"""LLM-bench: the Logo Color Diversity (LCD) benchmark.

Evaluates every model in data/processed/ on the official LLM-bench metric:
the number of distinct RGB colors among non-transparent pixels of the
model's preprocessed logo (256x256 RGBA, transparent background).

Outputs:
  results/scores.csv     -- per-model scores, sorted descending
  results/benchmark.png  -- the official LLM-bench leaderboard figure

Usage: python3 benchmark.py
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image

ROOT = Path(__file__).resolve().parent
PROCESSED_DIR = ROOT / "data" / "processed"
MANIFEST = ROOT / "data" / "manifest.csv"
RESULTS_DIR = ROOT / "results"
SCORES_CSV = RESULTS_DIR / "scores.csv"
FIGURE_PATH = RESULTS_DIR / "benchmark.png"

GITHUB_URL = "github.com/DeerInForestovo/LLM-bench"

# If the top tier exceeds the next tier by more than this factor, the empty
# stretch of the y-axis between them is collapsed (axis break).
BREAK_GAP_FACTOR = 1.8


def load_manifest() -> dict:
    with open(MANIFEST, newline="") as f:
        return {row["id"]: row for row in csv.DictReader(f)}


def count_colors(path: Path) -> int:
    """Official LLM-bench metric: distinct RGB tuples among pixels with
    alpha > 0. Fully transparent pixels are excluded by definition."""
    arr = np.array(Image.open(path).convert("RGBA"))
    opaque = arr[arr[:, :, 3] > 0][:, :3]
    return int(np.unique(opaque, axis=0).shape[0])


def dominant_color(path: Path) -> tuple:
    """Brand color of a logo, used to tint its leaderboard bar: the most
    frequent chromatic color (neutral white/black fills are ignored when
    a chromatic identity exists)."""
    arr = np.array(Image.open(path).convert("RGBA"))
    opaque = arr[arr[:, :, 3] > 128][:, :3]
    if len(opaque) == 0:
        return (0.5, 0.5, 0.5)
    chroma = opaque.max(axis=1).astype(int) - opaque.min(axis=1).astype(int)
    chromatic = opaque[chroma > 40]
    pool = chromatic if len(chromatic) > 0.02 * len(opaque) else opaque
    colors, counts = np.unique(pool, axis=0, return_counts=True)
    color = colors[np.argmax(counts)]
    if np.all(color >= 235):
        # A near-white bar would vanish on the beige canvas; fall back to
        # the most frequent non-white color (typically the glyph black).
        nonwhite = opaque[np.any(opaque < 235, axis=1)]
        if len(nonwhite) > 0:
            colors, counts = np.unique(nonwhite, axis=0, return_counts=True)
            color = colors[np.argmax(counts)]
    return tuple((color / 255.0).tolist())


def find_axis_break(scores: list[int]):
    """Locate the largest ratio-gap between adjacent sorted scores.
    Returns (low_max, high_min) if the gap qualifies for collapsing."""
    if len(scores) < 3:
        return None
    s = sorted(scores)
    best_gap, best_ratio = None, 1.0
    for lo, hi in zip(s[:-1], s[1:]):
        ratio = hi / max(lo, 1)
        if ratio > best_ratio and hi - lo > 0.25 * s[-1]:
            best_ratio, best_gap = ratio, (lo, hi)
    if best_gap and best_ratio >= BREAK_GAP_FACTOR:
        return best_gap
    return None


def beige_gradient(ax, xlim, ylim):
    """Paint a vertical beige gradient behind the plotting area."""
    top = np.array([0.984, 0.961, 0.902])    # light cream
    bottom = np.array([0.941, 0.886, 0.784]) # deeper beige
    grad = np.linspace(0, 1, 256)[:, None, None]
    img = top * (1 - grad) + bottom * grad
    img = np.repeat(img, 2, axis=1)
    ax.imshow(
        img,
        extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
        aspect="auto",
        zorder=0,
        interpolation="bicubic",
    )


def style_axes(ax):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(length=0)
    ax.grid(axis="y", color="white", linewidth=0.8, alpha=0.6, zorder=1)
    ax.set_axisbelow(True)


def render_leaderboard(rows: list[dict]) -> None:
    names = [r["model"] for r in rows]
    scores = [r["score"] for r in rows]
    colors = [r["bar_color"] for r in rows]
    x = np.arange(len(rows))

    gap = find_axis_break(scores)
    if gap:
        lo_max, hi_min = gap
        fig, (ax_top, ax_bot) = plt.subplots(
            2,
            1,
            sharex=True,
            figsize=(11, 6.5),
            dpi=150,
            gridspec_kw={"height_ratios": [1, 2.2], "hspace": 0.06},
        )
        ax_top.set_ylim(hi_min * 0.9, max(scores) * 1.08)
        ax_bot.set_ylim(0, lo_max * 1.25)
        axes = (ax_top, ax_bot)
    else:
        fig, ax_bot = plt.subplots(figsize=(11, 6), dpi=150)
        ax_top = None
        ax_bot.set_ylim(0, max(scores) * 1.12)
        axes = (ax_bot,)

    fig.patch.set_facecolor("#f7f0e0")
    xlim = (-0.6, len(rows) - 0.4)

    for ax in axes:
        ax.set_xlim(*xlim)
        beige_gradient(ax, xlim, ax.get_ylim())
        style_axes(ax)
        ax.bar(x, scores, width=0.62, color=colors, zorder=3,
               edgecolor="white", linewidth=0.6)

    if ax_top is not None:
        # Collapse mark: hide the touching spines and draw diagonal slashes.
        ax_top.spines["bottom"].set_visible(False)
        ax_bot.spines["top"].set_visible(False)
        ax_top.tick_params(bottom=False, labelbottom=False)
        d = 0.008
        kwargs = dict(transform=ax_top.transAxes, color="#8a7d63",
                      clip_on=False, linewidth=1.2)
        ax_top.plot((-d, +d), (-d * 2, +d * 2), **kwargs)
        ax_top.plot((1 - d, 1 + d), (-d * 2, +d * 2), **kwargs)
        kwargs.update(transform=ax_bot.transAxes)
        ax_bot.plot((-d, +d), (1 - d * 2, 1 + d * 2), **kwargs)
        ax_bot.plot((1 - d, 1 + d), (1 - d * 2, 1 + d * 2), **kwargs)

    # Value labels on top of each bar.
    label_ax = ax_top if ax_top is not None else ax_bot
    for xi, s in zip(x, scores):
        if ax_top is not None and s > ax_top.get_ylim()[0]:
            label_ax.annotate(
                f"{s:,}", (xi, s), textcoords="offset points",
                xytext=(0, 4), ha="center", fontsize=9,
                color="#4a4232", fontweight="bold", zorder=5,
            )
        else:
            ax_bot.annotate(
                f"{s:,}", (xi, s), textcoords="offset points",
                xytext=(0, 4), ha="center", fontsize=9,
                color="#4a4232", fontweight="bold", zorder=5,
            )

    # X-axis: the evaluated logos themselves, rendered at uniform size.
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(["" for _ in rows])
    for xi, row in zip(x, rows):
        logo = Image.open(row["path"]).convert("RGBA")
        box = OffsetImage(logo, zoom=52 / 256)
        ab = AnnotationBbox(
            box, (xi, 0), xybox=(0, -34), xycoords=("data", "axes fraction"),
            boxcoords="offset points", frameon=False, pad=0,
        )
        ax_bot.add_artist(ab)
        ax_bot.annotate(
            row["model"], (xi, 0), xytext=(0, -68),
            textcoords="offset points", ha="center", va="top",
            fontsize=8.5, color="#5c5340", annotation_clip=False,
        )
    ax_bot.tick_params(axis="x", pad=76)

    label_ax.set_ylabel("")
    fig.text(
        0.06, 0.5, "Distinct pixel colors (LCD score)", va="center",
        rotation="vertical", fontsize=11, color="#4a4232",
    )
    fig.suptitle(
        "LLM-bench: Logo Color Diversity Leaderboard",
        fontsize=16, fontweight="bold", color="#3d3629", y=0.97,
    )
    fig.text(
        0.5, 0.905,
        "Higher is better. Metric: distinct RGB colors in the official logo "
        "(256x256, background removed).",
        ha="center", fontsize=9.5, color="#7a6f58", style="italic",
    )
    ax_bot.annotate(
        GITHUB_URL, xy=(0.99, 0), xycoords="axes fraction",
        xytext=(0, -100), textcoords="offset points",
        ha="right", va="top", fontsize=8, color="#a29478",
        family="monospace", annotation_clip=False,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, facecolor=fig.get_facecolor(),
                bbox_inches="tight", pad_inches=0.35)
    plt.close(fig)
    print(f"[benchmark] figure written to {FIGURE_PATH}")


def main() -> None:
    manifest = load_manifest()
    rows = []
    for png in sorted(PROCESSED_DIR.glob("*.png")):
        model_id = png.stem
        info = manifest.get(model_id, {})
        score = count_colors(png)
        rows.append(
            {
                "id": model_id,
                "model": info.get("model", model_id),
                "organization": info.get("organization", ""),
                "score": score,
                "bar_color": dominant_color(png),
                "path": png,
            }
        )
        print(f"[benchmark] {info.get('model', model_id):10s} LCD={score}")

    rows.sort(key=lambda r: r["score"], reverse=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(SCORES_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "model", "organization", "lcd_score"])
        for rank, r in enumerate(rows, start=1):
            writer.writerow([rank, r["model"], r["organization"], r["score"]])
    print(f"[benchmark] scores written to {SCORES_CSV}")

    render_leaderboard(rows)


if __name__ == "__main__":
    main()
