import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

COLORS = {
    "kaggle_2xT4_tp2": "#534AB7",
    "colab_A100_fp16":  "#D85A30",
    "colab_A100_int8":  "#0F6E56",
}
STYLE = {"figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
         "font.family": "sans-serif", "axes.grid": True, "grid.alpha": 0.25}

plt.rcParams.update(STYLE)

def plot_ttft_scaling(stats: pd.DataFrame):
    prompt_labels = stats["prompt_label"].unique()
    fig, axes = plt.subplots(1, len(prompt_labels), figsize=(14, 4), sharey=False)
    
    for ax, pl in zip(axes, sorted(prompt_labels)):
        grp = stats[stats["prompt_label"] == pl]
        for backend, bg in grp.groupby("backend"):
            bg = bg.sort_values("concurrency")
            ax.plot(bg["concurrency"], bg["ttft_p50"], marker="o", label=backend,
                    color=COLORS.get(backend, "gray"), linewidth=2)
            ax.fill_between(bg["concurrency"], bg["ttft_p50"], bg["ttft_p95"],
                            color=COLORS.get(backend, "gray"), alpha=0.12)
        ax.set_title(pl, fontsize=11)
        ax.set_xlabel("Concurrency")
        ax.set_ylabel("TTFT p50 (ms)")
        ax.set_ylim(0, max(stats["ttft_p95"].dropna()) * 1.15)  # force consistent y-axis from data
        ax.xaxis.set_major_locator(mticker.FixedLocator([1, 2, 4, 8, 16]))
    
    axes[0].legend(fontsize=8)
    fig.suptitle("Time to first token vs concurrency (shaded = p50→p95)", fontsize=12)
    fig.tight_layout()
    fig.savefig("results/ttft_scaling.png")
    plt.close(fig)
    print("Saved: results/ttft_scaling.png")

def plot_tpot_heatmap(stats: pd.DataFrame):
    backends = stats["backend"].unique()
    fig, axes = plt.subplots(1, len(backends), figsize=(5 * len(backends), 4))
    if len(backends) == 1:
        axes = [axes]
    for ax, backend in zip(axes, backends):
        sub = stats[stats["backend"] == backend]
        pivot = sub.pivot_table(index="prompt_label", columns="concurrency", values="tpot_mean")
        im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn_r")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=8)
        ax.set_xlabel("Concurrency")
        ax.set_title(backend, fontsize=10)
        fig.colorbar(im, ax=ax, label="TPOT (ms/tok)")
    fig.suptitle("TPOT heatmap — decode-phase bottleneck under KV cache pressure", fontsize=11)
    fig.tight_layout()
    fig.savefig("results/tpot_heatmap.png")
    plt.close(fig)
    print("Saved: results/tpot_heatmap.png")

def plot_roofline(stats: pd.DataFrame, hw: dict):
    MODEL_GB = 14.0

    fig, ax = plt.subplots(figsize=(9, 5))

    for name, params in hw.items():
        floor = (MODEL_GB / params["bw_gbs"]) * 1000
        ax.axhline(floor, linestyle="--", color=params["color"], alpha=0.6,
                   label=f"{name} BW floor ({floor:.1f} ms/tok) — best achievable")

    for backend, grp in stats[stats["prompt_label"] == "short_64"].groupby("backend"):
        grp = grp.sort_values("concurrency")
        ax.plot(grp["concurrency"], grp["tpot_mean"], marker="s", linewidth=2,
                color=COLORS.get(backend, "gray"), label=f"{backend} measured")

    ax.set_xlabel("Concurrency (parallel requests)")
    ax.set_ylabel("TPOT (ms/tok) — lower is better")
    ax.set_title("Roofline: decode TPOT vs memory-bandwidth floor\n"
                 "Measured values above floor = gap to hardware limit", fontsize=11)
    ax.legend(fontsize=8)
    ax.xaxis.set_major_locator(mticker.FixedLocator([1, 2, 4, 8, 16]))
    # no invert — floors at bottom, measured above, gap is visible
    fig.tight_layout()
    fig.savefig("results/roofline.png")
    plt.close(fig)
    print("Saved: results/roofline.png")
