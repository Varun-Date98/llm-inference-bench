import pandas as pd
import numpy as np
import glob
from plot import plot_ttft_scaling, plot_tpot_heatmap, plot_roofline


def load_all():
    dfs = []
    for f in glob.glob("results/*.csv"):
        df = pd.read_csv(f)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def summary_stats(df):
    stats = df.groupby(["backend", "prompt_label", "concurrency"]).agg(
        ttft_p50=("ttft_ms", lambda x: x.quantile(0.50)),
        ttft_p95=("ttft_ms", lambda x: x.quantile(0.95)),
        ttft_p99=("ttft_ms", lambda x: x.quantile(0.99)),
        tpot_mean=("tpot_ms", "mean"),
        throughput_mean=("throughput_toks", "mean"),
        n=("ttft_ms", "count"),
    ).reset_index()
    return stats


def roofline_params():
    """
    Hardware roofline bounds — used to anchor where each backend
    sits relative to compute vs memory-bandwidth limits.
    T4:   65 TFLOPS fp16,  320 GB/s HBM BW  (per GPU)
    A100: 312 TFLOPS fp16, 2000 GB/s HBM BW
    Ridge point = FLOPS / BW (ops per byte)
    """
    return {
        "T4_x1":  {"tflops": 65,  "bw_gbs": 320,  "color": "#7F77DD"},
        # tensor parallel
        "T4_x2":  {"tflops": 130, "bw_gbs": 640,  "color": "#534AB7"},
        "A100":   {"tflops": 312, "bw_gbs": 2000, "color": "#D85A30"},
    }


if __name__ == "__main__":
    df = load_all()
    stats = summary_stats(df)
    print(stats.to_string())
    stats.to_csv("results/summary_stats.csv", index=False)

    plot_ttft_scaling(stats)    # TTFT vs concurrency, per backend
    plot_tpot_heatmap(stats)    # TPOT heatmap: concurrency × prompt_len
    # throughput vs arithmetic intensity
    plot_roofline(stats, roofline_params())
    print("Plots saved to results/")
