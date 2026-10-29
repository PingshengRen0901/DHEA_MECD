"""Plot only actual summary CSV data, with editable text and seed-level CIs."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from summarize import interval

NAMES = {"dhea_mecd": "DHEA-MECD", "pdqn": "PDQN", "ppo": "PPO", "sac": "SAC",
         "ddpg": "DDPG", "trpo": "TRPO", "ac": "AC", "dqn": "DQN"}
COLORS = {"dhea_mecd": "#0072B2", "pdqn": "#D55E00", "ppo": "#009E73", "sac": "#CC79A7",
          "ddpg": "#E69F00", "trpo": "#56B4E9", "ac": "#666666", "dqn": "#8C564B"}


def save(fig, output, name, smoke):
    if smoke:
        fig.suptitle("FUNCTIONAL SMOKE TEST — not paper results", fontsize=8)
    fig.savefig(output / f"{name}.png", dpi=600)
    fig.savefig(output / f"{name}.svg")
    fig.savefig(output / f"{name}.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.summary / "figures"
    output.mkdir(parents=True, exist_ok=True)
    notes = json.loads((args.summary / "statistics.json").read_text())
    smoke = notes["protocol"] == "smoke"
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
                        "font.size": 7, "axes.titlesize": 8, "legend.fontsize": 6,
                        "legend.frameon": False,
                        "svg.fonttype": "none", "pdf.fonttype": 42,
                        "axes.spines.right": False, "axes.spines.top": False})
    curves = pd.read_csv(args.summary / "learning_curves.csv")
    metrics = pd.read_csv(args.summary / "summary.csv")
    standard = curves[(curves.ablation == "none") & (curves.top_k == 2)]
    dimensions = sorted(standard.dimension.unique())
    def axes_grid(count):
        fig, axes = plt.subplots(1, count, figsize=(7.2, 2.85), squeeze=False,
                                 layout="constrained")
        return fig, axes[0]
    if dimensions:
        fig, axes = axes_grid(len(dimensions))
        for ax, dimension in zip(axes, dimensions):
            for name, frame in standard[standard.dimension == dimension].groupby("algorithm", sort=False):
                frame = frame.sort_values("episode")
                ax.plot(frame.episode, frame["mean"], label=NAMES[name], color=COLORS[name],
                        linewidth=1.5 if name == "dhea_mecd" else 0.8,
                        marker="o" if len(frame) == 1 else None, markersize=3)
                valid = frame.ci_low.notna() & frame.ci_high.notna()
                if valid.any():
                    if len(frame) == 1:
                        ax.vlines(frame.episode[valid], frame.ci_low[valid], frame.ci_high[valid], color=COLORS[name], linewidth=0.8)
                    else:
                        ax.fill_between(frame.episode[valid], frame.ci_low[valid], frame.ci_high[valid], color=COLORS[name], alpha=0.12)
            ax.set(title=f"{dimension}D", xlabel="Training episode")
        axes[0].set_ylabel("Episode return")
        fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=4)
        save(fig, output, "convergence", smoke)
        for metric, label in (("success_rate", "Tracking success (%)"), ("collision_rate", "Collision rate (%)"),
                              ("energy_cost", "Normalized command energy"), ("inference_mean_ms", "Inference latency (ms)")):
            fig, axes = axes_grid(len(dimensions))
            for ax, dimension in zip(axes, dimensions):
                frame = metrics[(metrics.dimension == dimension) & (metrics.ablation == "none") &
                                (metrics.top_k == 2) & (metrics.metric == metric)].copy()
                multiplier = 100 if metric.endswith("rate") else 1
                for i, row in enumerate(frame.itertuples()):
                    ax.scatter(i, row.mean*multiplier, color=COLORS[row.algorithm], s=18)
                    if np.isfinite(row.ci_low) and np.isfinite(row.ci_high):
                        ax.vlines(i, row.ci_low*multiplier, row.ci_high*multiplier, color=COLORS[row.algorithm], linewidth=0.8)
                ax.set(title=f"{dimension}D")
                ax.set_xticks(range(len(frame)), [NAMES[x] for x in frame.algorithm],
                              rotation=90, rotation_mode="anchor", ha="right", va="center")
            axes[0].set_ylabel(label)
            save(fig, output, metric, smoke)
    ablated = curves[(curves.algorithm == "dhea_mecd") & (curves.top_k == 2)]
    dims = [d for d in (18, 36) if ((ablated.dimension == d) & (ablated.ablation != "none")).any()]
    if dims:
        fig, axes = axes_grid(len(dims))
        labels = {"none": "Full", "no_encoder_attention": "w/o encoder-attention", "no_mecd": "w/o MECD"}
        for ax, dimension in zip(axes, dims):
            for name, frame in ablated[ablated.dimension == dimension].groupby("ablation"):
                frame = frame.sort_values("episode")
                ax.plot(frame.episode, frame["mean"], label=labels[name],
                        marker="o" if len(frame) == 1 else None, markersize=3)
                if frame.ci_low.notna().any():
                    ax.fill_between(frame.episode, frame.ci_low, frame.ci_high, alpha=0.12)
            ax.set(title=f"{dimension}D", xlabel="Training episode")
        axes[0].set_ylabel("Episode return")
        fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=3)
        save(fig, output, "ablation", smoke)
    sweep = metrics[(metrics.dimension == 36) & (metrics.algorithm == "dhea_mecd") & (metrics.ablation == "none")]
    if sweep.top_k.nunique() > 1:
        fig, axes = axes_grid(2)
        for ax, metric, label in zip(axes, ("success_rate", "collision_rate"), ("Tracking success (%)", "Collision rate (%)")):
            frame = sweep[sweep.metric == metric].sort_values("top_k")
            ax.plot(frame.top_k, frame["mean"]*100, marker="o", color=COLORS["dhea_mecd"])
            if frame.ci_low.notna().any():
                ax.fill_between(frame.top_k, frame.ci_low*100, frame.ci_high*100, alpha=0.12)
            ax.set(xlabel="Selected experts k", ylabel=label, xticks=range(1, 6))
        save(fig, output, "topk", smoke)
    savings_file = args.summary / "energy_savings_vs_ppo.csv"
    if savings_file.exists():
        savings = pd.read_csv(savings_file)
        savings = savings[(savings.ablation == "none") & (savings.top_k == 2) & (savings.algorithm != "ppo")]
        dims = sorted(savings.dimension.unique())
        if dims:
            fig, axes = axes_grid(len(dims))
            for ax, dimension in zip(axes, dims):
                names = []
                for i, (name, frame) in enumerate(savings[savings.dimension == dimension].groupby("algorithm")):
                    stats = interval(frame.saving_percent)
                    names.append(NAMES[name])
                    ax.scatter(i, stats["mean"], color=COLORS[name], s=18)
                    if stats["ci_low"] is not None:
                        ax.vlines(i, stats["ci_low"], stats["ci_high"], color=COLORS[name], linewidth=0.8)
                ax.axhline(0, color="#888888", linewidth=0.5)
                ax.set_xticks(range(len(names)), names, rotation=90, rotation_mode="anchor", ha="right", va="center")
                ax.set_title(f"{dimension}D")
            axes[0].set_ylabel("Command-energy saving vs PPO (%)")
            save(fig, output, "energy_savings_vs_ppo", smoke)
    (output / "figure_notes.txt").write_text(
        "Source: ../summary.csv, ../learning_curves.csv and ../energy_savings_vs_ppo.csv. No smoothing or imputation.\n"
        "Lines/points are seed means; bands/intervals are two-sided 95% Student t CI across independent training seeds.\n"
        "n_seeds is in the source CSV; no interval when n<2. Conditional successful paths are not zero-filled.\n"
        f"Protocol: {notes['protocol']}. {len(notes['exclusions'])} incomplete runs omitted with explicit opt-in.\n")
    print(output)


if __name__ == "__main__":
    main()
