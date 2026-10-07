"""Plot the swarm-size sweep (compare.py --prefix scalability).

    python scalability_plot.py results/scalability_per_seed.csv results/scalability.png

Small multiples: one panel per metric, x = swarm size, one line per method
(mean over seeds, faint dots = individual seeds).
"""

import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

METHODS = [("cb-ele", "CB-ELE", "#2a78d6", "o"),
           ("pacnav", "PACNav", "#eb6834", "s"),
           ("couzin", "Couzin", "#1baf7a", "^")]
PANELS = [("success", "Success rate", "{:.0%}"),
          ("completion_time", "Completion time (rounds)", "{:.0f}"),
          ("order", "Order (alignment)", "{:.2f}"),
          ("fragmented_fraction", "Rounds with the swarm split", "{:.0%}"),
          ("swarm_radius", "Swarm radius (m)", "{:.0f}"),
          ("safety_breaches", "5 m safety breaches (pair-rounds)", "{:.0f}")]


def main(path, out):
    df = pd.read_csv(path)
    df["success"] = df["success"].astype(float)
    sizes = sorted(df.n_agents.unique())
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), dpi=110, constrained_layout=True)
    for ax, (col, title, fmt) in zip(axes.flat, PANELS):
        for key, label, color, marker in METHODS:
            sub = df[df.method == key]
            if sub.empty:
                continue
            means = sub.groupby("n_agents")[col].mean().reindex(sizes)
            ax.scatter(sub.n_agents + {"cb-ele": -1, "pacnav": 0, "couzin": 1}[key],
                       sub[col], s=10, color=color, alpha=0.25, linewidths=0)
            ax.plot(sizes, means, color=color, lw=2, marker=marker, ms=8,
                    label=label)
            last = means.dropna()
            if len(last):
                ax.annotate(label, (last.index[-1], last.iloc[-1]),
                            xytext=(6, 0), textcoords="offset points",
                            va="center", fontsize=8, color="#333333")
        ax.set_title(title, fontsize=11, loc="left", color="#222222")
        ax.set_xticks(sizes)
        ax.set_xlim(min(sizes) - 4, max(sizes) + 14)
        ax.grid(True, axis="y", color="#e5e5e5", lw=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors="#555555", labelsize=9)
        ax.set_xlabel("Swarm size (drones)", fontsize=9, color="#555555")
        if col in ("success", "fragmented_fraction"):
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes.flat[0].legend(frameon=False, fontsize=9, loc="lower left")
    n_seeds = df.seed.nunique()
    fig.suptitle(f"Scalability, 30% informed, {n_seeds} seeds per point "
                 "(lines = mean, dots = individual seeds)",
                 fontsize=12, color="#222222")
    fig.savefig(out, facecolor="white")
    table = df.groupby(["n_agents", "method"])[[c for c, _, _ in PANELS]].mean()
    print(table.round(3).to_string())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
