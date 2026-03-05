import os
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import re
import matplotlib
matplotlib.use("pdf")
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

# ── Font size controls ────────────────────────────────────────────────────────
TITLE_SIZE        = 9
LABEL_SIZE        = 5
TICK_SIZE         = 5
LEGEND_TITLE_SIZE = 7
LEGEND_SIZE       = 6
# ─────────────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "text.usetex":      True,
    "font.family":      "serif",
    "pdf.fonttype":     42,
    "ps.fonttype":      42,
    "axes.titlesize":   TITLE_SIZE,
    "axes.labelsize":   LABEL_SIZE,
    "xtick.labelsize":  TICK_SIZE,
    "ytick.labelsize":  TICK_SIZE,
    "legend.fontsize":  LEGEND_SIZE,
    "legend.title_fontsize": LEGEND_TITLE_SIZE,
    "xtick.major.pad":  1,
    "ytick.major.pad":  1,
    "xtick.major.size": 2,
    "ytick.major.size": 2,
})

df_awac  = pd.read_csv("awac_bias_sweep_metrics.csv")
df_uawac = pd.read_csv("awac-unifloral-eval_metrics.csv")
df_awac["algo_group"]  = "awac"
df_uawac["algo_group"] = "u_awac"
df = pd.concat([df_awac, df_uawac], ignore_index=True)

fig_dir = "figures"
os.makedirs(fig_dir, exist_ok=True)

metrics_list = ["min_q_bias", "q_pred_mean"]

all_datasets = df["dataset_name"].dropna().unique()

benchmark_keywords = ["pen", "walker2d", "antmaze"]

def get_benchmark(d):
    for k in benchmark_keywords:
        if k in d:
            return k
    return None

datasets_by_benchmark = {k: [] for k in benchmark_keywords}
for d in all_datasets:
    bk = get_benchmark(d)
    if bk and "antmaze-large" not in d:
        datasets_by_benchmark[bk].append(d)

algo_configs = [
    ("awac",   "num_critics", r"shared targets -- N",   "viridis"),
    ("u_awac", "beta_id",     r"independent targets -- $\beta$", "magma"),
]

ylim_map = {
    ("walker",  "awac"):   (-500, 500),
    ("pen",     "awac"):   (-5000, 5000),
    ("antmaze", "awac"):   (-10, 10),
    ("walker",  "u_awac"): (-500, 500),
    ("pen",     "u_awac"): (-3000, 3000),
    ("antmaze", "u_awac"): (-10, 10),
}

def get_ylim(dataset, algo):
    for key, lim in ylim_map.items():
        if key[0] in str(dataset) and key[1] == algo:
            return lim
    return None

for metric in metrics_list:

    PAGE_W = 7.0
    FIG_H  = 1.9

    n_benchmarks = len(benchmark_keywords)
    n_algos      = len(algo_configs)

    # width_ratios: [1, 1, spacer, 1, 1, spacer, 1, 1]
    width_ratios = []
    for b in range(n_benchmarks):
        width_ratios += [0.1, 0.1]
        if b < n_benchmarks - 1:
            width_ratios += [0.005]
    n_total_cols = len(width_ratios)
    spacer_indices = set(
        b * n_algos + b + n_algos
        for b in range(n_benchmarks - 1)
    )
    fig, all_axes = plt.subplots(
        1, n_total_cols,
        figsize=(PAGE_W, FIG_H),
        gridspec_kw={"width_ratios": width_ratios, "wspace": 0.11},
    )
    # Hide spacer axes
    for i in spacer_indices:
        all_axes[i].set_visible(False)
    # Real axes in order, skipping spacers
    real_axes = [ax for i, ax in enumerate(all_axes) if i not in spacer_indices]

    # Share y per benchmark pair (indices into real_axes)
    for b in range(n_benchmarks):
        left  = real_axes[b * n_algos]
        right = real_axes[b * n_algos + 1]
        right.sharey(left)
        right.tick_params(labelleft=False)

    legend_handles_per_algo = {algo: [] for algo, *_ in algo_configs}

    for b, bk in enumerate(benchmark_keywords):
        bk_datasets = sorted(datasets_by_benchmark[bk])

        for a, (algo, param_col, legend_title, cmap_name) in enumerate(algo_configs):
            real_col = b * n_algos + a
            ax       = real_axes[real_col]

            is_uawac   = algo == "u_awac"
            sub_algo   = df[df["algo_group"] == algo]
            param_vals = sorted(sub_algo[param_col].dropna().unique())

            cmap   = cm.get_cmap(cmap_name)
            colors = cmap(np.linspace(0.85, 0.2, max(len(param_vals), 1)))

            ds_matches = [d for d in bk_datasets if bk in d]
            if not ds_matches:
                continue
            dataset = ds_matches[0]
            sub     = sub_algo[sub_algo["dataset_name"] == dataset]

            handles_this = []
            for color, pval in zip(colors, param_vals):
                group = sub[sub[param_col] == pval].groupby("step")[metric]
                mean  = group.mean()
                std   = group.std()
                label = rf"${pval:.4g}$" if is_uawac else rf"${int(pval)}$"
                line, = ax.plot(mean.index, mean.values, color=color,
                                label=label, linewidth=0.8)
                ax.fill_between(mean.index, mean - std, mean + std,
                                color=color, alpha=0.15, rasterized=False)
                handles_this.append(line)

            if b == 0:
                legend_handles_per_algo[algo] = handles_this

            ax.set_xlabel("Step", labelpad=2, fontsize=6)

            if real_col == 0:
                ylabel = "Q-Bias" if metric == "min_q_bias" else "Q Pred Mean"
                ax.set_ylabel(ylabel, labelpad=3)
            else:
                ax.set_ylabel("")

            ax.grid(True, alpha=0.3, linewidth=0.4)
            ax.axhline(0, color="gray", linestyle="--", linewidth=0.6, alpha=0.7)

            ylim = get_ylim(dataset, algo)
            if ylim is not None:
                ax.set_ylim(ylim)

            ax.xaxis.set_major_locator(plt.MaxNLocator(3))
            ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    # ADD INSET HERE
            if bk == "antmaze":
                axins = inset_axes(ax, width="35%", height="35%", loc="upper right")
                for color, pval in zip(colors, param_vals):
                    group = sub[sub[param_col] == pval].groupby("step")[metric]
                    mean = group.mean()
                    std = group.std()
                    axins.plot(mean.index, mean.values, color=color, linewidth=0.6)
                    axins.fill_between(mean.index, mean - std, mean + std,
                                      color=color, alpha=0.15)
                
                x_max = mean.index.max()
                x_min = x_max * 0.7  # zoom into last 30% of steps
                axins.set_xlim(x_min, x_max)

                if algo == "awac":
                    axins.set_ylim(-0.5, 0.05)  # tune this
                else:
                    axins.set_ylim(-0.05, 0.05)
                axins.tick_params(labelsize=5, labelbottom=False, bottom=False)  # hide x ticks/labels
                axins.grid(True, alpha=0.3, linewidth=0.3)
                axins.axhline(0, color="gray", linestyle="--", linewidth=0.5, alpha=0.7)
                mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="gray", linewidth=0.5)

    for algo, handles in legend_handles_per_algo.items():
        for h in handles:
            h.set_linewidth(1.5)

    fig.subplots_adjust(
        left=0.08, right=0.99,
        top=0.84,  bottom=0.38,
        wspace=0.3,
    )

    fig.canvas.draw()

    # Centered benchmark titles above each pair using real_axes positions
    for b, bk in enumerate(benchmark_keywords):
        x0 = real_axes[b * n_algos].get_position().x0
        x1 = real_axes[b * n_algos + 1].get_position().x1
        fig.text(
            (x0 + x1) / 2, 0.87, bk,
            ha='center', va='bottom',
            fontsize=TITLE_SIZE,
            transform=fig.transFigure,
        )


    # Legends: AWAC left-aligned, U-AWAC right-aligned
    total_x0 = real_axes[0].get_position().x0
    total_x1 = real_axes[-1].get_position().x1
    mid = (total_x0 + total_x1) / 2

    legend_positions = {
        "awac":   mid - 0.20,
        "u_awac": mid + 0.15,
    }
    legend_locs = {
        "awac":   "lower center",
        "u_awac": "lower center",
    }

    for algo, (_, param_col, legend_title, _) in zip(
            [c[0] for c in algo_configs], algo_configs):
        handles    = legend_handles_per_algo[algo]
        param_vals = sorted(df[df["algo_group"] == algo][param_col].dropna().unique())

        fig.legend(
            handles=handles,
            title=legend_title,
            loc=legend_locs[algo],
            bbox_to_anchor=(legend_positions[algo], 0.05),
            bbox_transform=fig.transFigure,
            ncol=len(param_vals),
            frameon=True,
            framealpha=0.9,
            edgecolor='lightgray',
            borderpad=0.5,
            handlelength=1.4,
            handletextpad=0.4,
            columnspacing=1.0,
        )


    plt.savefig(f"{fig_dir}/{metric}_combined_plot.pdf", bbox_inches="tight", backend="pdf")
    plt.savefig(f"{fig_dir}/{metric}_combined_plot.png", dpi=300, bbox_inches="tight")
    print(f"Saved {metric} combined plot.")
