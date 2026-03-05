from rliable import library as rly
from rliable import metrics
import matplotlib.pyplot as plt
import re
from rliable import plot_utils
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
from typing import Dict, Tuple
import os


from tex_setup import set_size
from evaluation import load_results_dataframe





FILTERED_ALGORITHMS = ["unified_edac", "unified_pbrl", "unified_msg", "unified_sacn", "rebrac"]

mujoco_datasets = ['halfcheetah-medium-expert-v2', 'hopper-medium-v2',
                   'walker2d-medium-replay-v2']
adroit_datasets = ['pen-human-v1', 'pen-cloned-v1', 'kitchen-mixed-v0']


def plot_violin_triple(df_long, result_path="figures",
                       width_fraction=1.0, height_fraction=0.25,
                       algorithms=None, ext="triple", vertical=False):
    os.makedirs(result_path, exist_ok=True)

    dataset_groups = [mujoco_datasets, adroit_datasets]

    df_plot = df_long.copy()
    if algorithms is not None:
        df_plot = df_plot[df_plot["algorithm"].isin(algorithms)]

    algos = sorted(df_plot["algorithm"].unique())

    def clean_name(s):
        if any(k in s.lower() for k in ["hopper", "walker2d", "halfcheetah"]):
            return s.split("-")[0]
        elif "kitchen" in s.lower():
            return "kitchen"
        else:
            return re.sub(r"-v\d+$", "", s)

    all_used_datasets = sorted(set(ds for group in dataset_groups for ds in group))
    all_used_clean = [clean_name(ds) for ds in all_used_datasets]

    colors = [
        "#d7191c",  # halfcheetah
        "#fdae61",  # hopper
        "#2166ac",  # walker2d
        "#7570b3",  # kitchen-mixed
        "#1a9641",  # pen-cloned
        "#2b83ba",  # pen-human
    ]
    nice_palette = {name: col for name, col in zip(all_used_clean, colors)}

    group_dataset_cleans = [
        [clean_name(ds) for ds in sorted(set(group))]
        for group in dataset_groups
    ]

    df_plot["dataset_clean"] = df_plot["dataset"].apply(clean_name)

    if vertical:
        fig, axes = plt.subplots(2, 1,
                                 figsize=set_size(width_fraction * 0.5, height_fraction * 2),
                                 sharex=False, sharey=False)
    else:
        fig, axes = plt.subplots(1, 2,
                                 figsize=set_size(width_fraction, height_fraction),
                                 sharey=True)

    for ax, ds_group in zip(axes, dataset_groups):
        df_sub = df_plot[df_plot["dataset"].isin(ds_group)].copy()

        violin_palette = {algo: (0.7, 0.7, 0.7, 0.3) for algo in algos}

        sns.violinplot(
            data=df_sub, x="algorithm", y="score",
            inner=None, scale='width', bw=0.2,
            linewidth=0.8, alpha=0.3,
            palette=violin_palette, order=algos, ax=ax
        )
        sns.stripplot(
            data=df_sub, x="algorithm", y="score",
            hue="dataset_clean", jitter=True,
            order=algos, palette=nice_palette,
            edgecolor='black', size=2, dodge=True, ax=ax
        )

        ax.set_title("")
        ax.set_xticks(range(len(algos)))
        labels = [l.replace("_", "-").replace("unified-", "").upper() for l in algos]
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=6)
        ax.set_xlabel("")
        ax.set_ylabel("D4RL Normalized Score", fontsize=8)
        ax.grid(True, axis='y', linestyle='-', alpha=0.3, linewidth=0.5)
        ax.grid(True, axis='y', which='minor', linestyle=':', alpha=0.2, linewidth=0.3)

        if ax.get_legend():
            ax.get_legend().remove()

    if not vertical:
        axes[1].set_ylabel("")

    plt.tight_layout()
    fig.canvas.draw()

    left_cleans  = group_dataset_cleans[0]
    right_cleans = group_dataset_cleans[1]

    legend_handles = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=nice_palette[name],
               markeredgecolor='black', markersize=5, label=name)
        for name in left_cleans if name in nice_palette
    ]
    right_handles = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=nice_palette[name],
               markeredgecolor='black', markersize=5, label=name)
        for name in right_cleans if name in nice_palette
    ]

    if vertical:
        for ax, handles in zip(axes, [legend_handles, right_handles]):
            ax_pos = ax.get_position()
            fig.legend(
                handles=handles,
                loc="lower left",
                bbox_to_anchor=(ax_pos.x0 , ax_pos.y0 - 0.10),
                ncol=len(handles),
                frameon=False,
                handlelength=0.6,
                handletextpad=0.3,
                columnspacing=0.8,
                fontsize=6,
                bbox_transform=fig.transFigure,
            )
    else:
        pos0 = axes[0].get_position()
        pos1 = axes[1].get_position()
        fig.legend(
            handles=legend_handles,
            loc="lower left",
            bbox_to_anchor=(pos0.x0, -0.05),
            ncol=len(legend_handles),
            frameon=False,
            handlelength=0.6,
            handletextpad=0.3,
            columnspacing=0.8,
            fontsize=6,
            bbox_transform=fig.transFigure,
        )
        fig.legend(
            handles=right_handles,
            loc="lower left",
            bbox_to_anchor=(pos1.x0, -0.05),
            ncol=len(right_handles),
            frameon=False,
            handlelength=0.6,
            handletextpad=0.3,
            columnspacing=0.8,
            fontsize=6,
            bbox_transform=fig.transFigure,
        )

    plt.savefig(f"{result_path}/violin_{ext}.pdf", dpi=300, bbox_inches='tight')
    plt.close(fig)


if __name__ == "__main__":
    df = load_results_dataframe("rollouts/results_full_eval")

    df_long = pd.melt(
        df,
        id_vars=['algorithm', 'dataset'],
        value_vars=['final_scores_mean'],
        var_name='metric',
        value_name='score'
    )
    df_long['score'] = df_long['score'].astype(float)
    df_long['dataset'] = df_long['dataset'].astype(str)
    df_long['algorithm'] = df_long['algorithm'].astype(str)

    plot_violin_triple(df_long, result_path="figures", width_fraction=1.0,
                       height_fraction=0.25, algorithms=FILTERED_ALGORITHMS,
                       ext="triple_filtered", vertical=True)
