import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os

# Load both datasets
df_awac   = pd.read_csv("awac_bias_sweep_metrics.csv")
df_uawac  = pd.read_csv("awac-unifloral-eval_metrics.csv")

df_awac["algo_group"]  = "awac"
df_uawac["algo_group"] = "u_awac"

print("uawac datasets:", df_uawac["dataset_name"].unique())
print("awac datasets:",  df_awac["dataset_name"].unique())
sub = df_uawac[df_uawac["dataset_name"].str.contains("antmaze", na=False)]
# Combine
df = pd.concat([df_awac, df_uawac], ignore_index=True)

fig_dir = "figures"
os.makedirs(fig_dir, exist_ok=True)

ylim_map = {
    ("walker",  "awac"):   (-500, 500),
    ("pen",     "awac"):   (-10000, 1000),
    ("antmaze", "awac"):   (-10, 10),
    ("walker",  "u_awac"): (-500, 500),
    ("pen",     "u_awac"): (-1000, 1000),
    ("antmaze", "u_awac"): (-0.15, 0.15),
}

def get_ylim(dataset, algo):
    for key, lim in ylim_map.items():
        if key[0] in str(dataset) and key[1] == algo:
            return lim
    return None

metrics    = ["min_q_bias", "q_pred_mean"]
algorithms = ["awac", "u_awac"]
datasets   = df["dataset_name"].dropna().unique()

for metric in metrics:
    fig, axes = plt.subplots(
        2, len(datasets),
        figsize=(5 * len(datasets), 8),
        sharex="col"
    )

    for row, algo in enumerate(["awac", "u_awac"]):
        is_uawac  = algo == "u_awac"
        param_col = "beta_id" if is_uawac else "num_critics"
        sub_algo  = df[df["algo_group"] == algo]
        param_vals = sorted(sub_algo[param_col].dropna().unique())
        cmap   = cm.get_cmap("plasma")
        colors = cmap(np.linspace(0.1, 0.9, max(len(param_vals), 1)))

        for col, dataset in enumerate(datasets):
            ax  = axes[row, col]
            sub = sub_algo[sub_algo["dataset_name"] == dataset]

            for color, pval in zip(colors, param_vals):
                group = sub[sub[param_col] == pval].groupby("step")[metric]
                mean  = group.mean()
                std   = group.std()
                ax.plot(mean.index, mean.values, color=color,
                        label=f"{param_col}={pval:.4g}")
                ax.fill_between(mean.index, mean - std, mean + std,
                                color=color, alpha=0.15)

            ax.grid(True, alpha=0.3)
            if row == 0:
                ax.set_title(dataset, fontsize=11, fontweight="bold")
            if col == 0:
                ax.set_ylabel(algo, fontsize=10, fontweight="bold")
            if row == 1:
                ax.set_xlabel("Step")
            if col == len(datasets) - 1:
                ax.legend(fontsize=7, title=param_col)


            # add line at y=0 for min_q_bias
            if metric == "min_q_bias":
                ax.axhline(0, color="black", linestyle="--", alpha=0.7)

            ylim = get_ylim(dataset, algo)
            if ylim is not None:
                ax.set_ylim(ylim)

    fig.suptitle(metric, fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/{metric}_plot.pdf", bbox_inches="tight")
    plt.savefig(f"{fig_dir}/{metric}_plot.png", dpi=150, bbox_inches="tight")
    print(f"Saved {metric} plot.")
