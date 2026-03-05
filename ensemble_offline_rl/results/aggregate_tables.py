from evaluation import load_results_dataframe

from evaluation import load_results_dataframe
import numpy as np

df = load_results_dataframe("rollouts/results_full_eval")

df["final_scores_mean"] = df["final_scores_mean"].astype(float)
df["final_scores_std"] = df["final_scores_std"].astype(float)

print(df["algorithm"].unique())
df["algorithm"] = df["algorithm"].str.replace("augmented_methods/unified_uawac", "uawac")
print(df["algorithm"].unique())


# remove unified substring from algo
df["algorithm"] = df["algorithm"].str.replace("unified_", "")

# best policy
idx = df.groupby(["algorithm", "dataset"])["final_scores_mean"].idxmax()
best_results = df.loc[idx]


# median policy
def median_policy(group):
    median = group["final_scores_mean"].median()
    idx = (group["final_scores_mean"] - median).abs().idxmin()
    return group.loc[idx]

median_results = (
    df.groupby(["algorithm", "dataset"], group_keys=False)
    .apply(median_policy)
)

algorithms = [
    "awac","bc","edac","iql","msg","pbrl","rebrac","sacn", "urebrac", "uawac"
]

def format_cell(mean, std, best=False):
    if best:
        return f"\\textbf{{{mean:.2f}}} & \\textbf{{{std:.2f}}}"
    return f"{mean:.2f} & {std:.2f}"


def make_table(results):

    datasets = sorted(results["dataset"].unique())

    print("\\begin{tabular}{l " + " ".join(["r@{$\\,\\pm\\,$}l"]*len(algorithms)) + "}")
    header = "Dataset"
    for a in algorithms:
        header += f" & \\multicolumn{{2}}{{c}}{{{a.upper()}}}"
    print(header + " \\\\")

    print("\\midrule")

    # dataset rows
    for dataset in datasets:

        row = dataset
        dataset_df = results[results.dataset == dataset]

        best_score = dataset_df["final_scores_mean"].max()

        for algo in algorithms:

            algo_row = dataset_df[dataset_df.algorithm == algo]

            if len(algo_row) == 0:
                row += " & - & -"
                continue

            mean = algo_row["final_scores_mean"].values[0]
            std = algo_row["final_scores_std"].values[0]

            best = np.isclose(mean, best_score)

            row += " & " + format_cell(mean, std, best)

        print(row + " \\\\")

    print("\\midrule")

    # average row
    avg_means = []
    avg_stds = []

    for algo in algorithms:
        algo_df = results[results.algorithm == algo]

        avg_means.append(algo_df["final_scores_mean"].mean())
        avg_stds.append(algo_df["final_scores_std"].mean())

    best_avg = max(avg_means)

    row = "Average"
    for mean, std in zip(avg_means, avg_stds):

        best = np.isclose(mean, best_avg)
        row += " & " + format_cell(mean, std, best)

    print(row + " \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
print("Best policies:")
make_table(best_results)

print("\nMedian policies:")
make_table(median_results)
