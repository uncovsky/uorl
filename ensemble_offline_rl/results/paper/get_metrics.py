import wandb
import pandas as pd


import wandb
import pandas as pd
import os

#set wandb entity to ahoj

os.environ["WANDB_ENTITY"] = "ahoj"

def fetch_all(project, metric_names):
    api = wandb.Api()
    runs = api.runs(project)
    records = []

    for run in runs:
        print(f"Processing run: {run.name} (ID: {run.id})")
        base = {"run_id": run.id, "run_name": run.name, **run.config}

        history = run.scan_history(keys=metric_names + ["_step"])
        for row in history:
            records.append({
                **base,
                "step": row.get("_step"),
                **{m: row.get(m) for m in metric_names}
            })

    df = pd.DataFrame(records)
    output_path = f"{project}_metrics.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} rows to {output_path}")

if __name__ == "__main__":
    #fetch_all("awac_bias_sweep", ["min_q_bias", "q_pred_mean"])
    fetch_all("awac-unifloral-eval", ["min_q_bias", "q_pred_mean"])


