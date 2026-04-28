"""
Read morehopqa_results_summary.csv and plot total token usage (in/out)
per model and mode, summed across cases 1–6.
For plan/code-plan modes, plan tokens are added to answer tokens.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

CSV_FILE = Path(__file__).parent / "morehopqa_results_summary.csv"
OUT_FILE = Path(__file__).parent / "morehopqa_tokens_chart.pdf"

MODES = ["code-plan", "plan", "default"]

def parse_filename(fname):
    stem = fname.replace(".jsonl", "")
    for mode in MODES:
        prefix = mode + "_"
        if stem.startswith(prefix):
            rest = stem[len(prefix):]
            suffix = "_" + mode + "_"
            idx = rest.find(suffix)
            if idx != -1:
                return mode, rest[:idx]
    parts = stem.split("_", 1)
    return parts[0], parts[1] if len(parts) > 1 else stem

# ---------------------------------------------------------------------------
# Load and compute total tokens per run
# ---------------------------------------------------------------------------
df = pd.read_csv("morehopqa_results_summary.csv")

records = []
for _, row in df.iterrows():
    mode, model = parse_filename(row["file"])
    vals_in, vals_out = [], []
    for i in range(1, 7):
        if pd.notna(row.get(f"case_{i}_tokens_in")):
            vals_in.append(row[f"case_{i}_tokens_in"])
            vals_out.append(row[f"case_{i}_tokens_out"])
        else:
            vals_in.append(row.get(f"case_{i}_plan_tokens_in",  0) + row.get(f"case_{i}_answer_tokens_in",  0))
            vals_out.append(row.get(f"case_{i}_plan_tokens_out", 0) + row.get(f"case_{i}_answer_tokens_out", 0))
    records.append({"mode": mode, "model": model,
                    "tokens_in": np.mean(vals_in), "tokens_out": np.mean(vals_out)})

results = pd.DataFrame(records)
results = results.groupby(["model", "mode"], as_index=False)[["tokens_in", "tokens_out"]].mean()

# ---------------------------------------------------------------------------
# Ordering / labels
# ---------------------------------------------------------------------------
MODEL_LABELS = {
    "gpt-4o":          "GPT-4o",
    "gpt-4.1":         "GPT-4.1",
    "gpt-4.1-mini":    "GPT-4.1 mini",
    "claude-4-sonnet": "Claude Sonnet 4",
}
MODE_ORDER  = ["default", "plan", "code-plan"]
MODE_LABELS = {"default": "Default", "plan": "Plan", "code-plan": "Code Plan"}
MODEL_ORDER = ["gpt-4o", "gpt-4.1-mini", "gpt-4.1", "claude-4-sonnet"]

seen_models = [m for m in MODEL_ORDER if m in results["model"].values]
for m in results["model"].unique():
    if m not in seen_models:
        seen_models.append(m)
seen_modes = [m for m in MODE_ORDER if m in results["mode"].values]

# nested dict: model → mode → {tokens_in, tokens_out}
data = {m: {} for m in seen_models}
for _, row in results.iterrows():
    if row["model"] in data:
        data[row["model"]][row["mode"]] = {"in": row["tokens_in"], "out": row["tokens_out"]}

# ---------------------------------------------------------------------------
# Plot — two subplots: tokens in / tokens out
# ---------------------------------------------------------------------------
COLORS = {"default": "#4C72B0", "plan": "#DD8452", "code-plan": "#55A868"}
HATCHES = {"default": "///", "plan": "...", "code-plan": "xxx"}

n_models  = len(seen_models)
n_modes   = len(seen_modes)
bar_w     = 0.22
x_centers = np.arange(n_models)

fig, axes = plt.subplots(1, 2, figsize=(n_models * 3.2, 4.5))

for ax, key, label in [
    (axes[0], "in",  "Tokens In"),
    (axes[1], "out", "Tokens Out"),
]:
    for mi, mode in enumerate(seen_modes):
        offsets = x_centers + (mi - (n_modes - 1) / 2) * bar_w
        values  = [data[m].get(mode, {}).get(key, np.nan) for m in seen_models]
        bars = ax.bar(
            offsets, values,
            width=bar_w * 0.9,
            color=COLORS[mode],
            hatch=HATCHES[mode],
            edgecolor="black",
            zorder=3,
        )
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.01,
                    f"{val/1000:.1f}k",
                    ha="center", va="bottom",
                    fontsize=8, color="black",
                )

    ax.set_title(label, fontsize=13)
    ax.set_xticks(x_centers)
    ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in seen_models],
                       rotation=20, ha="right", fontsize=11)
    ax.set_ylabel("Avg. tokens per case (cases 1–6)", fontsize=11)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=10)

legend_patches = [
    mpatches.Patch(color=COLORS[c], hatch=HATCHES[c], edgecolor="black", label=MODE_LABELS[c])
    for c in seen_modes
]
fig.legend(handles=legend_patches, loc="lower center", ncol=len(seen_modes),
           fontsize=12, bbox_to_anchor=(0.5, -0.05))

plt.tight_layout()
plt.savefig(OUT_FILE, bbox_inches="tight")
print(f"Saved → {OUT_FILE}")
plt.show()
