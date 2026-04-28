"""
Read morehopqa_results_summary.csv and plot a 2×3 panel of grouped bar charts,
one panel per case (1–6), grouped by model with bars for each mode.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

CSV_FILE = Path(__file__).parent / "morehopqa_results_summary.csv"
OUT_FILE = Path(__file__).parent / "morehopqa_results_chart.pdf"

# ---------------------------------------------------------------------------
# Parse model / mode from filename
# e.g. "default_gpt-4.1_default_zeroshot_morehopqa_260427-114528.jsonl"
#      "code-plan_gpt-4.1-mini_code-plan_zeroshot_morehopqa_260428-144031.jsonl"
# ---------------------------------------------------------------------------
MODES = ["code-plan", "plan", "default"]

def parse_filename(fname: str) -> tuple[str, str]:
    """Return (mode, model) extracted from a result filename."""
    stem = fname.replace(".jsonl", "")
    for mode in MODES:
        prefix = mode + "_"
        if stem.startswith(prefix):
            rest = stem[len(prefix):]
            # next field after mode prefix is the model name, ends at "_<mode>_"
            suffix = "_" + mode + "_"
            idx = rest.find(suffix)
            if idx != -1:
                model = rest[:idx]
                return mode, model
    # fallback: split on first underscore
    parts = stem.split("_", 1)
    return parts[0], parts[1] if len(parts) > 1 else stem

# ---------------------------------------------------------------------------
# Load and process
# ---------------------------------------------------------------------------
df = pd.read_csv(CSV_FILE)

CASES = list(range(1, 7))
case_em_cols = [f"case_{i}_em" for i in CASES]

records = []
for _, row in df.iterrows():
    mode, model = parse_filename(row["file"])
    rec = {"mode": mode, "model": model}
    for i in CASES:
        rec[f"case_{i}_em"] = row[f"case_{i}_em"] * 100  # → percent
    records.append(rec)

results = pd.DataFrame(records)

# If the same (model, mode) appears more than once, average them
results = results.groupby(["model", "mode"], as_index=False)[case_em_cols].mean()

# ---------------------------------------------------------------------------
# Display labels and ordering
# ---------------------------------------------------------------------------
MODEL_LABELS = {
    "gpt-4o":          "GPT-4o",
    "gpt-4.1":         "GPT-4.1",
    "gpt-4.1-mini":    "GPT-4.1 mini",
    "claude-4-sonnet": "Claude Sonnet 4",
}

MODE_ORDER  = ["default", "plan", "code-plan"]
MODE_LABELS = {
    "default":   "Default",
    "plan":      "Plan",
    "code-plan": "Code Plan",
}

MODEL_ORDER = ["gpt-4o", "gpt-4.1-mini", "gpt-4.1", "claude-4-sonnet"]

seen_models = [m for m in MODEL_ORDER if m in results["model"].values]
for m in results["model"].unique():
    if m not in seen_models:
        seen_models.append(m)

seen_modes = [m for m in MODE_ORDER if m in results["mode"].values]

# Build nested dict: case → model → mode → em
data: dict[int, dict[str, dict[str, float]]] = {
    i: {m: {} for m in seen_models} for i in CASES
}
for _, row in results.iterrows():
    model, mode = row["model"], row["mode"]
    if model not in seen_models:
        continue
    for i in CASES:
        data[i][model][mode] = row[f"case_{i}_em"]

CASE_TITLES = {
    1: "Case 1 — MoreHopQA question",
    2: "Case 2 — Original multi-hop question",
    3: "Case 3 — Composite path question",
    4: "Case 4 — Last sub-question",
    5: "Case 5 — Middle sub-question",
    6: "Case 6 — First sub-question",
}

# ---------------------------------------------------------------------------
# Plot — 2×3 panel
# ---------------------------------------------------------------------------
COLORS = {
    "default":   "#4C72B0",
    "plan":      "#DD8452",
    "code-plan": "#55A868",
}
HATCHES = {
    "default":   "///",
    "plan":      "...",
    "code-plan": "xxx",
}

n_models  = len(seen_models)
n_modes   = len(seen_modes)
bar_w     = 0.22
x_centers = np.arange(n_models)

fig, axes = plt.subplots(2, 3, figsize=(n_models * 3.5, 8), sharey=True)

for idx, case_i in enumerate(CASES):
    ax = axes[idx // 3][idx % 3]
    case_data = data[case_i]

    for mi, mode in enumerate(seen_modes):
        offsets = x_centers + (mi - (n_modes - 1) / 2) * bar_w
        values  = [case_data[m].get(mode, np.nan) for m in seen_models]
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
                    bar.get_height() + 0.8,
                    f"{val:.0f}",
                    ha="center", va="bottom",
                    fontsize=8, color="black",
                )

    ax.set_title(CASE_TITLES[case_i], fontsize=13, pad=6)
    ax.set_xticks(x_centers)
    ax.set_xticklabels(
        [MODEL_LABELS.get(m, m) for m in seen_models],
        rotation=20, ha="right", fontsize=11,
    )
    ax.set_ylim(0, 112)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=11)
    if idx % 3 == 0:
        ax.set_ylabel("Exact Match (%)", fontsize=12)

legend_patches = [
    mpatches.Patch(color=COLORS[c], hatch=HATCHES[c], edgecolor="black",
                   label=MODE_LABELS[c])
    for c in seen_modes
]
fig.legend(handles=legend_patches, loc="lower center", ncol=len(seen_modes),
           fontsize=13, bbox_to_anchor=(0.5, -0.02))

plt.tight_layout()
plt.savefig(OUT_FILE, bbox_inches="tight")
print(f"Saved → {OUT_FILE}")
plt.show()

# ---------------------------------------------------------------------------
# Single-panel: Case 1 only
# ---------------------------------------------------------------------------
OUT_FILE_CASE1 = Path(__file__).parent / "morehopqa_results_case1.pdf"

fig1, ax1 = plt.subplots(figsize=(n_models * 1.6, 4))
case1_data = data[1]

for mi, mode in enumerate(seen_modes):
    offsets = x_centers + (mi - (n_modes - 1) / 2) * bar_w
    values  = [case1_data[m].get(mode, np.nan) for m in seen_models]
    bars = ax1.bar(
        offsets, values,
        width=bar_w * 0.9,
        color=COLORS[mode],
        hatch=HATCHES[mode],
        edgecolor="black",
        zorder=3,
    )
    for bar, val in zip(bars, values):
        if not np.isnan(val):
            ax1.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8,
                f"{val:.0f}",
                ha="center", va="bottom",
                fontsize=10, color="black",
            )

ax1.set_xticks(x_centers)
ax1.set_xticklabels(
    [MODEL_LABELS.get(m, m) for m in seen_models],
    rotation=20, ha="right", fontsize=12,
)
ax1.set_ylabel("Exact Match (%)", fontsize=13)
ax1.set_ylim(0, 112)
ax1.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
ax1.set_axisbelow(True)
ax1.spines[["top", "right"]].set_visible(False)
ax1.tick_params(axis="y", labelsize=11)

legend_patches1 = [
    mpatches.Patch(color=COLORS[c], hatch=HATCHES[c], edgecolor="black",
                   label=MODE_LABELS[c])
    for c in seen_modes
]
ax1.legend(handles=legend_patches1, loc="upper left", fontsize=12)

plt.tight_layout()
plt.savefig(OUT_FILE_CASE1, bbox_inches="tight")
print(f"Saved → {OUT_FILE_CASE1}")
plt.show()
