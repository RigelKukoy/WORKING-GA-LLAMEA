"""
Generate Figure 4.9 -- Mean Operator Reward over Evaluation Budget.

Shows the mean AOCC fitness reward received from each DTS operator arm
(crossover, refine, simplify, random_new) across generations, aggregated
over all 5 GA-LLaMEA runs with ±1 std shading.

Output: Figure_4_9_Mean_Operator_Reward.png
Run from: WORKING-GA-LLAMEA/examples/
"""

import json
import os
import math
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive; safe to run without a display
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "GA-LLAMEA-8-INIT-100")
RUNS = [
    "run-GA-LLAMEA-8-INIT-100-MA_BBOB-0",
    "run-GA-LLAMEA-8-INIT-100-MA_BBOB-1",
    "run-GA-LLAMEA-8-INIT-100-MA_BBOB-2",
    "run-GA-LLAMEA-8-INIT-100-MA_BBOB-3",
    "run-GA-LLAMEA-8-INIT-100-MA_BBOB-4",
]

OUT_PATH = os.path.join(os.path.dirname(__file__), "Figure_4_9_Mean_Operator_Reward.png")

OPERATOR_LABELS = {
    "crossover":  "Crossover",
    "refine":     "Refine",
    "simplify":   "Simplify",
    "random_new": "Random New",
}

COLORS = {
    "crossover":  "#1f77b4",   # blue
    "refine":     "#d62728",   # red
    "simplify":   "#2ca02c",   # green
    "random_new": "#ff7f0e",   # orange
}

MARKERS = {
    "crossover":  "o",
    "refine":     "s",
    "simplify":   "^",
    "random_new": "D",
}

# ── Data loading ──────────────────────────────────────────────────────────────
def load_run(run_dir):
    """
    Returns dict: {generation -> {operator -> [fitness values]}}
    Skips init entries and -inf / error fitness values.
    """
    log_path = os.path.join(RESULTS_DIR, run_dir, "log.jsonl")
    data = defaultdict(lambda: defaultdict(list))

    with open(log_path, encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            op = entry["operator"]
            gen = entry["generation"]
            fitness = entry["fitness"]

            if op == "init":
                continue
            if not isinstance(fitness, float) or math.isinf(fitness) or math.isnan(fitness):
                continue

            data[gen][op].append(fitness)

    return data


def aggregate_runs(all_run_data):
    """
    Combine reward data across runs.
    Returns: {operator -> {generation -> [mean_fitness_per_run]}}
    where each value is the per-run mean for that operator/generation.
    Runs without any reward for a given op/gen contribute NaN.
    """
    # Collect all generations seen
    all_gens = sorted({g for run in all_run_data for g in run})
    all_ops  = sorted(OPERATOR_LABELS.keys())

    # per_run_means[op][gen] = list of per-run mean fitness (one entry per run)
    per_run_means = {op: {g: [] for g in all_gens} for op in all_ops}

    for run_data in all_run_data:
        for op in all_ops:
            for gen in all_gens:
                vals = run_data.get(gen, {}).get(op, [])
                if vals:
                    per_run_means[op][gen].append(float(np.mean(vals)))
                else:
                    per_run_means[op][gen].append(float("nan"))

    return all_gens, per_run_means


# ── Plot ──────────────────────────────────────────────────────────────────────
def plot(all_gens, per_run_means):
    fig, ax = plt.subplots(figsize=(10, 5.5))

    for op in sorted(OPERATOR_LABELS.keys()):
        label  = OPERATOR_LABELS[op]
        color  = COLORS[op]
        marker = MARKERS[op]

        gens_plot, means_plot, stds_plot = [], [], []

        for gen in all_gens:
            vals = [v for v in per_run_means[op][gen] if not math.isnan(v)]
            if len(vals) == 0:
                continue
            gens_plot.append(gen)
            means_plot.append(float(np.mean(vals)))
            stds_plot.append(float(np.std(vals)) if len(vals) > 1 else 0.0)

        if not gens_plot:
            continue

        means_arr = np.array(means_plot)
        stds_arr  = np.array(stds_plot)

        ax.plot(
            gens_plot, means_arr,
            label=label,
            color=color,
            marker=marker,
            linewidth=2.0,
            markersize=6,
            zorder=3,
        )
        ax.fill_between(
            gens_plot,
            means_arr - stds_arr,
            means_arr + stds_arr,
            color=color,
            alpha=0.15,
            zorder=2,
        )

    ax.set_title(
        "Mean Operator Reward over Evaluation Budget\n"
        "(mean AOCC per operator arm, averaged over 5 runs, ±1 std shading)",
        fontsize=12,
        pad=10,
    )
    ax.set_xlabel("Generation", fontsize=11)
    ax.set_ylabel("Mean AOCC Reward", fontsize=11)
    ax.set_ylim(bottom=0)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(title="Operator Arm", fontsize=10, title_fontsize=10, loc="lower right")

    # Annotate late-stage dominance
    ax.axvspan(
        max(all_gens) * 0.6, max(all_gens),
        color="lightyellow", alpha=0.4, zorder=1,
        label="_nolegend_"
    )
    ax.text(
        max(all_gens) * 0.75, ax.get_ylim()[0] + 0.02,
        "Exploitation\nphase",
        fontsize=8, color="goldenrod", ha="center", va="bottom",
    )

    fig.text(
        0.99, 0.01,
        f"GA-LLaMEA (8-parent init, 100 LLM evals)  |  MA-BBOB 5D  |  n=5 runs",
        ha="right", va="bottom", fontsize=8, color="gray",
    )

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
    print(f"Saved: {OUT_PATH}")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("Loading run data...")
    all_run_data = []
    for run_dir in RUNS:
        run_data = load_run(run_dir)
        all_run_data.append(run_data)
        gens = sorted(run_data.keys())
        total_evals = sum(len(v) for g in run_data.values() for v in g.values())
        print(f"  {run_dir}: {len(gens)} gens, {total_evals} valid operator evals")

    print("\nAggregating across runs...")
    all_gens, per_run_means = aggregate_runs(all_run_data)

    # Print summary table
    print(f"\n{'Gen':>4} | {'Crossover':>10} {'Refine':>10} {'Simplify':>10} {'RandomNew':>10}")
    print("-" * 52)
    for gen in all_gens:
        row = []
        for op in ["crossover", "refine", "simplify", "random_new"]:
            vals = [v for v in per_run_means[op][gen] if not math.isnan(v)]
            row.append(f"{np.mean(vals):.4f}" if vals else "  N/A  ")
        print(f"{gen:>4} | {row[0]:>10} {row[1]:>10} {row[2]:>10} {row[3]:>10}")

    print("\nGenerating figure...")
    plot(all_gens, per_run_means)


if __name__ == "__main__":
    main()
