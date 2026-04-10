"""
Plot arm selection percentage per generation for the most adaptive GA-LLAMEA run.
Most adaptive run is determined by the most dominant arm switches across generations,
reflecting the bandit's ability to shift strategy over time.
"""

import json
import os
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "GA-LLAMEA-8-INIT-100")
EXPERIMENT_LOG = os.path.join(RESULTS_DIR, "experimentlog.jsonl")


def load_experiment_log():
    runs = []
    with open(EXPERIMENT_LOG) as f:
        for line in f:
            d = json.loads(line)
            runs.append(d)
    return runs


def find_most_adaptive_run(runs, results_dir):
    """Select run with the most dominant arm switches across generations."""
    best_run, best_score = None, -1
    for run in runs:
        gen_ops = load_arm_data(run["log_dir"], results_dir)
        dominant = [
            Counter(gen_ops[g]).most_common(1)[0][0]
            for g in sorted(gen_ops)
        ]
        switches = sum(1 for i in range(1, len(dominant)) if dominant[i] != dominant[i - 1])
        if switches > best_score:
            best_score, best_run = switches, run
    print(f"Most adaptive run: seed={best_run['seed']} ({best_score} dominant arm switches)")
    return best_run


def load_arm_data(run_dir_name, results_dir=None):
    if results_dir is None:
        results_dir = RESULTS_DIR
    log_path = os.path.join(results_dir, run_dir_name, "log.jsonl")
    gen_ops = defaultdict(list)
    with open(log_path) as f:
        for line in f:
            d = json.loads(line)
            op = d["operator"]
            gen = d["generation"]
            if op != "init":
                gen_ops[gen].append(op)
    return gen_ops


def compute_percentages(gen_ops):
    all_arms = sorted({op for ops in gen_ops.values() for op in ops})
    generations = sorted(gen_ops.keys())
    arm_pct = {arm: [] for arm in all_arms}

    for gen in generations:
        total = len(gen_ops[gen])
        counts = Counter(gen_ops[gen])
        for arm in all_arms:
            arm_pct[arm].append(counts.get(arm, 0) / total * 100)

    return generations, arm_pct


def plot(generations, arm_pct, best_run):
    fig, ax = plt.subplots(figsize=(9, 5))

    # Match the style in the reference image
    colors = {
        "crossover": "#1f77b4",   # blue
        "random_new": "#ff7f0e",  # orange
        "simplify": "#2ca02c",    # green
        "refine": "#d62728",      # red
    }
    markers = {
        "crossover": "o",
        "random_new": "o",
        "simplify": "o",
        "refine": "s",
    }

    for arm, pcts in arm_pct.items():
        color = colors.get(arm, None)
        marker = markers.get(arm, "o")
        ax.plot(
            generations,
            pcts,
            label=arm,
            color=color,
            marker=marker,
            linewidth=1.5,
            markersize=5,
        )

    ax.set_title("GA-LLAMEA Arm Selection Percentage per Generation", fontsize=13)
    ax.set_xlabel("Generation", fontsize=11)
    ax.set_ylabel("Selection Probability (%)", fontsize=11)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.grid(True, linestyle="-", linewidth=0.5, alpha=0.7)
    ax.legend(title="Operator Arm", fontsize=9, title_fontsize=9)

    seed = best_run["seed"]
    fitness = best_run["solution"]["fitness"]
    fig.text(
        0.99, 0.01,
        f"Most adaptive run: seed {seed} (fitness={fitness:.4f})",
        ha="right", va="bottom", fontsize=8, color="gray",
    )

    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "arm_selection_adaptive_run.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved to: {out_path}")
    plt.show()


def main():
    runs = load_experiment_log()
    best_run = find_most_adaptive_run(runs, RESULTS_DIR)
    print(f"  fitness={best_run['solution']['fitness']:.4f}, dir={best_run['log_dir']}")

    gen_ops = load_arm_data(best_run["log_dir"])
    generations, arm_pct = compute_percentages(gen_ops)

    print("\nArm selection per generation:")
    for gen in generations:
        total = sum(1 for ops in [gen_ops[gen]] for _ in ops)
        counts = Counter(gen_ops[gen])
        pcts = {arm: f"{counts.get(arm, 0) / len(gen_ops[gen]) * 100:.1f}%" for arm in arm_pct}
        print(f"  gen={gen}: {pcts}")

    plot(generations, arm_pct, best_run)


if __name__ == "__main__":
    main()
