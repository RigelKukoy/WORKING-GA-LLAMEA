"""
elo_rating.py
-------------
Standalone script to compute and plot Elo (Tournament) ratings for all
algorithms found in the experiment IOH-data directories.

Usage
-----
  python elo_rating.py                      # uses default EXPERIMENT_DIRS below
  python elo_rating.py --dirs dir1 dir2     # override directories
  python elo_rating.py --save elo.png       # save plot to file (default: show interactively)
  python elo_rating.py --no-plot            # print table only; don't show/save plot
  python elo_rating.py --rounds 100000      # number of tournament rounds (default: 100000)
  python elo_rating.py --output MA_BBOB-elo.png  # output filename (default: elo_rating.png)
"""

import os
import sys
import argparse

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# --- Default configuration ---------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_EXPERIMENT_DIRS = [
    os.path.join(_SCRIPT_DIR, '..', 'results', 'GA-LLAMEA-8-INIT-100'),
    os.path.join(_SCRIPT_DIR, '..', 'results', 'CROSSOVER-ABLATION'),
    os.path.join(_SCRIPT_DIR, '..', 'results', 'EoH'),
]

IOH_SUBDIRS = ['ioh-data', 'ioh_data']
# -----------------------------------------------------------------------------


def find_ioh_dirs(experiment_dirs):
    """Return all existing IOH-data subdirectories across experiment_dirs."""
    ioh_dirs = []
    for exp_dir in experiment_dirs:
        for subdir in IOH_SUBDIRS:
            candidate = os.path.join(exp_dir, subdir)
            if os.path.isdir(candidate):
                ioh_dirs.append(os.path.abspath(candidate))
    return ioh_dirs


def compute_and_plot_elo(ioh_dirs, nrounds=100_000, save_path=None, no_plot=False):
    """Load IOH data, compute Elo ratings via iohinspector.plot.plot_tournament_ranking,
    re-render with custom styling, and optionally save.
    """
    try:
        import iohinspector
    except ImportError as exc:
        print(f"ERROR: required package not installed: {exc}")
        print("       Install with:  pip install iohinspector")
        sys.exit(1)

    print(f"Loading IOH data from {len(ioh_dirs)} directory/directories...")
    manager = iohinspector.DataManager()
    for ioh_dir in ioh_dirs:
        print(f"  + {ioh_dir}")
        manager.add_folder(ioh_dir)

    df = manager.load(monotonic=True, include_meta_data=True)
    algorithms = df["algorithm_name"].unique().to_list()
    print(f"Loaded {len(df):,} rows. Algorithms ({len(algorithms)}): {algorithms}")

    # Use iohinspector's plot function to run the tournament and get ratings back.
    # It draws into a temporary axes; we then clear and re-draw with our own style.
    print(f"\nRunning tournament ({nrounds:,} rounds)...")
    _, ax_tmp = plt.subplots(1, 1, figsize=(12, 7))
    dt_elo = iohinspector.plots.plot_tournament_ranking(df, nrounds=nrounds, ax=ax_tmp)
    plt.clf()
    plt.close("all")

    # Coerce types
    dt_elo["Rating"]    = pd.to_numeric(dt_elo["Rating"],    errors="coerce")
    dt_elo["Deviation"] = pd.to_numeric(dt_elo["Deviation"], errors="coerce").fillna(0)

    # Print table
    print("\nElo Ratings (sorted by rating, best first):")
    print(dt_elo.sort_values(by="Rating", ascending=False).to_string(index=False))

    if no_plot:
        return dt_elo

    # Re-draw with custom styling
    dt_elo_sorted = dt_elo.sort_values(by="algorithm_name").reset_index(drop=True)

    _, ax = plt.subplots(1, 1, figsize=(10, 8))
    sns.pointplot(
        data=dt_elo_sorted,
        x="algorithm_name",
        y="Rating",
        linestyle="none",
        ax=ax,
    )
    ax.errorbar(
        dt_elo_sorted["algorithm_name"],
        dt_elo_sorted["Rating"],
        yerr=dt_elo_sorted["Deviation"],
        fmt="o",
        color="blue",
        alpha=0.8,
        capsize=7,
        elinewidth=2.5,
    )
    ax.grid()
    ax.tick_params(axis="x", rotation=45)
    ax.set_xlabel("")
    ax.set_title("Tournament Ranking (all algorithms)", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nPlot saved to: {save_path}")
    else:
        plt.show()

    return dt_elo


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute and plot Elo tournament ratings from IOH experiment data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dirs", nargs="+", metavar="DIR", default=None,
        help="Experiment directories to scan for IOH data.",
    )
    parser.add_argument(
        "--rounds", type=int, default=100_000, metavar="N",
        help="Number of tournament rounds (default: 100000).",
    )
    parser.add_argument(
        "--save", nargs="?", const="elo_rating.png", metavar="FILE",
        help="Save plot to FILE instead of showing it (default filename: elo_rating.png).",
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Skip plotting; only print the ratings table.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    experiment_dirs = [os.path.abspath(d) for d in (args.dirs or DEFAULT_EXPERIMENT_DIRS)]
    ioh_dirs = find_ioh_dirs(experiment_dirs)

    if not ioh_dirs:
        print("ERROR: No IOH data directories found.")
        print("Searched inside:")
        for exp_dir in experiment_dirs:
            for sub in IOH_SUBDIRS:
                print(f"  - {os.path.join(exp_dir, sub)}")
        print("\nTip: run generate-ioh-data.py first, or pass --dirs.")
        sys.exit(1)

    save_path = None if args.save is None else os.path.abspath(args.save)
    compute_and_plot_elo(
        ioh_dirs,
        nrounds=args.rounds,
        save_path=save_path,
        no_plot=args.no_plot,
    )


if __name__ == "__main__":
    main()
