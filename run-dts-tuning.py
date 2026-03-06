"""
DTS Parameter Tuning Script
============================

Run GA-LLAMEA experiments with adjustable D-TS bandit parameters.
All parameters are configurable via command-line arguments.

Usage:
    # Run with improved defaults (paper-aligned)
    python run-dts-tuning.py

    # Adjust D-TS parameters
    python run-dts-tuning.py --discount 0.72 --tau_max 0.15 --epsilon_exploration 0.1

    # Full parameter sweep
    python run-dts-tuning.py --discount 0.8 --tau_max 0.2 --epsilon_exploration 0.15 \
        --budget 100 --n_parents 4 --n_offspring 8 --runs 3 --seeds 0 1 2

    # Test with 2-arm bandit (no crossover)
    python run-dts-tuning.py --arms simplify random_new

    # Use a different LLM model
    python run-dts-tuning.py --model gemini-2.0-flash-lite
"""

import argparse
import os
from datetime import datetime

import numpy as np
from dotenv import load_dotenv

from iohblade.experiment import MA_BBOB_Experiment
from iohblade.llm import Gemini_LLM
from iohblade.loggers import ExperimentLogger
from iohblade.methods import GA_LLaMEA_Method
from iohblade.problems import MA_BBOB
from iohblade.solution import Solution


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run GA-LLAMEA with adjustable D-TS parameters",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # D-TS bandit parameters
    bandit = parser.add_argument_group("D-TS Bandit Parameters")
    bandit.add_argument(
        "--discount", type=float, default=0.95,
        help="Discount factor gamma in (0,1]. Paper: gamma=1-sqrt(B_T/T). "
             "Lower = faster adaptation. 0.95 = 5%% decay per observation."
    )
    bandit.add_argument(
        "--tau_max", type=float, default=0.10,
        help="Max sampling std dev. Paper recommends mu_max/5. "
             "For graduated rewards with mu_max~0.7, use 0.10-0.15."
    )
    bandit.add_argument(
        "--epsilon_exploration", type=float, default=0.1,
        help="Epsilon-greedy exploration floor. Probability of random arm "
             "selection. 0=pure TS, 0.05=5%% random."
    )
    bandit.add_argument(
        "--arms", nargs="+", default=["simplify", "crossover", "random_new"],
        help="Operator arms to use.",
        choices=["simplify", "crossover", "random_new", "refine_weakness"],
    )

    # GA-LLAMEA parameters
    ga = parser.add_argument_group("GA-LLAMEA Parameters")
    ga.add_argument("--budget", type=int, default=100, help="Total LLM queries")
    ga.add_argument("--n_parents", type=int, default=4, help="Population size (mu)")
    ga.add_argument("--n_offspring", type=int, default=8, help="Offspring per gen (lambda)")
    ga.add_argument("--elitism", action="store_true", default=True, help="(mu+lambda) selection")
    ga.add_argument("--always_select_best", action="store_true", default=False, help="Always select highest fitness parent (disables tournament selection)")
    ga.add_argument("--use_init_prompt_for_random_new", action="store_true", default=False, help="Use the initialization prompt for random new operator instead of generating new strategies")

    # Experiment parameters
    exp = parser.add_argument_group("Experiment Parameters")
    exp.add_argument("--runs", type=int, default=3, help="Number of runs")
    exp.add_argument("--seeds", nargs="+", type=int, default=None, help="Random seeds (default: 0..runs-1)")
    exp.add_argument("--model", type=str, default="gemini-2.0-flash", help="LLM model name")
    exp.add_argument("--name", type=str, default=None, help="Method name (auto-generated if not set)")
    exp.add_argument("--output_dir", type=str, default=None, help="Output directory (auto-generated if not set)")
    exp.add_argument("--eval_timeout", type=int, default=600, help="Evaluation timeout in seconds")
    exp.add_argument("--no_ioh", action="store_true", help="Skip IOH data generation")

    return parser.parse_args()


def main():
    args = parse_args()
    load_dotenv()

    # Setup LLM
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found. Set it in .env file.")
    llm = Gemini_LLM(api_key, args.model)

    # Seeds
    seeds = args.seeds if args.seeds else list(range(args.runs))
    if len(seeds) != args.runs:
        seeds = seeds[:args.runs] if len(seeds) > args.runs else seeds + list(range(len(seeds), args.runs))

    # Method name
    name = args.name or f"GA-LLAMEA-d{args.discount}-t{args.tau_max}-e{args.epsilon_exploration}"

    # Print configuration
    print("=" * 80)
    print("DTS Parameter Tuning Experiment")
    print("=" * 80)
    print()
    print("D-TS Bandit Parameters:")
    print(f"  discount (gamma):     {args.discount}")
    print(f"  tau_max:              {args.tau_max}")
    print(f"  epsilon_exploration:  {args.epsilon_exploration}")
    print(f"  arms:                 {args.arms}")
    print()
    print("GA-LLAMEA Parameters:")
    print(f"  budget:               {args.budget}")
    print(f"  n_parents (mu):       {args.n_parents}")
    print(f"  n_offspring (lambda): {args.n_offspring}")
    print(f"  elitism:              {args.elitism}")
    print()
    print("Experiment Parameters:")
    print(f"  runs:                 {args.runs}")
    print(f"  seeds:                {seeds}")
    print(f"  model:                {args.model}")
    print(f"  method name:          {name}")
    print()

    # Create method
    method = GA_LLaMEA_Method(
        llm=llm,
        budget=args.budget,
        name=name,
        n_parents=args.n_parents,
        n_offspring=args.n_offspring,
        elitism=args.elitism,
        discount=args.discount,
        tau_max=args.tau_max,
        epsilon_exploration=args.epsilon_exploration,
        arm_names=args.arms,
        always_select_best=args.always_select_best,
        use_init_prompt_for_random_new=args.use_init_prompt_for_random_new,
    )

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or f"results/DTS-TUNING_{timestamp}"
    logger = ExperimentLogger(output_dir)

    print(f"Results will be saved to: {output_dir}")
    print()
    print("=" * 80)
    print("Starting Experiment...")
    print("=" * 80)
    print()

    training_instances = list(range(0, 20))
    test_instances = list(range(20, 70))

    # Run experiment
    experiment = MA_BBOB_Experiment(
        methods=[method],
        runs=args.runs,
        seeds=seeds,
        dims=[5],
        budget_factor=2000,
        budget=args.budget,
        eval_timeout=args.eval_timeout,
        show_stdout=True,
        exp_logger=logger,
        training_instances=training_instances,
        test_instances=test_instances,
    )
    experiment()

    print()
    print("=" * 80)
    print("Experiment Complete!")
    print("=" * 80)
    print()

    # IOH data generation
    if not args.no_ioh:
        print("=" * 80)
        print("Generating IOH data for best solutions...")
        print("=" * 80)
        print()

        ioh_dir = os.path.join(output_dir, "ioh-data")
        os.makedirs(ioh_dir, exist_ok=True)

        try:
            exp_data = logger.get_data()
            if exp_data.empty:
                print("⚠ No experiment data found, skipping IOH generation.")
            else:
                problem = MA_BBOB(
                    dims=[5],
                    budget_factor=2000,
                    training_instances=training_instances,
                    test_instances=test_instances,
                )
                problem._ensure_env()

                total_solutions = len(exp_data)
                for idx, (_, row) in enumerate(exp_data.iterrows(), 1):
                    method_name = row.get("method_name", "Unknown")
                    seed = row.get("seed", 0)
                    sol_data = row.get("solution", {})

                    print(f"[{idx}/{total_solutions}] Processing {method_name} seed={seed}...", end=" ")

                    if not sol_data or not isinstance(sol_data, dict):
                        print("⚠ No solution data, skipping")
                        continue

                    solution = Solution()
                    solution.from_dict(sol_data)

                    if not solution.code:
                        print("⚠ No code in solution, skipping")
                        continue

                    try:
                        for test_seed in range(5):
                            np.random.seed(test_seed)
                            problem.test(solution, ioh_dir=ioh_dir)
                        print(f"✓ IOH data written (5 seeds, 50 instances)")
                    except Exception as e:
                        print(f"✗ Failed: {e}")

                problem.cleanup()
                print()
                print(f"✓ IOH data saved to: {os.path.abspath(ioh_dir)}")
        except Exception as e:
            import traceback
            print()
            print("✗ Error during IOH data generation:")
            traceback.print_exc()

    print()
    print("=" * 80)
    print("All Done!")
    print("=" * 80)
    print()
    print(f"Results: {output_dir}")
    print()
    print("Configuration used:")
    print(f"  discount={args.discount}, tau_max={args.tau_max}, "
          f"epsilon={args.epsilon_exploration}, arms={args.arms}")


if __name__ == "__main__":
    main()
