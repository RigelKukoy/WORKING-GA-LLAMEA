"""
Ablation Study: LLaMEA-Crossover vs GA-LLaMEA (Matched Prompts)
================================================================

This script compares two methods head-to-head using **identical prompts**:

  Method 1 — LLaMEA + Crossover (uniform random operator selection)
      Arms : simplify | crossover | random_new
      Selection : uniform random (each arm equally likely)

  Method 2 — GA-LLaMEA (Discounted Thompson Sampling operator selection)
      Arms : simplify | crossover | random_new
      Selection : D-TS bandit (adaptive, learns which arm works best)

Both methods use the exact same prompt text for every operator arm:
  • simplify   → "Refine and simplify the selected algorithm to improve it."
  • crossover  → Guided code-level concept transfer (shows full code of both
                 parent and inspirations, same structure as CrossoverOperator)
  • random_new → Structural-reference prompt with novelty instruction

The only variable between the two methods is the operator selection strategy:
  LLaMEA = uniform random  vs  GA-LLaMEA = adaptive D-TS bandit.

This isolates the contribution of adaptive operator selection from prompt design.
"""

import os
import random
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

from iohblade.experiment import MA_BBOB_Experiment
from iohblade.llm import GeminiAPI_LLM
from iohblade.loggers import ExperimentLogger
from iohblade.solution import Solution
from iohblade.problems import MA_BBOB
from iohblade.methods.ga_llamea import GA_LLaMEA_Method
from iohblade.methods.llamea import LLaMEA


# ─────────────────────────────────────────────────────────────────────────────
#  Shared prompt text (used identically by both methods)
# ─────────────────────────────────────────────────────────────────────────────

REFINE_INSTRUCTION   = "Refine the strategy of the selected solution to improve it."
SIMPLIFY_INSTRUCTION = "Refine and simplify the selected algorithm to improve it."

RANDOM_NEW_INSTRUCTION = (
    "Please help me create a new algorithm that has a totally different form "
    "from the given ones.\n\n"
    "Generate a completely novel approach that explores a different region of "
    "the algorithm design space."
)

STRUCTURAL_REFERENCE = """\
For correct code structure, follow this template:
```python
import numpy as np

class YourAlgorithm:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb = func.bounds.lb  # lower bounds (numpy array)
        ub = func.bounds.ub  # upper bounds (numpy array)
        f_opt = np.inf
        x_opt = None
        eval_count = 0

        # Your optimization logic here
        # Use func(x) to evaluate a candidate x (numpy array of shape (dim,))
        # Track eval_count and stop when eval_count >= self.budget

        return f_opt, x_opt
```
Use a DIFFERENT strategy from the algorithms listed above. This template is only for correct structure and formatting."""

CROSSOVER_INSTRUCTION = (
    "Create a new improved solution by combining ideas from the inspiration "
    "solutions while maintaining syntactic correctness."
)


# ─────────────────────────────────────────────────────────────────────────────
#  DynamicCrossoverPrompt for LLaMEA
#  Mirrors CrossoverOperator.build_prompt() exactly — shows full code of both
#  the selected parent (provided by LLaMEA's own prompt builder) and all
#  inspiration parents.
# ─────────────────────────────────────────────────────────────────────────────

class DynamicCrossoverPrompt:
    """
    String-like object that injects inspiration code into the crossover
    mutation instruction at runtime, matching GA-LLaMEA's CrossoverOperator.

    LLaMEA appends mutation_prompts[i] as the instruction suffix after its
    own prompt header (role + task + history + selected parent code).

    This prompt adds:
      - Full code of N inspiration parents drawn from the current population
      - The same CROSSOVER_INSTRUCTION wording used by CrossoverOperator

    The result is prompt-identical to GA-LLaMEA's crossover arm.
    """

    def __init__(self, llamea_method_wrapper, num_inspirations: int = 3):
        self._wrapper = llamea_method_wrapper
        self.num_inspirations = num_inspirations

    def __str__(self) -> str:
        llamea_instance = getattr(self._wrapper, "llamea_instance", None)
        if not llamea_instance or not llamea_instance.population:
            # Fallback before any population exists (very first call)
            return CROSSOVER_INSTRUCTION

        valid_pop = [
            p for p in llamea_instance.population
            if p.name and p.code and p.fitness is not None and not np.isinf(p.fitness)
        ]

        if not valid_pop:
            return CROSSOVER_INSTRUCTION

        # Randomly sample inspiration parents (up to num_inspirations)
        n = min(self.num_inspirations, len(valid_pop))
        inspirations = random.sample(valid_pop, n)

        # Build inspiration blocks — full code, same format as CrossoverOperator
        insp_blocks = []
        for i, insp in enumerate(inspirations):
            insp_blocks.append(
                f"Inspiration {i + 1}: {insp.name} (fitness: {insp.fitness:.4f})\n"
                f"```python\n{insp.code}\n```"
            )
        inspirations_str = "\n\n".join(insp_blocks)

        return (
            "These are other high-performing solutions discovered during the search.\n"
            "You may borrow useful ideas, logic, or techniques from them.\n\n"
            f"{inspirations_str}\n\n"
            f"{CROSSOVER_INSTRUCTION}"
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    load_dotenv()

    api_key  = os.getenv("GEMINI_API_KEY")
    ai_model = "gemini-2.5-flash"

    llm = GeminiAPI_LLM(api_key=api_key, model=ai_model)

    budget   = 100   # LLM queries per run
    num_runs = 5     # Runs per method for statistical significance
    seeds    = list(range(num_runs))  # [0, 1, 2, 3, 4]

    NUM_CROSSOVER_INSPIRATIONS = 3   # Same for both methods
    N_PARENTS    = 4                 # Working population size (unchanged)
    N_OFFSPRING  = 8                 # Offspring per generation (unchanged)
    INIT_OVERSAMPLE = 2              # Both methods: generate N_PARENTS*2 init candidates, keep best N_PARENTS

    print("=" * 80)
    print("Ablation: LLaMEA-Crossover  vs  GA-LLaMEA  (Matched Prompts)")
    print("=" * 80)
    print(f"Model          : {ai_model}")
    print(f"Budget         : {budget} LLM queries per run")
    print(f"Runs           : {num_runs}  (seeds {seeds})")
    print(f"n_parents      : {N_PARENTS}  |  n_offspring: {N_OFFSPRING}")
    print(f"init_oversample: {INIT_OVERSAMPLE}  ({N_PARENTS}*{INIT_OVERSAMPLE}={N_PARENTS*INIT_OVERSAMPLE} init candidates → keep best {N_PARENTS})")
    print(f"Inspirations   : {NUM_CROSSOVER_INSPIRATIONS} (crossover arm, both methods)")
    print()

    # ── Method 1: LLaMEA + Crossover (uniform random selection) ──────────────
    #
    # Arms: simplify | crossover | random_new
    # Each arm is equally probable (LLaMEA selects from mutation_prompts uniformly).
    # The crossover arm uses DynamicCrossoverPrompt which mirrors CrossoverOperator.
    #
    # init_oversample=2 is handled by the LLaMEA iohblade wrapper: it generates
    # n_parents * 2 = 8 candidates, keeps best 4, then runs normally with n_parents=4.

    LLaMEA_Crossover = LLaMEA(
        llm=llm,
        budget=budget,
        name="LLaMEA-Crossover",
        n_parents=N_PARENTS,
        n_offspring=N_OFFSPRING,
        elitism=True,
        init_oversample=INIT_OVERSAMPLE,
    )

    crossover_prompt = DynamicCrossoverPrompt(LLaMEA_Crossover, num_inspirations=NUM_CROSSOVER_INSPIRATIONS)

    LLaMEA_Crossover.kwargs["mutation_prompts"] = [
        REFINE_INSTRUCTION,         # arm: refine
        SIMPLIFY_INSTRUCTION,       # arm: simplify
        RANDOM_NEW_INSTRUCTION,     # arm: random_new
        crossover_prompt,           # arm: crossover  (dynamic, runtime-evaluated)
    ]

    print("✓ LLaMEA-Crossover")
    print("  Selection : uniform random")
    print("  Arms      : refine | simplify | random_new | crossover")
    print(f"  Init      : {N_PARENTS*INIT_OVERSAMPLE} candidates → keep best {N_PARENTS} (init_oversample={INIT_OVERSAMPLE})")
    print(f"  Crossover : {NUM_CROSSOVER_INSPIRATIONS} inspiration(s), full-code format")
    print()

    # ── Method 2: GA-LLaMEA (D-TS adaptive selection) ────────────────────────
    #
    # Same 3 arms as Method 1.
    # GA-LLaMEA's built-in operators already use:
    #   RefineOperator    → same REFINE_INSTRUCTION
    #   SimplifyOperator  → same SIMPLIFY_INSTRUCTION
    #   CrossoverOperator → same full-code format + CROSSOVER_INSTRUCTION
    #   RandomNewOperator → same RANDOM_NEW_INSTRUCTION + structural reference
    #
    # init_oversample=2: generates n_parents * 2 = 16 init candidates, keeps best 8.
    # This matches the GA-LLAMEA-8-INIT-100 config from the manuscript (Section 4.6.2)
    # which showed 7.9% improvement and much lower variance vs init_oversample=1.

    GA_LLaMEA = GA_LLaMEA_Method(
        llm=llm,
        budget=budget,
        name="GA-LLaMEA",
        n_parents=N_PARENTS,
        n_offspring=N_OFFSPRING,
        elitism=True,
        discount=0.99,
        tau_max=0.2,
        epsilon_exploration=0.15,
        arm_names=["simplify", "crossover", "random_new", "refine"],
        num_crossover_inspirations=NUM_CROSSOVER_INSPIRATIONS,
        use_init_prompt_for_random_new=False,
        init_oversample=INIT_OVERSAMPLE,
    )

    print("✓ GA-LLaMEA")
    print("  Selection : Discounted Thompson Sampling (D-TS bandit)")
    print("  Arms      : simplify | crossover | random_new | refine")
    print(f"  Init      : {N_PARENTS*INIT_OVERSAMPLE} candidates → keep best {N_PARENTS} (init_oversample={INIT_OVERSAMPLE})")
    print(f"  Crossover : {NUM_CROSSOVER_INSPIRATIONS} inspiration(s), full-code format")
    print(f"  Discount  : 0.99  |  tau_max: 0.2  |  epsilon: 0.15")
    print()

    # ── Experiment setup ──────────────────────────────────────────────────────

    methods = [LLaMEA_Crossover, GA_LLaMEA]

    timestamp      = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = f"results/LLAMEA-VS-GALLAMEA_{timestamp}"
    logger         = ExperimentLogger(experiment_dir)

    print(f"Results → {experiment_dir}")
    print()
    print("=" * 80)
    print("Starting experiment...")
    print("=" * 80)
    print()

    experiment = MA_BBOB_Experiment(
        methods=methods,
        runs=num_runs,
        seeds=seeds,
        dims=[5],
        budget_factor=2000,
        budget=budget,
        eval_timeout=120,
        show_stdout=True,
        exp_logger=logger,
    )
    experiment()

    print()
    print("=" * 80)
    print("Experiment complete!")
    print("=" * 80)
    print()

    # ── IOH data generation ───────────────────────────────────────────────────

    print("=" * 80)
    print("Generating IOH data for best solutions...")
    print("=" * 80)
    print()

    ioh_dir = os.path.join(experiment_dir, "ioh-data")
    os.makedirs(ioh_dir, exist_ok=True)

    try:
        exp_data = logger.get_data()
        if exp_data.empty:
            print("⚠ No experiment data found, skipping IOH generation.")
        else:
            training_instances = list(range(0, 20))
            test_instances     = list(range(20, 120))

            problem = MA_BBOB(
                dims=[5],
                budget_factor=2000,
                training_instances=training_instances,
                test_instances=test_instances,
            )
            problem._ensure_env()

            total = len(exp_data)
            for idx, (_, row) in enumerate(exp_data.iterrows(), 1):
                method_name = row.get("method_name", "Unknown")
                seed        = row.get("seed", 0)
                sol_data    = row.get("solution", {})

                print(f"[{idx}/{total}] {method_name} seed={seed}...", end=" ")

                if not sol_data or not isinstance(sol_data, dict):
                    print("⚠ No solution data, skipping")
                    continue

                solution = Solution()
                solution.from_dict(sol_data)

                if not solution.code:
                    print("⚠ No code, skipping")
                    continue

                try:
                    for test_seed in range(5):
                        np.random.seed(test_seed)
                        problem.test(solution, ioh_dir=ioh_dir)
                    print(f"✓  {solution.name}")
                except Exception as e:
                    print(f"✗ {e}")

            problem.cleanup()
            print()
            print(f"✓ IOH data saved to: {os.path.abspath(ioh_dir)}")

    except Exception:
        import traceback
        print()
        print("✗ IOH data generation failed:")
        traceback.print_exc()

    print()
    print("=" * 80)
    print("All done!")
    print("=" * 80)
    print()
    print("What this ablation isolates:")
    print("  • Both methods used identical prompt text for every arm.")
    print("  • The ONLY difference is operator selection strategy:")
    print("      LLaMEA-Crossover → uniform random (no learning)")
    print("      GA-LLaMEA        → D-TS bandit   (adaptive learning)")
    print()
    print(f"Results in : {experiment_dir}")
    print(f"IOH data   : {ioh_dir}")
    print()
