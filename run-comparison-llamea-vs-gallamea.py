"""
LLaMEA-Crossover with GA-LLaMEA-Matched Prompts
=================================================

Runs LLaMEA-Crossover (uniform random operator selection) with prompts that
are structurally identical to GA-LLaMEA's internal operators.

PatchedLLaMEA overrides construct_prompt() for ALL four arms so that the
prompt format — header, history, parent section, instruction — exactly mirrors
what GA-LLaMEA's RefineOperator / SimplifyOperator / CrossoverOperator /
RandomNewOperator produce. This means the only remaining variable when comparing
this run against a GA-LLaMEA run is the operator selection strategy:
  LLaMEA-Crossover = uniform random
  GA-LLaMEA        = Discounted Thompson Sampling (D-TS bandit)

Prompt format per arm (matches GA-LLaMEA exactly):
  refine     → role + task + example | history | Selected algorithm… Name/Fitness/Code | instruction
  simplify   → role + task + example | history | Selected algorithm… Name/Fitness/Code | instruction
  crossover  → role + task + example | history | Working Algorithm… | inspirations | instruction
  random_new → role + task + example | history | structural skeleton | instruction  (no parent)
"""

import os
import random
import numpy as np
from datetime import datetime
from dotenv import load_dotenv

from llamea import LLaMEA as _LLAMEA_Algorithm

from iohblade.experiment import MA_BBOB_Experiment
from iohblade.llm import AIML_LLM
from iohblade.loggers import ExperimentLogger
from iohblade.solution import Solution
from iohblade.problems import MA_BBOB
from iohblade.methods.llamea import LLaMEA
from iohblade.methods.ga_llamea import GA_LLaMEA_Method


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

# Compound prompt for LLaMEA's random_new arm.
# PatchedLLaMEA detects this arm via RANDOM_NEW_MARKER and builds the prompt
# without any parent code — matching RandomNewOperator's behaviour exactly.
RANDOM_NEW_MUTATION_PROMPT = STRUCTURAL_REFERENCE + "\n\n" + RANDOM_NEW_INSTRUCTION


# ─────────────────────────────────────────────────────────────────────────────
#  PatchedLLaMEA — full GA-LLaMEA prompt format for all four arms
#
#  Overrides construct_prompt() entirely so that each arm's prompt is
#  structurally identical to the corresponding GA-LLaMEA operator:
#    refine     → RefineOperator.build_prompt()
#    simplify   → SimplifyOperator.build_prompt()
#    crossover  → CrossoverOperator.build_prompt()
#    random_new → RandomNewOperator.build_prompt()  (no parent code)
#
#  Differences eliminated vs the original LLaMEA construct_prompt():
#    ✓ example_prompt now included in every mutation call
#    ✓ History format: "- name: X.XXXX" sorted by fitness (not "name: desc (Score: X)")
#    ✓ Parent section: Name/Fitness/Code labels (not "The selected solution to update")
#    ✓ Feedback section: removed (GA-LLaMEA operators do not include it)
#    ✓ random_new: no parent shown at all (only structural skeleton)
#    ✓ crossover: "Working Algorithm (fitness: X):" label matching CrossoverOperator
# ─────────────────────────────────────────────────────────────────────────────

class PatchedLLaMEA(_LLAMEA_Algorithm):
    """LLaMEA with construct_prompt() rebuilt to match GA-LLaMEA's exact format."""

    RANDOM_NEW_MARKER = "For correct code structure, follow this template:"

    def _ga_header(self) -> str:
        """role + task + example — matches GA-LLaMEA's _get_task_prompt()."""
        return f"{self.role_prompt}\n{self.task_prompt}\n{self.example_prompt}"

    def _ga_history(self) -> str:
        """Sorted-by-fitness list — matches GA-LLaMEA's _get_population_history()."""
        sorted_pop = sorted(
            self.population,
            key=lambda s: s.fitness if s.fitness is not None and not np.isnan(s.fitness) else -np.inf,
            reverse=True,
        )
        lines = "List of previously generated algorithm names with mean AOCC score:\n"
        for sol in sorted_pop:
            fitness_str = f"{sol.fitness:.4f}" if sol.fitness is not None and np.isfinite(sol.fitness) else "N/A"
            lines += f"- {sol.name}: {fitness_str}\n"
        return lines

    def construct_prompt(self, individual):
        mutation_operator = random.choice(self.mutation_prompts)
        individual.set_operator(mutation_operator)

        # Detect arm without calling __str__ on DynamicCrossoverPrompt twice
        is_crossover = isinstance(mutation_operator, DynamicCrossoverPrompt)
        mutation_str = "" if is_crossover else str(mutation_operator)

        header  = self._ga_header()
        history = self._ga_history()
        fmt     = self.output_format_prompt

        fitness_val = individual.fitness
        fitness_str = f"{fitness_val:.4f}" if fitness_val is not None and np.isfinite(fitness_val) else "N/A"

        if mutation_str.startswith(self.RANDOM_NEW_MARKER):
            # random_new — no parent, only structural skeleton + instruction
            content = (
                f"{header}\n\n"
                f"{history}\n"
                f"{mutation_str}\n\n"
                f"{fmt}"
            )

        elif is_crossover:
            # crossover — matches CrossoverOperator.build_prompt() exactly
            inspirations_str = mutation_operator.get_inspirations_str()
            working_block = (
                f"Working Algorithm (fitness: {fitness_str}):\n"
                f"```python\n{individual.code}\n```"
            )
            if inspirations_str:
                cross_body = (
                    f"{working_block}\n\n"
                    f"These are other high-performing solutions discovered during the search.\n"
                    f"You may borrow useful ideas, logic, or techniques from them.\n\n"
                    f"{inspirations_str}"
                )
            else:
                cross_body = working_block
            content = (
                f"{header}\n\n"
                f"{history}\n"
                f"{cross_body}\n\n"
                f"{CROSSOVER_INSTRUCTION}\n\n"
                f"{fmt}"
            )

        elif mutation_str.startswith("Refine and simplify"):
            # simplify — matches SimplifyOperator.build_prompt()
            content = (
                f"{header}\n\n"
                f"{history}\n"
                f"Selected algorithm to simplify and improve:\n"
                f"Name: {individual.name}\n"
                f"Fitness: {fitness_str}\n"
                f"Code:\n```python\n{individual.code}\n```\n\n"
                f"{mutation_str}\n\n"
                f"{fmt}"
            )

        else:
            # refine (default) — matches RefineOperator.build_prompt()
            content = (
                f"{header}\n\n"
                f"{history}\n"
                f"Selected algorithm to refine and improve:\n"
                f"Name: {individual.name}\n"
                f"Fitness: {fitness_str}\n"
                f"Code:\n```python\n{individual.code}\n```\n\n"
                f"{mutation_str}\n\n"
                f"{fmt}"
            )

        return [{"role": "user", "content": content}]


# ─────────────────────────────────────────────────────────────────────────────
#  DynamicCrossoverPrompt
#  Acts as a sentinel in mutation_prompts so PatchedLLaMEA can detect the
#  crossover arm via isinstance(), and provides get_inspirations_str() so
#  PatchedLLaMEA can build the full CrossoverOperator-format prompt itself.
# ─────────────────────────────────────────────────────────────────────────────

class DynamicCrossoverPrompt:
    """
    Sentinel + inspiration-sampler for the crossover arm.

    PatchedLLaMEA detects this object via isinstance() and calls
    get_inspirations_str() to retrieve the inspiration code blocks.
    It then builds the full prompt in GA-LLaMEA's CrossoverOperator format.
    """

    def __init__(self, llamea_method_wrapper, num_inspirations: int = 3):
        self._wrapper = llamea_method_wrapper
        self.num_inspirations = num_inspirations

    def _valid_population(self):
        llamea_instance = getattr(self._wrapper, "llamea_instance", None)
        if not llamea_instance or not llamea_instance.population:
            return []
        return [
            p for p in llamea_instance.population
            if p.name and p.code and p.fitness is not None and np.isfinite(p.fitness)
        ]

    def get_inspirations_str(self) -> str:
        """Return inspiration code blocks — same format as CrossoverOperator."""
        valid_pop = self._valid_population()
        if not valid_pop:
            return ""
        n = min(self.num_inspirations, len(valid_pop))
        inspirations = random.sample(valid_pop, n)
        blocks = []
        for i, insp in enumerate(inspirations):
            blocks.append(
                f"Inspiration {i + 1}: {insp.name} (fitness: {insp.fitness:.4f})\n"
                f"```python\n{insp.code}\n```"
            )
        return "\n\n".join(blocks)

    def __str__(self) -> str:
        """Fallback string representation (used for operator logging)."""
        return f"crossover ({self.num_inspirations} inspirations)"


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    load_dotenv()

    api_key  = os.getenv("AIML_API_KEY") or os.getenv("AIMLAPI_API_KEY")
    ai_model = "google/gemini-2.0-flash"

    llm = AIML_LLM(api_key=api_key, model=ai_model)

    budget   = 100   # LLM queries per run
    num_runs = 5     # Runs per method for statistical significance
    seeds    = list(range(num_runs))  # [0, 1, 2, 3, 4]

    NUM_CROSSOVER_INSPIRATIONS = 3   # Same for both methods
    N_PARENTS    = 4                 # Working population size (unchanged)
    N_OFFSPRING  = 8                 # Offspring per generation (unchanged)
    INIT_OVERSAMPLE = 2              # Both methods: generate N_PARENTS*2 init candidates, keep best N_PARENTS

    print("=" * 80)
    print("LLaMEA-Crossover vs GA-LLaMEA  (ablation: operator selection only)")
    print("=" * 80)
    print(f"Model          : {ai_model}")
    print(f"Budget         : {budget} LLM queries per run")
    print(f"Runs           : {num_runs}  (seeds {seeds})")
    print(f"n_parents      : {N_PARENTS}  |  n_offspring: {N_OFFSPRING}")
    print(f"init_oversample: {INIT_OVERSAMPLE}  ({N_PARENTS}*{INIT_OVERSAMPLE}={N_PARENTS*INIT_OVERSAMPLE} init candidates → keep best {N_PARENTS})")
    print(f"Inspirations   : {NUM_CROSSOVER_INSPIRATIONS} (crossover arm)")
    print()

    # ── LLaMEA-Crossover (uniform random, GA-LLaMEA-format prompts) ──────────
    #
    # PatchedLLaMEA overrides construct_prompt() for all four arms so the
    # prompt format is identical to GA-LLaMEA's internal operators:
    #   refine     → RefineOperator.build_prompt()    format
    #   simplify   → SimplifyOperator.build_prompt()  format
    #   crossover  → CrossoverOperator.build_prompt() format
    #   random_new → RandomNewOperator.build_prompt() format  (no parent)
    #
    # The ONLY difference vs a GA-LLaMEA run is operator selection:
    #   here  = uniform random
    #   GA    = Discounted Thompson Sampling (D-TS bandit)

    LLaMEA_Crossover = LLaMEA(
        llm=llm,
        budget=budget,
        name="LLaMEA-Crossover",
        algorithm_class=PatchedLLaMEA,
        n_parents=N_PARENTS,
        n_offspring=N_OFFSPRING,
        elitism=True,
        init_oversample=INIT_OVERSAMPLE,
    )

    crossover_prompt = DynamicCrossoverPrompt(LLaMEA_Crossover, num_inspirations=NUM_CROSSOVER_INSPIRATIONS)

    LLaMEA_Crossover.kwargs["mutation_prompts"] = [
        REFINE_INSTRUCTION,           # arm: refine
        SIMPLIFY_INSTRUCTION,         # arm: simplify
        RANDOM_NEW_MUTATION_PROMPT,   # arm: random_new (structural skeleton, no parent)
        crossover_prompt,             # arm: crossover  (DynamicCrossoverPrompt sentinel)
    ]

    print("✓ LLaMEA-Crossover")
    print("  Prompts   : GA-LLaMEA-format (role+task+example | history | parent section)")
    print("  Selection : uniform random")
    print("  Arms      : refine | simplify | random_new | crossover")
    print(f"  Init      : {N_PARENTS*INIT_OVERSAMPLE} candidates → keep best {N_PARENTS} (init_oversample={INIT_OVERSAMPLE})")
    print(f"  Crossover : {NUM_CROSSOVER_INSPIRATIONS} inspiration(s), full-code format")
    print()

    # ── GA-LLaMEA (Discounted Thompson Sampling) ──────────────────────────────

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
        min_pulls_per_arm=0,
    )

    print("✓ GA-LLaMEA")
    print("  Selection : Discounted Thompson Sampling (D-TS bandit)")
    print("  Arms      : simplify | crossover | random_new | refine")
    print(f"  Init      : {N_PARENTS*INIT_OVERSAMPLE} candidates → keep best {N_PARENTS} (init_oversample={INIT_OVERSAMPLE})")
    print(f"  Crossover : {NUM_CROSSOVER_INSPIRATIONS} inspiration(s), full-code format")
    print(f"  Discount  : 0.99  |  tau_max: 0.2  |  epsilon: 0.15  |  min_pulls: 0")
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
    print(f"Results in : {experiment_dir}")
    print(f"IOH data   : {ioh_dir}")
    print()
