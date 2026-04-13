import math

from llamea import LLaMEA as LLAMEA_Algorithm

from ..llm import LLM
from ..method import Method
from ..problem import Problem

# We import the LLaMEA algorithm directly from the pypi package. This has the advantage that we can easily get the latest version.


class LLaMEA(Method):
    def __init__(self, llm: LLM, budget: int, name="LLaMEA", algorithm_class=None, **kwargs):
        """
        Initializes the LLaMEA algorithm within the benchmarking framework.

        Args:
            problem (Problem): The problem instance to optimize.
            llm (LLM): The LLM instance to use for solution generation.
            budget (int): The maximum number of evaluations.
            name (str): The name of the method.
            kwargs: Additional arguments for configuring LLaMEA.
                init_oversample (int): If > 1, generates n_parents * init_oversample
                    candidates during initialization, evaluates all, and keeps the best
                    n_parents. Matches GA-LLaMEA's init_oversample behavior.
                    Default: 1 (standard initialization).
        """
        super().__init__(llm, budget, name)
        self.kwargs = kwargs
        self._algorithm_class = algorithm_class if algorithm_class is not None else LLAMEA_Algorithm

    def __call__(self, problem: Problem):
        """
        Executes the evolutionary search process via LLaMEA.

        Returns:
            Solution: The best solution found.
        """
        # Extract init_oversample before passing kwargs to LLaMEA
        # (LLaMEA does not accept this parameter natively)
        kwargs = dict(self.kwargs)
        init_oversample = kwargs.pop("init_oversample", 1)
        n_parents = kwargs.get("n_parents", 5)

        self.llamea_instance = self._algorithm_class(
            f=problem,  # Ensure evaluation integrates with our framework
            llm=self.llm,
            role_prompt="You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel metaheuristic algorithms to solve black box optimization problems.",
            task_prompt=problem.task_prompt,
            example_prompt=problem.example_prompt,
            output_format_prompt=problem.format_prompt,
            log=None,  # We do not use the LLaMEA native logger, we use the experiment logger instead which is attached on problem level.
            budget=self.budget,
            max_workers=1,  # We do not use parallelization, as it is not supported in combination with the BLADE parrallelization.
            **kwargs,
        )

        if init_oversample > 1:
            # Generate n_parents * init_oversample candidates, keep best n_parents.
            # This mirrors GA-LLaMEA's init_oversample behavior: more diverse starting
            # population without increasing the working population size.
            n_init = n_parents * init_oversample
            oversampled = []
            for _ in range(n_init):
                sol = self.llamea_instance.initialize_single()
                if math.isnan(sol.fitness):
                    sol.fitness = self.llamea_instance.worst_value
                oversampled.append(sol)
                self.llamea_instance.run_history.append(sol)

            # Sort descending by fitness (LLaMEA maximizes)
            oversampled.sort(
                key=lambda s: s.fitness if not math.isnan(s.fitness) else -math.inf,
                reverse=True,
            )

            # Pre-populate with best n_parents so LLaMEA's initialize() is a no-op
            self.llamea_instance.population = oversampled[:n_parents]
            self.llamea_instance.update_best()
            self.llamea_instance.generation += 1

        return self.llamea_instance.run()

    def to_dict(self):
        """
        Returns a dictionary representation of the method including all parameters.

        Returns:
            dict: Dictionary representation of the method.
        """
        return {
            "method_name": self.name if self.name != None else "LLaMEA",
            "budget": self.budget,
            "kwargs": self.kwargs,
        }
