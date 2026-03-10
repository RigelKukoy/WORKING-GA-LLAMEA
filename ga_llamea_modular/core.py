"""
GA-LLAMEA Core Algorithm
========================

This module contains the main GA_LLaMEA class that orchestrates the evolutionary
process using Discounted Thompson Sampling for operator selection.

ALGORITHM OVERVIEW:
    GA-LLAMEA is an extension of LLaMEA (Large Language Model Evolutionary Algorithm)
    that uses adaptive operator selection instead of fixed mutation. The algorithm:
    
    1. Initializes a population of algorithms by querying the LLM
    2. For each generation:
        a. Select operator using D-TS bandit (mutation/crossover/random_new)
        b. Generate offspring using selected operator
        c. Evaluate offspring fitness
        d. Update bandit with reward (fitness improvement)
        e. Select best individuals for next generation
    3. Return best algorithm found

BLADE INTEGRATION:
    GA_LLaMEA is designed to work seamlessly with BLADE:
    
    1. Pass BLADE's classes directly - no adapters needed:
        - llm: BLADE's LLM instance (Gemini_LLM, OpenAI_LLM, etc.)
        - solution_class: BLADE's Solution class
        - problem: BLADE's Problem subclass (MA_BBOB, BBOB_SBOX, etc.)
    
    2. Works with BLADE's Experiment class for automated runs:
        experiment = Experiment(methods=[method], problems=[problem])
    
    3. Logging integrates with BLADE's ExperimentLogger

KEY PARAMETERS:
    - budget: Total LLM queries allowed (controls experiment duration)
    - n_parents (μ): Population size (default 4)
    - n_offspring (λ): Generated per generation (default 16)
    - discount: D-TS forgetting factor (default 0.99)
    - epsilon_exploration: Random arm selection floor (default 0.4)
    - elitism: (μ+λ) if True, (μ,λ) if False

USAGE:
    from ga_llamea_modular import GA_LLaMEA
    from iohblade.llm import Gemini_LLM
    from iohblade.solution import Solution
    from iohblade.problems import MA_BBOB
    
    llm = Gemini_LLM(api_key="...", model="gemini-2.0-flash")
    method = GA_LLaMEA(
        llm=llm,
        budget=100,
        solution_class=Solution,
    )
    
    problem = MA_BBOB(dims=[5], budget_factor=2000)
    best = method(problem)
    print(f"Best fitness: {best.fitness}")
"""

import random
from typing import Any, List, Optional, Tuple, Type

from .bandit import DiscountedThompsonSampler
from .interfaces import LLMProtocol, ProblemProtocol, SolutionProtocol
from .operators import SimplifyOperator, CrossoverOperator, RandomNewOperator, RefineOperator
from .utils import (
    calculate_reward,
    extract_code,
    extract_description,
    extract_classname,
    create_solution,
    validate_code,
    DefaultSolution,
)


class GA_LLaMEA:
    """GA-LLAMEA: LLaMEA with Discounted Thompson Sampling for operator selection.
    
    This method uses a multi-armed bandit (D-TS) to adaptively select between
    genetic operators:
        - **Simplify**: Simplify and improve a single parent
        - **Crossover**: Combine the best elements of two parents
        - **Random New**: Generate a completely new algorithm
    
    The bandit learns which operator produces the highest-quality solutions,
    using absolute fitness as the reward signal (no per-operator baselines).
    Stagnation detection forces explorative operators (crossover/random_new)
    when the best fitness hasn't improved for a configurable number of evaluations.
    
    Attributes:
        llm: LLM instance for code generation
        budget: Total LLM query budget
        name: Method name for logging
        n_parents: Population size (μ)
        n_offspring: Offspring per generation (λ)
        elitism: Use (μ+λ) selection if True
        bandit: D-TS bandit for operator selection
        population: Current population of solutions
        best_solution: Best solution found so far
        generation: Current generation number
    
    Example:
        >>> llm = Gemini_LLM(api_key="...", model="gemini-2.0-flash")
        >>> method = GA_LLaMEA(llm=llm, budget=100, solution_class=Solution)
        >>> problem = MA_BBOB(dims=[5])
        >>> best = method(problem)
    """

    def __init__(
        self,
        llm: LLMProtocol,
        budget: int,
        solution_class: Type[Any] = None,
        name: str = "GA-LLAMEA",
        n_parents: int = 4,
        n_offspring: int = 16,
        elitism: bool = True,
        discount: float = 0.99,
        tau_max: float = 0.20,
        epsilon_exploration: float = 0.4,
        arm_names: Optional[List[str]] = None,
        always_select_best: bool = False,
        use_init_prompt_for_random_new: bool = False,
        num_crossover_inspirations: int = 1,
        min_pulls_per_arm: int = 5,
        **kwargs,
    ):
        """Initialize GA-LLAMEA.
        
        Args:
            llm: LLM instance for code generation. Must have query() method.
                 BLADE's LLM classes (Gemini_LLM, OpenAI_LLM, etc.) work directly.
            
            budget: Total number of LLM queries allowed.
            
            solution_class: Solution class to use for storing algorithms.
                           Pass BLADE's Solution class: `solution_class=Solution`
                           If None, uses a minimal DefaultSolution.
            
            name: Method name for logging and identification.
            
            n_parents: Population size (mu). Default 4.
            
            n_offspring: Offspring per generation (lambda). Default 16.
            
            elitism: True = (mu+lambda) selection, False = (mu,lambda).
            
            discount: D-TS discount factor gamma in (0, 1]. Default 0.99.
                     Gentle discount preserves DTS adaptation while using
                     nearly all observations at typical budgets (~100).
            
            tau_max: Maximum sampling std dev for D-TS. Paper recommends
                     tau_max ~ mu_max/5. Default 0.20.
            
            epsilon_exploration: Probability of random arm selection (exploration
                               floor). Prevents permanent arm extinction.
                               0.0 = pure TS, 1.0 = fully random. Default 0.4.
            
            arm_names: List of operator arm names. Default: ["simplify", "crossover", "random_new"].
            
            min_pulls_per_arm: Minimum number of times each operator must be selected
                              before the bandit strategy takes over. This "burn-in"
                              phase ensures initial statistics are based on real data.
                              Default 5.
            
            **kwargs: Additional arguments stored but not used directly.
        """
        self.llm = llm
        self.budget = budget
        self.name = name
        self.n_parents = n_parents
        self.n_offspring = n_offspring
        self.elitism = elitism
        self.always_select_best = always_select_best
        self.num_crossover_inspirations = num_crossover_inspirations
        self.min_pulls_per_arm = min_pulls_per_arm
        self.kwargs = kwargs

        # Solution factory
        if solution_class is not None:
            self._solution_class = solution_class
        else:
            self._solution_class = DefaultSolution

        if arm_names is None:
            arm_names = ["simplify", "crossover", "random_new"]

        self.bandit = DiscountedThompsonSampler(
            arm_names=arm_names,
            discount=discount,
            tau_max=tau_max,
            epsilon_exploration=epsilon_exploration,
            min_pulls=min_pulls_per_arm,
        )

        # Initialize operators
        self._simplify = SimplifyOperator()
        self._refine = RefineOperator()
        self._crossover = CrossoverOperator(num_inspirations=num_crossover_inspirations)
        self._random_new = RandomNewOperator(use_init_prompt=use_init_prompt_for_random_new)

        # State
        self.population: List[Any] = []
        self.best_solution: Optional[Any] = None
        self.generation = 0
        self.llm_calls = 0
        
        # Stagnation detection: force crossover/random_new when no improvement
        self._stagnation_counter = 0
        self._stagnation_threshold = 10
        self._best_fitness_at_last_improvement = -float('inf')
        
        # Arm history for post-hoc analysis (does not affect algorithm behavior)
        self.arm_history: List[dict] = []

    def __call__(self, problem: ProblemProtocol) -> Any:
        """Execute GA-LLAMEA evolution.
        
        This is the main entry point. BLADE's Experiment class calls this method.
        
        Args:
            problem: Problem instance to optimize. Must have evaluate() method
                    and prompt attributes. BLADE's Problem subclasses work directly.
        
        Returns:
            Best solution found (Solution object with code and fitness)
        
        Raises:
            RuntimeError: If population initialization fails completely
        """
        # Phase 1: Initialize population
        self._initialize_population(problem)
        self.llm_calls = self.n_parents
        
        if not self.population:
            raise RuntimeError(
                "Population initialization failed - no valid solutions were generated. "
                "Check that the LLM is responding correctly and generating valid code."
            )

        # Phase 2: Evolutionary loop
        self.generation = 0
        max_safety_generations = self.budget * 2  # Safety guard

        while self.llm_calls < self.budget:
            self.generation += 1
            if self.generation > max_safety_generations:
                print(f"Stopping early: Exceeded safety limit of {max_safety_generations} generations with insufficient valid solutions.")
                break
            
            current_best_fit = self.best_solution.fitness if self.best_solution else -float('inf')
            print(f"Generation {self.generation}: Best Fitness = {current_best_fit:.4f}")

            offspring = self._generate_offspring(problem)

            # Selection
            if self.elitism:
                # (μ+λ): Select from parents + offspring
                combined = self.population + offspring
                self.population = self._select(combined, self.n_parents)
            else:
                # (μ,λ): Select only from offspring  
                self.population = self._select(offspring, self.n_parents)

            # Update best solution
            if self.population:
                current_best = max(self.population, key=lambda s: s.fitness)
                if self.best_solution is None or current_best.fitness > self.best_solution.fitness:
                    self.best_solution = current_best

        return self.best_solution if self.best_solution else self.population[0]

    def _initialize_population(self, problem: ProblemProtocol) -> None:
        """Initialize population with diverse algorithms.
        
        Generates n_parents initial solutions by querying the LLM with
        the task description and example code.
        """
        for i in range(self.n_parents):
            if self.llm_calls >= self.budget:
                break

            try:
                prompt = self._random_new.build_prompt(problem, [], is_init=True)
                child = self._generate_solution(
                    prompt, problem,
                    parent_ids=[],
                    operator="init",
                    generation=0,
                )
                self.arm_history.append({"eval": self.llm_calls, "operator": "init"})
                
                # Code extraction failure — log manually and skip
                if child.error:
                    if hasattr(self.llm, "logger") and hasattr(self.llm.logger, "log_individual"):
                        self.llm.logger.log_individual(child)
                    continue

                # Set metadata before evaluation (preserved through subprocess pickling)
                child.metadata["generation"] = 0
                child.metadata["operator"] = "init"

                # Evaluate via Problem.__call__() for subprocess isolation,
                # timeout enforcement, and consistent error handling with LLaMEA/EoH.
                # Problem.__call__() handles logging internally.
                child = problem(child)

                if not child.error:
                    self.population.append(child)
                else:
                    print(f"Evaluation failed for {child.name}: {child.error}")
            except Exception as e:
                import traceback
                traceback.print_exc()

        if self.population:
            self.best_solution = max(self.population, key=lambda s: s.fitness)
            print(f"   [INFO] Initialization complete. Best Fitness: {self.best_solution.fitness:.4f}")

    def _generate_offspring(self, problem: ProblemProtocol) -> List[Any]:
        """Generate offspring using adaptive operator selection.
        
        For each offspring slot, the D-TS bandit selects an operator
        (or stagnation detection overrides with crossover/random_new),
        the appropriate parent(s) are chosen, and the reward is the
        child's absolute fitness so all operators are compared fairly.
        """
        offspring = []
        
        for _ in range(self.n_offspring):
            if self.llm_calls >= self.budget:
                break

            # Stagnation override: one-shot kick with explorative operator,
            # then reset counter so the bandit can try refine/simplify again.
            if self._stagnation_counter >= self._stagnation_threshold:
                explorative_arms = [a for a in self.bandit.arm_names
                                    if a in ("crossover", "random_new")]
                if not explorative_arms:
                    explorative_arms = self.bandit.arm_names
                operator_name = random.choice(explorative_arms)
                theta = 0.0
                self._stagnation_counter = 0
            else:
                operator_name, theta = self.bandit.select_arm()

            try:
                if operator_name == "simplify":
                    parent = self._select_parent()
                    prompt = self._simplify.build_prompt(problem, self.population, parent)
                    parent_ids = [parent.id]
                elif operator_name == "refine":
                    parent = self._select_parent()
                    prompt = self._refine.build_prompt(problem, self.population, parent)
                    parent_ids = [parent.id]
                elif operator_name == "crossover":
                    parent1, inspirations = self._select_crossover_parents()
                    prompt = self._crossover.build_prompt(problem, self.population, parent1, inspirations=inspirations)
                    parent_ids = [parent1.id] + [p.id for p in inspirations]
                else:  # random_new
                    prompt = self._random_new.build_prompt(problem, self.population)
                    parent_ids = []

                child = self._generate_solution(
                    prompt, problem,
                    parent_ids=parent_ids,
                    operator=operator_name,
                    generation=self.generation,
                )
                self.arm_history.append({"eval": self.llm_calls, "operator": operator_name})
                
                if child.error:
                    if hasattr(self.llm, "logger") and hasattr(self.llm.logger, "log_individual"):
                        self.llm.logger.log_individual(child)
                    self.bandit.update(operator_name, calculate_reward(0.0, is_valid=False))
                    self._stagnation_counter += 1
                    continue

                is_valid, validation_error = validate_code(child.code)
                if not is_valid:
                    child.error = f"Validation failed: {validation_error}"
                    if hasattr(self.llm, "logger") and hasattr(self.llm.logger, "log_individual"):
                        self.llm.logger.log_individual(child)
                    self.bandit.update(operator_name, calculate_reward(0.0, is_valid=False))
                    self._stagnation_counter += 1
                    print(f"Code validation failed for {child.name}: {validation_error}")
                    continue

                child.metadata["operator"] = operator_name
                child.metadata["theta_sampled"] = theta
                child.metadata["generation"] = self.generation

                child = problem(child)

                if not child.error:
                    offspring.append(child)

                    reward = calculate_reward(child.fitness, is_valid=True)
                    self.bandit.update(operator_name, reward)
                    child.metadata["reward"] = reward

                    # Stagnation tracking
                    if child.fitness > self._best_fitness_at_last_improvement:
                        self._best_fitness_at_last_improvement = child.fitness
                        self._stagnation_counter = 0
                    else:
                        self._stagnation_counter += 1
                else:
                    failure_reward = calculate_reward(0.0, is_valid=False)
                    self.bandit.update(operator_name, failure_reward)
                    child.metadata["reward"] = failure_reward
                    self._stagnation_counter += 1
                    print(f"Evaluation failed for {child.name}: {child.error}")

            except Exception as e:
                self.bandit.update(operator_name, calculate_reward(0.0, is_valid=False))
                self._stagnation_counter += 1
                continue
                
        return offspring

    def _generate_solution(
        self,
        prompt: str,
        problem: ProblemProtocol,
        parent_ids: Optional[list] = None,
        operator: Optional[str] = None,
        generation: int = 0,
    ) -> Any:
        """Generate a solution by querying LLM and extracting code.
        
        Args:
            prompt: The prompt to send to the LLM.
            problem: The problem instance.
            parent_ids: List of parent solution UUIDs for CEG lineage.
            operator: Name of the operator that requested this solution.
            generation: Current generation number.
        """
        response = ""
        try:
            self.llm_calls += 1 # Count every attempt
            
            # BLADE's LLM expects list of message dicts
            session_messages = [{"role": "user", "content": prompt}]
            response = self.llm.query(session_messages)

            code = extract_code(response)
            if not code:
                # Create failed solution with lineage info
                solution = create_solution(
                    self._solution_class, code="", fitness=-float('inf'),
                    parent_ids=parent_ids, operator=operator, generation=generation,
                )
                solution.metadata["llm_response"] = response
                solution.error = "No code extracted from response"
                return solution

            # Create solution using factory with lineage info
            solution = create_solution(
                self._solution_class, code=code, fitness=-float('inf'),
                parent_ids=parent_ids, operator=operator, generation=generation,
            )
            solution.metadata["llm_response"] = response

            description = extract_description(response)
            if description:
                solution.description = description

            return solution

        except Exception as e:
            print(f"   [DEBUG] _generate_solution error: {e}")
            solution = create_solution(
                self._solution_class, code="", fitness=-float('inf'),
                parent_ids=parent_ids, operator=operator, generation=generation,
            )
            solution.metadata["llm_response"] = response
            solution.error = f"Generation error: {str(e)}"
            return solution

    def _select_parent(self, tournament_size: int = 2) -> Any:
        """Select a parent randomly (uniform selection).
        
        Changed from tournament selection to random selection to match standard LLaMEA.
        
        Args:
            tournament_size: Ignored. Kept for signature compatibility.
        """
        if self.always_select_best:
            return max(self.population, key=lambda s: s.fitness)

        return random.choice(self.population)
    
    def _select_parent_from(self, pool: List[Any], tournament_size: int = 2) -> Any:
        """Select a parent from a specific pool randomly (uniform selection).
        
        Changed from tournament selection to random selection.
        
        Args:
            pool: List of candidate solutions to select from.
            tournament_size: Ignored. Kept for signature compatibility.
        """
        return random.choice(pool)

    def _select_crossover_parents(self) -> Tuple[Any, List[Any]]:
        """Select diverse parents for crossover using random selection.
        
        Uses uniform random selection for parents with a diversity constraint:
        inspirations are randomly selected from solutions with *different code* than parent1.
        
        The parents are ordered so parent1 has the highest fitness (used as the
        "foundation" in the crossover prompt).
        """
        parent1 = self._select_parent()
        
        inspirations = []
        remaining = [p for p in self.population if p.code != parent1.code]
        if not remaining:
            # All have same code; pick any other individual
            remaining = [p for p in self.population if p is not parent1]
            
        for _ in range(self.num_crossover_inspirations):
            if not remaining:
                if not inspirations:
                    # Population of 1; crossover degrades to mutation-like
                    inspirations = [parent1]
                break
                
            insp = self._select_parent_from(remaining)
            inspirations.append(insp)
            
            # Remove selected inspiration to ensure diverse inspirations
            remaining = [p for p in remaining if p.code != insp.code]
            if not remaining:
                remaining = [p for p in self.population if p is not parent1 and p not in inspirations]
        
        # Ensure parent1 has higher fitness than all inspirations
        best_insp = max(inspirations, key=lambda s: s.fitness)
        if best_insp.fitness > parent1.fitness:
            inspirations.remove(best_insp)
            inspirations.append(parent1)
            parent1 = best_insp
        
        return parent1, inspirations

    def _select(self, population: List[Any], n: int) -> List[Any]:
        """Select best n individuals with diversity preservation.
        
        Selection prioritizes fitness but also ensures diversity by
        preferring individuals with different code.
        """
        if not population:
            return []

        sorted_pop = sorted(population, key=lambda s: s.fitness, reverse=True)

        # Always include best
        selected = [sorted_pop[0]]
        if n == 1:
            return selected

        # Add diverse individuals (different code)
        seen_codes = {sorted_pop[0].code}
        for sol in sorted_pop[1:]:
            if len(selected) >= n:
                break
            if sol.code not in seen_codes:
                selected.append(sol)
                seen_codes.add(sol.code)

        # Fill remaining slots if needed
        for sol in sorted_pop:
            if len(selected) >= n:
                break
            if sol not in selected:
                selected.append(sol)

        return selected[:n]

    def to_dict(self) -> dict:
        """Return dictionary representation for logging.
        
        Used by BLADE's logging system to record method configuration.
        """
        return {
            "method_name": self.name,
            "budget": self.budget,
            "n_parents": self.n_parents,
            "n_offspring": self.n_offspring,
            "elitism": self.elitism,
            "min_pulls_per_arm": self.min_pulls_per_arm,
            "bandit_state": self.bandit.get_state_snapshot() if hasattr(self, "bandit") else {},
            **self.kwargs,
        }
