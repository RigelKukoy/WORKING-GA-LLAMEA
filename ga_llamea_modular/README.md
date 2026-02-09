# GA-LLAMEA Modular Package

A modular, well-documented implementation of **GA-LLAMEA** (LLaMEA with Discounted Thompson Sampling) designed for seamless integration with the BLADE framework.

## Overview

GA-LLAMEA extends the standard LLaMEA (Large Language Model Evolutionary Algorithm) by adding **adaptive operator selection** through a Discounted Thompson Sampling bandit. Instead of using only mutation, GA-LLAMEA dynamically chooses between four operators:

| Operator | Description | Use Case |
|----------|-------------|----------|
| **Simplify** | Simplify and improve a parent | Reduce complexity, fewer errors |
| **Crossover** | Combine best elements of two parents | Hybridize strategies |
| **Random New** | Generate from scratch | Explore new approaches |

The simplify operator matches LLAMEA's proven Prompt5 instructions, while crossover is GA-LLAMEA's unique addition.

The bandit learns which operator works best for the current problem and adapts over time.

## Package Structure

```
ga_llamea_modular/
├── __init__.py      # Package exports
├── README.md        # This file
├── bandit.py        # Discounted Thompson Sampling (D-TS) bandit
├── operators.py     # Genetic operators (simplify, crossover, random_new)
├── core.py          # Main GA_LLaMEA class
├── interfaces.py    # Protocol definitions for BLADE compatibility
└── utils.py         # Helper functions
```

## Quick Start

### Basic Usage with BLADE

```python
from ga_llamea_modular import GA_LLaMEA

from iohblade.llm import Gemini_LLM
from iohblade.solution import Solution
from iohblade.problems import MA_BBOB
from iohblade.experiment import Experiment
from iohblade.loggers import ExperimentLogger

# 1. Initialize LLM
llm = Gemini_LLM(api_key="your-api-key", model="gemini-2.0-flash")

# 2. Create GA-LLAMEA method - PASS Solution CLASS
method = GA_LLaMEA(
    llm=llm,
    budget=100,
    solution_class=Solution,  # IMPORTANT: Pass BLADE's Solution class
)

# 3. Create problem
problem = MA_BBOB(dims=[5], budget_factor=2000)

# 4. Run experiment
logger = ExperimentLogger("results/ga_llamea")
experiment = Experiment(
    methods=[method],
    problems=[problem],
    runs=5,
    exp_logger=logger
)
experiment()
```

### Direct Usage (Without Experiment)

```python
from ga_llamea_modular import GA_LLaMEA
from iohblade.llm import Gemini_LLM
from iohblade.solution import Solution
from iohblade.problems import MA_BBOB

llm = Gemini_LLM(api_key="...", model="gemini-2.0-flash")
method = GA_LLaMEA(llm=llm, budget=50, solution_class=Solution)

problem = MA_BBOB(dims=[5])
best_solution = method(problem)

print(f"Best Fitness: {best_solution.fitness:.4f}")
print(f"Best Code:\n{best_solution.code}")
```

## Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `llm` | LLM | Required | BLADE's LLM instance |
| `budget` | int | Required | Total LLM queries allowed |
| `solution_class` | Type | None | Solution class (use `Solution` from BLADE) |
| `n_parents` | int | 4 | Population size (μ) |
| `n_offspring` | int | 16 | Offspring per generation (λ) |
| `elitism` | bool | True | Use (μ+λ) selection |
| `discount` | float | 0.9 | D-TS discount factor (0.9 = 10% decay) |
| `tau_max` | float | 1.0 | Max posterior uncertainty |

### Parameter Tuning Tips

- **budget**: Start with 50-100 for testing, use 500+ for real experiments
- **discount**: Lower (0.7-0.8) for faster adaptation, higher (0.95) for stable learning
- **n_parents**: Larger populations (8-16) for harder problems
- **elitism=False**: More exploration, but can lose good solutions

## Module Details

### `bandit.py` - Discounted Thompson Sampling

The D-TS bandit maintains a posterior distribution for each operator and samples to select:

```python
from ga_llamea_modular.bandit import DiscountedThompsonSampler

bandit = DiscountedThompsonSampler(
    arm_names=["simplify", "crossover", "random_new"],
    discount=0.9,  # 10% decay per observation
)

# Select operator
operator, theta = bandit.select_arm()

# Update after observing reward
bandit.update(operator, reward=0.5)

# Get statistics
state = bandit.get_state_snapshot()
print(f"Simplify pulls: {state['simplify']['pulls']}")
```

### `operators.py` - Genetic Operators

Each operator generates prompts for the LLM:

```python
from ga_llamea_modular.operators import SimplifyOperator, CrossoverOperator

simplify = SimplifyOperator()
prompt = simplify.build_prompt(problem, population, parent=best_solution)

crossover = CrossoverOperator()
prompt = crossover.build_prompt(problem, population, parent=sol1, parent2=sol2)
```

### `interfaces.py` - Protocol Definitions

Defines the interfaces that BLADE classes automatically satisfy:

```python
from ga_llamea_modular.interfaces import LLMProtocol, SolutionProtocol, ProblemProtocol

# BLADE's classes satisfy these protocols automatically
# No adapters or wrappers needed!
```

## BLADE Integration Details

### Why It Works Without Modifying BLADE

GA-LLAMEA uses Python **Protocols** (PEP 544) for structural typing. Instead of requiring BLADE classes to inherit from our interfaces, we define what methods/attributes are needed, and Python checks at runtime if the classes have them.

BLADE's classes naturally satisfy our protocols:
- `LLM.query()` → `LLMProtocol.query()`
- `Solution.code, .fitness, .error, .name, .metadata` → `SolutionProtocol`
- `Problem.evaluate(), .task_prompt, .format_prompt` → `ProblemProtocol`

### Compatible LLM Classes

Works with all BLADE LLM implementations:
- `Gemini_LLM` (Google Gemini)
- `OpenAI_LLM` (GPT-4, etc.)
- `Claude_LLM` (Anthropic Claude)
- `DeepSeek_LLM` (DeepSeek)
- `Ollama_LLM` (Local models)

### Compatible Problem Classes

Works with all BLADE problem types:
- `MA_BBOB` (Many-Affine BBOB)
- `BBOB_SBOX` (SBOX-COST)
- `BBOB` (Standard BBOB)
- Custom problems extending `Problem`

## Comparison with LLaMEA Baseline

| Feature | LLaMEA | GA-LLAMEA |
|---------|--------|-----------|
| Operators | Mutation only (3 prompts) | Simplify, Crossover, Random New |
| Selection | Fixed/round-robin | Adaptive (D-TS bandit) |
| Dependencies | `llamea` PyPI package | Pure Python, no external deps |
| Adaptability | Static | Learns best operator over time |
| Reward Signal | N/A | Parent-relative (per-operator baseline) |
| Code Validation | None | Pre-evaluation static checks |

## Troubleshooting

### "No code extracted from LLM response"

The LLM response didn't contain code in the expected format. Check:
1. LLM is responding correctly
2. API key is valid
3. Model supports code generation

### "Population initialization failed"

All initial solutions failed to generate. Causes:
- LLM quota exceeded
- Network issues
- Problem configuration error

### Low fitness scores

1. Increase budget for more exploration
2. Adjust discount (lower = faster adaptation)
3. Check if problem is set up correctly

## License

MIT License - see BLADE project for details.
