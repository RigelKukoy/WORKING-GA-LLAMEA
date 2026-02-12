# GA-LLAMEA Detailed Prompts Documentation

This document provides a comprehensive overview of all prompts used in GA-LLAMEA (Genetic Algorithm - Large Language Model Evolutionary Algorithm).

## Table of Contents
1. [Overview](#overview)
2. [Base Components](#base-components)
3. [Operator Prompts](#operator-prompts)
   - [Simplify Operator](#simplify-operator)
   - [Crossover Operator](#crossover-operator)
   - [Random New Operator](#random-new-operator)
4. [Prompt Components](#prompt-components)
5. [Design Philosophy](#design-philosophy)

---

## Overview

GA-LLAMEA uses three genetic operators, each with carefully designed prompts:
- **SimplifyOperator**: Simplifies and improves a single parent algorithm
- **CrossoverOperator**: Combines concepts from two parent algorithms using Guided Concept Transfer
- **RandomNewOperator**: Generates completely new algorithms

Each operator builds prompts from standardized components to ensure consistency with LLAMEA.

---

## Base Components

All operators use these shared prompt components:

### 1. Role Prompt
```
You are a highly skilled computer scientist in the field of natural computing. 
Your task is to design novel metaheuristic algorithms to solve black box 
optimization problems.
```

### 2. Task Prompt
Provided by the problem instance (e.g., `MA_BBOB.task_prompt`). This describes:
- The optimization task
- Input/output requirements
- Constraints and specifications

### 3. Example Prompt
Provided by the problem instance (e.g., `MA_BBOB.example_prompt`). Shows:
- Example algorithm implementation
- Expected code structure
- Proper use of the evaluation budget

### 4. Population History
Dynamically generated list of previously tried algorithms:
```
List of previously generated algorithm names with mean AOCC score:
- Algorithm1: 0.8234
- Algorithm2: 0.7891
- Algorithm3: 0.7456
...
```
Sorted by fitness (best first) to provide context on what's been tried.

### 5. Format Prompt
Provided by the problem instance (e.g., `MA_BBOB.format_prompt`). Specifies:
- Required output format
- Class structure requirements
- Naming conventions

---

## Operator Prompts

### Simplify Operator

**Purpose**: Refine and simplify a single parent algorithm to improve its performance and reduce complexity.

**When Used**: Selected by the D-TS bandit when simplification shows good historical rewards.

**Full Prompt Structure**:
```
{role_prompt}
{task_prompt}
{example_prompt}

{population_history}

Selected algorithm to simplify and improve:
Name: {parent.name}
Fitness: {parent.fitness:.4f}
Code:
```python
{parent.code}
```

Refine and simplify the selected algorithm to improve it.

{format_prompt}
```

**Key Features**:
- Shows the complete code of one parent
- Explicitly asks to "simplify" (reduces code bloat)
- Matches LLAMEA's proven Prompt5 instruction verbatim
- Focuses on improvement through reduction of complexity

**Example Rendered Prompt**:
```
You are a highly skilled computer scientist in the field of natural computing.
Your task is to design novel metaheuristic algorithms to solve black box optimization problems.

[Task description from problem...]
[Example code from problem...]

List of previously generated algorithm names with mean AOCC score:
- AdaptiveDE_v2: 0.8234
- ParticleSwarm_Elite: 0.7891
- GeneticAlgorithm_Simple: 0.7456

Selected algorithm to simplify and improve:
Name: AdaptiveDE_v2
Fitness: 0.8234
Code:
```python
class AdaptiveDE_v2:
    def __init__(self, budget):
        self.budget = budget
        # ... algorithm code ...
```

Refine and simplify the selected algorithm to improve it.

[Format instructions from problem...]
```

**Design Rationale**:
- Combats code bloat that occurs from repeated crossover/mutation
- Simpler algorithms are more robust and less prone to bugs
- Proven effective in original LLAMEA (Prompt5)

---

### Crossover Operator

**Purpose**: Create improved algorithms by combining strategic concepts from two parents using **Guided Concept Transfer**.

**When Used**: Selected by the D-TS bandit when hybridization shows good historical rewards.

**Full Prompt Structure**:
```
{role_prompt}
{task_prompt}
{example_prompt}

{population_history}

Working Algorithm (fitness: {parent1.fitness:.4f}):
```python
{parent1.code}
```

Alternative approach for inspiration: "{parent2.name}" (fitness: {parent2.fitness:.4f})
Strategy: {parent2.description}

Create an improved algorithm by redesigning the working algorithm above.
Draw inspiration from the alternative approach "{parent2.name}" — think about 
what strategic concept it might use that could address a weakness in the working 
algorithm.
Write a clean implementation from scratch. Do not copy-paste code.
Use only numpy. Keep it simple and functional.

{format_prompt}
```

**Key Features**:
- Shows **only one parent's full code** (the better performer)
- Shows **only name and description** of the second parent (no code)
- Explicitly instructs to "write from scratch" (prevents Frankenstein code)
- Focuses on **concept transfer** rather than code merging

**Example Rendered Prompt**:
```
You are a highly skilled computer scientist in the field of natural computing.
Your task is to design novel metaheuristic algorithms to solve black box optimization problems.

[Task description from problem...]
[Example code from problem...]

List of previously generated algorithm names with mean AOCC score:
- AdaptiveDE_v2: 0.8234
- ParticleSwarm_Elite: 0.7891
- GeneticAlgorithm_Simple: 0.7456

Working Algorithm (fitness: 0.8234):
```python
class AdaptiveDE_v2:
    def __init__(self, budget):
        self.budget = budget
        self.F = 0.8
        self.CR = 0.9
        # ... complete implementation ...
```

Alternative approach for inspiration: "ParticleSwarm_Elite" (fitness: 0.7891)
Strategy: Uses swarm intelligence with elite-based velocity updates and adaptive inertia

Create an improved algorithm by redesigning the working algorithm above.
Draw inspiration from the alternative approach "ParticleSwarm_Elite" — think about 
what strategic concept it might use that could address a weakness in the working algorithm.
Write a clean implementation from scratch. Do not copy-paste code.
Use only numpy. Keep it simple and functional.

[Format instructions from problem...]
```

**Design Rationale**:
- Traditional crossover prompts (showing two full code snippets) lead to copy-paste failures
- By showing only one code implementation, we force coherent redesign
- Abstract description of second parent forces conceptual understanding
- "Write from scratch" prevents code concatenation errors
- Inspired by EoH's semantic crossover but adapted for better code generation

**Why This Approach Works**:
1. **Avoids Frankenstein Code**: LLM sees only one code structure to build from
2. **Forces Understanding**: Second parent as abstract concept requires comprehension, not copying
3. **Clean Implementation**: Explicit instruction to write from scratch ensures coherent code
4. **Strategic Transfer**: Focuses on transferring ideas, not syntax

---

### Random New Operator

**Purpose**: Generate completely novel algorithms different from existing population.

**When Used**: 
- During initialization (first generation)
- Selected by D-TS bandit for exploration when needed

**Full Prompt Structure (Initialization)**:
```
{role_prompt}
{task_prompt}
{example_prompt}

{format_prompt}
```

**Full Prompt Structure (During Evolution)**:
```
{role_prompt}
{task_prompt}
{example_prompt}

{population_history}

For reference, here is a working algorithm with correct structure and format:
```python
{best_solution.code}
```
Your new algorithm must use a DIFFERENT strategy from the above. Use it only 
as a reference for correct code structure and formatting.

Generate a new algorithm that is different from the algorithms you have tried before.

{format_prompt}
```

**Key Features**:
- **Initialization mode**: Minimal prompt with only task and format
- **Evolution mode**: Includes population history + structural reference
- Shows best algorithm as **structural reference only**
- Explicitly states to use a **DIFFERENT strategy**
- Reduces structural errors (missing `__call__`, wrong return types, etc.)

**Example Rendered Prompt (Initialization)**:
```
You are a highly skilled computer scientist in the field of natural computing.
Your task is to design novel metaheuristic algorithms to solve black box optimization problems.

[Task description from problem...]
[Example code from problem...]

[Format instructions from problem...]
```

**Example Rendered Prompt (During Evolution)**:
```
You are a highly skilled computer scientist in the field of natural computing.
Your task is to design novel metaheuristic algorithms to solve black box optimization problems.

[Task description from problem...]
[Example code from problem...]

List of previously generated algorithm names with mean AOCC score:
- AdaptiveDE_v2: 0.8234
- ParticleSwarm_Elite: 0.7891
- GeneticAlgorithm_Simple: 0.7456

For reference, here is a working algorithm with correct structure and format:
```python
class AdaptiveDE_v2:
    def __init__(self, budget):
        self.budget = budget
        # ... complete implementation ...
```
Your new algorithm must use a DIFFERENT strategy from the above. Use it only 
as a reference for correct code structure and formatting.

Generate a new algorithm that is different from the algorithms you have tried before.

[Format instructions from problem...]
```

**Design Rationale**:
- **Structural Reference**: Reduces ~40% failure rate from structural errors
- **Explicit Differentiation**: "DIFFERENT strategy" prevents duplicates
- **Population Awareness**: History shows what to avoid
- **Clean Slate**: Maintains exploration while reducing technical failures

**Why Include Reference Code?**:
Without a structural reference, random_new has a high failure rate due to:
- Missing or incorrect `__call__` method signature
- Wrong return types (returning float instead of numpy array)
- Array shape mismatches
- Budget tracking errors

The reference provides a "template" for correct structure while allowing strategic novelty.

---

## Prompt Components

### Population History Format
```python
def _get_population_history(self, population: List[Any]) -> str:
    if not population:
        return ""
    
    sorted_pop = sorted(population, key=lambda s: s.fitness, reverse=True)
    
    history = "List of previously generated algorithm names with mean AOCC score:\n"
    for sol in sorted_pop:
        name = sol.name if sol.name else "Unknown"
        fitness = sol.fitness if sol.fitness is not None else 0.0
        history += f"- {name}: {fitness:.4f}\n"
    return history
```

**Purpose**:
- Shows LLM what has been tried before
- Sorted by fitness (best first) for easy reference
- Helps avoid duplicate algorithms
- Provides performance context

### Task Prompt Assembly
```python
def _get_task_prompt(self, problem: ProblemProtocol) -> str:
    task_prompt = getattr(problem, 'task_prompt', '')
    example_prompt = getattr(problem, 'example_prompt', '')
    
    role_prompt = "You are a highly skilled computer scientist in the field of natural computing. Your task is to design novel metaheuristic algorithms to solve black box optimization problems."
    
    return f"""{role_prompt}
{task_prompt}
{example_prompt}"""
```

**Purpose**:
- Standardized prompt prefix across all operators
- Uses BLADE's built-in prompts (same as LLAMEA)
- Ensures consistency with baseline methods

---

## Design Philosophy

### 1. **Consistency with LLAMEA**
All prompts use BLADE's standard prompt components (task_prompt, example_prompt, format_prompt) to ensure fair comparison with LLAMEA baseline.

### 2. **Operator-Specific Instructions**
Each operator has a unique instruction that defines its behavior:
- **Simplify**: "Refine and simplify the selected algorithm to improve it."
- **Crossover**: "Create an improved algorithm by redesigning... Draw inspiration from..."
- **Random New**: "Generate a new algorithm that is different from..."

### 3. **Preventing Common Failures**

**Code Quality Issues**:
- **Simplify**: Reduces bloat and complexity → fewer bugs
- **Crossover**: "Write from scratch" → prevents concatenation errors
- **Random New**: Structural reference → reduces technical failures

**Diversity Issues**:
- Population history shown in all prompts → avoids duplicates
- Explicit "different from" instruction → encourages novelty
- Crossover shows only name/description of parent2 → forces creative interpretation

### 4. **Adaptive Operator Selection**
Prompts are selected by a Discounted Thompson Sampling bandit that learns:
- Which operator works best in current context
- When to simplify (reduce errors)
- When to crossover (combine good ideas)
- When to explore (random new)

The reward signal for the bandit is computed as:
```python
def calculate_reward(baseline_fitness, child_fitness, is_valid):
    if not is_valid:
        return -0.5  # Penalty for structural failures
    
    improvement = child_fitness - baseline_fitness
    if improvement > 0.01:
        return 1.0  # Significant improvement
    elif improvement > 0:
        return 0.5  # Marginal improvement
    else:
        return 0.0  # No improvement
```

This creates a ternary reward structure:
- **-0.5**: Failure (syntax error, timeout, wrong structure)
- **0.0**: Valid but no improvement
- **0.5**: Marginal improvement (> 0 but ≤ 0.01)
- **1.0**: Significant improvement (> 0.01)

### 5. **Concept Transfer, Not Code Merging**

Traditional crossover shows both parents' full code:
```
Parent 1 code:
[full implementation]

Parent 2 code:
[full implementation]

Combine these algorithms.
```
❌ **Problem**: LLM copy-pastes code snippets → Frankenstein code → crashes

**GA-LLAMEA's Guided Concept Transfer**:
```
Working Algorithm:
[full implementation]

Alternative approach: "AlgorithmName"
Strategy: [description only]

Redesign the working algorithm using the concept from the alternative.
Write from scratch.
```
✅ **Solution**: LLM sees one code structure + abstract concept → coherent redesign

---

## Prompt Usage Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    GA-LLAMEA Main Loop                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │ D-TS Bandit      │
                    │ Selects Operator │
                    └──────────────────┘
                              │
            ┌─────────────────┼─────────────────┐
            │                 │                 │
            ▼                 ▼                 ▼
    ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
    │  Simplify    │  │  Crossover   │  │  Random New  │
    │  Operator    │  │  Operator    │  │  Operator    │
    └──────────────┘  └──────────────┘  └──────────────┘
            │                 │                 │
            │                 │                 │
            └─────────────────┼─────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  build_prompt()  │
                    │                  │
                    │  Components:     │
                    │  - Role          │
                    │  - Task          │
                    │  - Example       │
                    │  - History       │
                    │  - Parent(s)     │
                    │  - Instruction   │
                    │  - Format        │
                    └──────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  LLM.query()     │
                    └──────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  Extract Code    │
                    │  Validate        │
                    │  Evaluate        │
                    └──────────────────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │  Compute Reward  │
                    │  Update Bandit   │
                    └──────────────────┘
```

---

## Example: Full Evolution Cycle

### Generation 0 (Initialization)
**Operator**: random_new (init mode)
**Prompt**: Role + Task + Example + Format
**Result**: 4 initial algorithms (population size μ=4)

### Generation 1
**Bandit Selection**: `simplify` (arm 0)
**Parent**: AdaptiveDE_v2 (fitness: 0.8234)
**Prompt**:
```
You are a highly skilled computer scientist...
[task description]
[example]

List of previously generated algorithm names:
- AdaptiveDE_v2: 0.8234
- ...

Selected algorithm to simplify and improve:
Name: AdaptiveDE_v2
Fitness: 0.8234
Code:
[full code]

Refine and simplify the selected algorithm to improve it.

[format instructions]
```
**Result**: AdaptiveDE_v3 (fitness: 0.8456) → Reward: 1.0 (improvement)

### Generation 2
**Bandit Selection**: `crossover` (arm 1)
**Parents**: 
- AdaptiveDE_v3 (fitness: 0.8456) [working algorithm]
- ParticleSwarm_Elite (fitness: 0.7891) [inspiration]

**Prompt**:
```
You are a highly skilled computer scientist...
[task description]
[example]

List of previously generated algorithm names:
- AdaptiveDE_v3: 0.8456
- AdaptiveDE_v2: 0.8234
- ...

Working Algorithm (fitness: 0.8456):
[AdaptiveDE_v3 full code]

Alternative approach: "ParticleSwarm_Elite" (fitness: 0.7891)
Strategy: Uses swarm intelligence with elite-based velocity updates

Create an improved algorithm by redesigning the working algorithm...
Draw inspiration from "ParticleSwarm_Elite"...
Write from scratch.

[format instructions]
```
**Result**: HybridDE_Swarm (fitness: 0.8678) → Reward: 1.0 (improvement)

### Generation 3
**Bandit Selection**: `random_new` (arm 2)
**Prompt**:
```
You are a highly skilled computer scientist...
[task description]
[example]

List of previously generated algorithm names:
- HybridDE_Swarm: 0.8678
- AdaptiveDE_v3: 0.8456
- ...

For reference, here is a working algorithm:
[HybridDE_Swarm code]
Use a DIFFERENT strategy. Reference is for structure only.

Generate a new algorithm different from what you've tried.

[format instructions]
```
**Result**: SimulatedAnnealing_Advanced (fitness: 0.7234) → Reward: 0.0 (no improvement over median)

---

## Customization Guide

### Adding a New Operator

```python
from ga_llamea_modular.operators import BaseOperator

class MyCustomOperator(BaseOperator):
    @property
    def name(self) -> str:
        return "my_operator"
    
    def build_prompt(self, problem, population, parent=None, parent2=None, **kwargs):
        task_prompt = self._get_task_prompt(problem)
        history = self._get_population_history(population)
        
        # Your custom prompt logic
        instruction = "Your custom instruction here..."
        
        return f"{task_prompt}\n\n{history}\n\n{instruction}\n\n{problem.format_prompt}"
```

Then register it in the bandit:
```python
self.bandit = DiscountedThompsonSampler(
    arm_names=["simplify", "crossover", "random_new", "my_operator"],
    ...
)
```

### Modifying Existing Prompts

Subclass the operator and override `build_prompt()`:

```python
from ga_llamea_modular.operators import SimplifyOperator

class MySimplifyOperator(SimplifyOperator):
    def build_prompt(self, problem, population, parent=None, **kwargs):
        # Use base components
        task_prompt = self._get_task_prompt(problem)
        history = self._get_population_history(population)
        
        # Custom instruction
        instruction = "Make the algorithm faster and more memory-efficient."
        
        algo_details = f"""
Selected algorithm:
{parent.name} (fitness: {parent.fitness:.4f})
```python
{parent.code}
```
"""
        return f"{task_prompt}\n\n{history}\n{algo_details}\n{instruction}\n\n{problem.format_prompt}"
```

---

## References

- **LLAMEA Paper**: Original prompts from LLAMEA (especially Prompt5 for simplify)
- **EoH Paper**: Inspiration for semantic crossover concept
- **D-TS Paper**: Discounted Thompson Sampling for non-stationary bandits
- **BLADE Framework**: Problem definitions and prompt standards

---

## Summary

GA-LLAMEA's prompt design achieves:

✅ **High Code Quality**: Explicit instructions reduce structural errors  
✅ **Effective Hybridization**: Concept transfer avoids Frankenstein code  
✅ **Strategic Diversity**: Population history + "different" instruction prevent duplicates  
✅ **Adaptive Evolution**: D-TS bandit learns which prompts work when  
✅ **LLAMEA Compatibility**: Uses same base prompts as LLAMEA for fair comparison  

The three operators work together:
- **Simplify**: Cleans up complex solutions → reduces bugs
- **Crossover**: Transfers concepts between good solutions → combines strengths
- **Random New**: Explores novel approaches → escapes local optima

The bandit adaptively balances these based on observed rewards, creating a self-tuning evolutionary system.
