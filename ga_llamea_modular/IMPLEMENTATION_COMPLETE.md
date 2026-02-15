# Implementation Complete: refine_weakness + random_new Improvements

## Status: ✅ COMPLETED AND TESTED

All changes from `PLAN_refine_weakness_and_random_new.md` have been successfully implemented and tested.

---

## Changes Made

### 1. **New `WeaknessRefinementOperator` (operators.py)**

**Location**: Lines 271-367

**What it does**:
- Takes a parent algorithm and extracts per-instance AOCC scores from `parent.metadata["aucs"]`
- Sorts instances by performance (worst to best)
- Labels bottom quartile as "WEAK" and top quartile as "STRONG"
- Shows the LLM the parent's code alongside this performance breakdown
- Asks the LLM to analyze what optimization landscapes cause failures
- Instructs redesign for robustness across all instances

**Baseline for reward**: Parent fitness (same as simplify)

**Example prompt output**:
```
Algorithm to improve (mean AOCC: 0.7500):
```python
class TestAlgo:
    def __call__(self, func):
        pass
```

Per-instance performance (AOCC scores, sorted worst to best):
- Instance 0: 0.4500 (WEAK)
- Instance 1: 0.5000 (WEAK)
...
- Instance 19: 0.9800 (STRONG)

This algorithm performs well on average but poorly on some problem instances.
Analyze what types of optimization landscapes or function properties might
cause these failures. Redesign the algorithm to be more robust across all
instances while maintaining its strengths on the ones it already handles well.
```

---

### 2. **Modified `RandomNewOperator` (operators.py)**

**Location**: Lines 369-455

**Previous behavior**: Showed the best solution's full code as "structural reference"
- **Problem**: This biased the LLM to produce similar algorithms (always DE variants)

**New behavior**: Shows minimal structural skeleton
- Template class with correct `__init__` and `__call__` signatures
- Shows how to access bounds (`func.bounds.lb`, `func.bounds.ub`)
- Shows evaluation pattern (`func(x)`, track `eval_count`)
- Explicit instruction: "Use a DIFFERENT strategy from the algorithms listed above"

**Key change** (lines 395-423):
```python
reference = """
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
Use a DIFFERENT strategy from the algorithms listed above. This template is only for correct structure and formatting.
"""
```

**Result**: Provides structural correctness without algorithmic bias

---

### 3. **Updated OPERATORS Registry (operators.py)**

**Location**: Lines 457-477

Added `refine_weakness` to the factory:
```python
OPERATORS = {
    "simplify": SimplifyOperator,
    "crossover": CrossoverOperator,
    "random_new": RandomNewOperator,
    "refine_weakness": WeaknessRefinementOperator,  # NEW
}
```

---

### 4. **Core Integration (core.py)**

#### 4.1 Import (line 64)
```python
from .operators import SimplifyOperator, CrossoverOperator, RandomNewOperator, WeaknessRefinementOperator
```

#### 4.2 Bandit Arms (line 190)
```python
self.bandit = DiscountedThompsonSampler(
    arm_names=["simplify", "crossover", "random_new", "refine_weakness"],
    ...
)
```

#### 4.3 Operator Instantiation (line 201)
```python
self._refine_weakness = WeaknessRefinementOperator()
```

#### 4.4 Routing Logic (lines 347-353)
```python
elif operator_name == "refine_weakness":
    parent = self._select_parent()
    prompt = self._refine_weakness.build_prompt(problem, self.population, parent)
    parent_ids = [parent.id]
    baseline_fitness = parent.fitness
```

---

### 5. **Package Exports (__init__.py)**

Updated to export the new operator:
```python
from .operators import SimplifyOperator, CrossoverOperator, RandomNewOperator, WeaknessRefinementOperator

__all__ = [
    ...
    "WeaknessRefinementOperator",
    ...
]
```

---

## Architecture After Changes

```
┌─────────────────────────────────────┐
│   Discounted Thompson Sampler       │
│   (4 arms)                          │
└──────────────┬──────────────────────┘
               │
       ┌───────┴───────┬───────┬───────────────┐
       │               │       │               │
   ┌───▼───┐   ┌──────▼──┐  ┌─▼──────┐  ┌────▼────────────┐
   │simplify│   │crossover│  │random  │  │refine_weakness  │
   │        │   │         │  │_new    │  │    (NEW)        │
   └───┬────┘   └────┬────┘  └─┬──────┘  └────┬────────────┘
       │             │          │              │
       │   ┌─────────┴──────────┴──────────────┘
       │   │
       ▼   ▼
    ┌────────┐
    │  LLM   │
    │ Query  │
    └───┬────┘
        │
        ▼
   ┌─────────┐
   │Evaluate │
   │+ Reward │
   └─────────┘
```

---

## Testing Results

### Unit Tests ✅
- All Python files compile without syntax errors
- Operators module imports successfully
- Core module imports successfully
- All 4 operators registered and accessible

### Integration Tests ✅
- WeaknessRefinementOperator instantiates correctly
- Builds prompts with WEAK/STRONG labels
- Per-instance AOCC breakdown working
- RandomNewOperator uses minimal skeleton
- Does NOT leak parent algorithm code
- GA_LLaMEA instantiates with 4 bandit arms
- Bandit can select all 4 arms
- All routing logic functional

### Sample Bandit Selection (100 trials)
```
- simplify: 30 selections
- crossover: 21 selections
- random_new: 27 selections
- refine_weakness: 22 selections
```
✅ All arms are being selected (no configuration errors)

---

## Expected Impact

### Problem Addressed
**Before**: Every top solution across all runs was a DE variant because:
1. `random_new` showed best solution's full code → LLM copied the strategy
2. `simplify` and `crossover` only saw aggregate fitness → couldn't reason about failures

### Solution
1. **`refine_weakness`**: Gives LLM diagnostic information (which instances fail)
2. **`random_new` skeleton**: Provides structure without algorithmic bias

### Predicted Outcome
- **More diversity**: Algorithms beyond DE variants
- **Better robustness**: Algorithms optimized for weak instances
- **Informed evolution**: LLM can reason about *why* algorithms fail

---

## Files Modified

1. ✅ `ga_llamea_modular/operators.py` (3 changes)
   - Added `WeaknessRefinementOperator` class
   - Modified `RandomNewOperator.build_prompt()`
   - Updated `OPERATORS` dict and `get_operator()`

2. ✅ `ga_llamea_modular/core.py` (4 changes)
   - Imported `WeaknessRefinementOperator`
   - Added to bandit `arm_names`
   - Instantiated operator
   - Added routing logic in `_generate_offspring`

3. ✅ `ga_llamea_modular/__init__.py` (2 changes)
   - Imported `WeaknessRefinementOperator`
   - Added to `__all__`

---

## Ready for Experiments

The implementation is **complete and tested**. The existing experiment scripts will work without modification:

```python
# run-comparison-prompts.py will work as-is
GA_LLaMEA_Baseline = GA_LLaMEA_Method(
    llm, 
    budget=budget, 
    name="GA-LLAMEA-Improved", 
    n_parents=4, 
    n_offspring=8, 
    elitism=True, 
    discount=0.9, 
    tau_max=0.1
)
```

The bandit will now adaptively select between 4 operators instead of 3.

---

## No Errors, No Breaking Changes

✅ All syntax validated
✅ All imports working
✅ Backward compatible (existing code still works)
✅ BLADE integration intact
✅ No modifications needed to experiment scripts

**Status**: Ready for deployment
