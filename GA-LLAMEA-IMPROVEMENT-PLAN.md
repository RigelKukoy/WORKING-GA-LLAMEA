# GA-LLAMEA Improvement Plan

## Experiment Context

| Parameter | Testing | Full Experiment |
|-----------|---------|-----------------|
| **Budget (LLM calls)** | 50 | 100 |
| **Initialization** | 4 calls | 4 calls |
| **Remaining for evolution** | 46 calls | 96 calls |
| **Generations (n_offspring=8)** | ~5-6 | ~12 |
| **Bandit decisions** | ~46 | ~96 |
| **Expected failures (30-40%)** | ~14-18 wasted | ~29-38 wasted |
| **Valid bandit observations** | ~28-32 | ~58-67 |
| **Per-arm observations (3 arms)** | ~9-11 | ~19-22 |

**Baseline comparison (same budget=100, same LLM, same problem):**

| Method | Run 0 | Run 1 | Run 2 | Run 3 | Run 4 | Average |
|--------|-------|-------|-------|-------|-------|---------|
| **LLaMEA-Prompt5** | 0.768 | 0.767 | 0.803 | 0.810 | 0.763 | **~0.782** |
| **EoH** | 0.700 | 0.814 | 0.752 | 0.769 | — | ~0.759 |
| **GA-LLAMEA** | 0.750 | 0.776 | — | — | — | ~0.763 |

---

## Root Cause Summary

### Issue 1: Missing "Refine" Operator

GA-LLAMEA has three operators: `simplify`, `crossover`, `random_new`.  
LLAMEA-Prompt5 has three mutation prompts: **"Refine"**, "New", "Simplify".

The critical difference: GA-LLAMEA is **missing the "Refine" prompt** — LLAMEA's
most effective mutation type. "Refine the strategy of the selected solution to
improve it" asks the LLM to incrementally enhance a working algorithm: tune
parameters, add mechanisms, improve strategies — without simplifying it. This is
the main driver of iterative progress in LLAMEA.

GA-LLAMEA replaced this with `crossover`, which has near-zero effectiveness
(see Issue 2).

### Issue 2: Crossover Does Not Work for LLM-Generated Code

Bandit statistics from completed runs prove crossover is non-functional:

| Run | Operator | Pulls | Mean Reward | Effective |
|-----|----------|-------|-------------|-----------|
| Run 0 (new exp) | crossover | 10 | 0.000018 | No |
| Run 1 (new exp) | crossover | 18 | 0.00017 | No |
| Run 0 (old exp) | crossover | 40 | 0.001 | No |

LLMs struggle to meaningfully merge two complex code implementations.
The typical failure modes are:
- Frankenstein code that combines incompatible mechanisms
- Over-complicated merges that time out (>300s)
- Subtle bugs from mismatched variable names or data structures
- The LLM ignores the "pick best elements" instruction and concatenates everything

### Issue 3: Budget Waste from Failures

Failure rates across all GA-LLAMEA runs:

| Run | Total Evals | Failed (-inf) | Failure Rate | Valid Budget |
|-----|-------------|---------------|--------------|--------------|
| Run 0 (new) | 100 | 42 | **42%** | 58 |
| Run 1 (new) | 100 | 31 | **31%** | 69 |
| Run 2 (new) | 81 | 32 | **40%** | 49 |

Each failure burns one LLM call without providing any useful fitness signal.
The `random_new` operator is the primary source of failures because it generates
code from scratch without a working parent as reference.

Failure types observed:
- `ValueError` (array shape mismatches) — LLM generates dimensionally incorrect numpy code
- `TypeError` (None operations) — uninitialized variables
- `NameError` (undefined imports like `scipy.optimize.minimize`) — forbidden libraries
- Timeout (>300s) — algorithms with O(n²) or O(n³) complexity per iteration
- "Evaluation failed without an exception" — silent failures

### Issue 4: D-TS Bandit Cannot Learn Effectively

The Discounted Thompson Sampling bandit has **four compounding problems** that
prevent it from learning which operator is best within the available budget.

---

## Detailed D-TS Bandit Analysis

### Current Configuration

From `run-comparison-prompts.py` (line 63-64) and `core.py` (line 118-120):

```python
GA_LLaMEA_Baseline = GA_LLaMEA_Method(
    discount=0.9,     # 10% decay per observation
    tau_max=0.2       # Max posterior std dev
)

# In core.py defaults:
prior_variance=1.0    # Prior uncertainty
reward_variance=1.0   # Expected reward variance
```

### Problem A: Discount Factor Destroys Signal (discount=0.9)

**How discounting works:** Every time ANY arm is updated, ALL arms' statistics
are multiplied by γ=0.9. After N total updates, an observation made at time t
retains weight γ^(N-t).

**Impact at budget=100:** With ~96 bandit updates (after init), the first
observation retains 0.9^96 ≈ 0.000037 of its original weight — effectively zero.
The "effective window" of the bandit is approximately 1/(1-γ) = 10 observations.

**Effective sample size per arm:** The logged `discounted_count` values confirm
this. From the experiment logs:

| Run | Arm | Total Pulls | Discounted Count | Retention |
|-----|-----|-------------|------------------|-----------|
| Run 0 | simplify | 35 | 4.36 | 12% |
| Run 0 | crossover | 10 | 2.04 | 20% |
| Run 0 | random_new | 51 | 3.60 | 7% |

The bandit is making decisions based on an effective sample of **2-4 observations
per arm**, not the 10-51 it actually collected. This is far too few for reliable
Bayesian inference.

**Why this hurts:** With only 2-4 effective samples, the posterior distributions
are wide and overlapping. Thompson Sampling degenerates to near-random selection.
The bandit cannot reliably distinguish a 0.001 mean reward (simplify) from a
0.00002 mean reward (crossover).

**Why discounting is inappropriate here:** Discounting is designed for
**non-stationary** environments where the reward distribution changes over time
(e.g., a previously good arm becomes bad). In GA-LLAMEA, operator quality is
**relatively stable** — crossover is always poor, simplify is always decent.
The non-stationarity assumption is wrong, and the discounting actively destroys
valuable signal.

### Problem B: Reward-Scale Mismatch (tau_max=0.2, prior_variance=1.0)

**The reward distribution:** Successful improvements yield rewards of 0.001-0.04
(from the `calculate_reward` formula). Most observations are 0.0 (failures or
no improvement). The actual difference between operator mean rewards is ~0.001-0.003.

**The posterior uncertainty:** With `prior_variance=1.0` and `tau_max=0.2`, the
posterior standard deviation ranges from 0.001 (after many observations) to 0.2
(with few observations). In early generations, the posterior std is **50-200x
larger** than the signal it's trying to detect.

**What this means:** When sampling θ ~ N(μ=0.001, σ=0.2), the sampled values
range from roughly -0.4 to +0.4. The actual mean difference between arms
(~0.001) is invisible at this noise level. The bandit is essentially selecting
randomly.

**Interaction with discounting:** Because discounting keeps the effective sample
size small (2-4), the posterior never narrows enough to distinguish arms. Even
after 100 evaluations, the posterior std remains at ~0.001-0.008 (from the logs),
which is still comparable to the signal magnitude.

### Problem C: Reward Function Produces Vanishing Gradients

The current reward formula (`utils.py` line 22-60):

```python
def calculate_reward(parent_score, child_score, is_valid):
    if not is_valid:
        return 0.0
    if child_score <= parent_score:
        return 0.0
    if parent_score <= 0:
        return min(1.0, child_score)
    return min(1.0, (child_score - parent_score) / max(0.01, parent_score))
```

**Problem 1: Vanishing rewards as population improves.**
When parent fitness is 0.75, a child at 0.78 (excellent improvement!) yields:
reward = (0.78 - 0.75) / 0.75 = **0.04**. A child at 0.76 yields only **0.013**.
These tiny values are indistinguishable from noise in the posterior.

**Problem 2: Failures and no-improvement are indistinguishable.**
Both return 0.0. The bandit cannot learn that `random_new` fails 40% of the time
(producing broken code) vs. `simplify` which fails 20% of the time but reliably
produces valid code. Both look equally "unrewarding."

**Problem 3: The crossover baseline is too high.**
Crossover baseline = `max(parent1.fitness, parent2.fitness)`. The child must beat
the BETTER parent. When both parents are already good (0.7+), this is extremely
difficult. Crossover almost never earns any reward.

### Problem D: No Exploration Guarantee

The bandit starts with zero data and immediately begins making adaptive
selections. Early random selections can create self-reinforcing bias:

1. `random_new` gets lucky on the first try → bandit favors it
2. `random_new` fails the next 5 times → but discount has already reduced
   the failure weight
3. Bandit keeps selecting `random_new` despite poor overall performance

With budget=100, there is no round-robin phase to ensure each arm gets a
minimum number of observations before the bandit starts exploiting.

---

## Recommended Changes

### Change 1: Add a "Refine" Operator (HIGH PRIORITY)

**What:** Add a fourth operator that matches LLAMEA's Prompt 1: "Refine the
strategy of the selected solution to improve it." This is the missing piece
that makes LLAMEA-Prompt5 outperform GA-LLAMEA.

**Why:** The "Refine" prompt asks the LLM to incrementally enhance a working
algorithm without simplifying it. This allows adding mechanisms, tuning
parameters, and improving strategies — the primary driver of iterative
improvement in LLAMEA.

**Implementation:** Create a `RefineOperator` class in `operators.py`:

```python
class RefineOperator(BaseOperator):
    """Refine operator: Improve a parent algorithm's strategy.
    
    Unlike Simplify (which reduces complexity), Refine encourages the LLM
    to enhance, tune, and add mechanisms to improve performance.
    Matches LLAMEA's Prompt 1 which is the most effective mutation type.
    """
    
    @property
    def name(self) -> str:
        return "refine"
    
    def build_prompt(
        self,
        problem: ProblemProtocol,
        population: List[Any],
        parent: Any = None,
        parent2: Any = None,
        **kwargs
    ) -> str:
        if parent is None:
            raise ValueError("Refine requires a parent solution")
        
        task_prompt = self._get_task_prompt(problem)
        history = self._get_population_history(population)
        
        algo_details = f"""
Selected algorithm to refine and improve:
Name: {parent.name}
Fitness: {parent.fitness:.4f}
Code:
```python
{parent.code}
```
"""
        instruction = "Refine the strategy of the selected solution to improve it."
        
        return (
            f"{task_prompt}\n\n{history}\n{algo_details}\n\n"
            f"{instruction}\n\n{problem.format_prompt}"
        )
```

**Changes needed in `core.py`:**

```python
from .operators import SimplifyOperator, CrossoverOperator, RandomNewOperator, RefineOperator

# In __init__:
self._refine = RefineOperator()

self.bandit = DiscountedThompsonSampler(
    arm_names=["refine", "simplify", "crossover", "random_new"],
    # ... other params
)

# In _generate_offspring, add the refine branch:
if operator_name == "refine":
    parent = self._select_parent()
    prompt = self._refine.build_prompt(problem, self.population, parent)
    parent_ids = [parent.id]
    baseline_fitness = parent.fitness
elif operator_name == "simplify":
    # ... existing code
```

**Impact at budget=100:** Adds the single most effective mutation type. With 4
arms and 96 decisions, each arm gets ~24 observations — still learnable.

**Impact at budget=50:** With 46 decisions and 4 arms, each arm gets ~11-12
observations. This is tight but workable, especially with the other bandit
fixes below.

### Change 2: Disable Discounting (HIGH PRIORITY)

**What:** Set `discount=1.0` (no forgetting) for both budget=50 and budget=100.

**Why:** Operator quality is stable in this problem. Crossover never works well,
simplify is consistently decent. Discounting is designed for non-stationary
environments where reward distributions shift over time — this is NOT the case
here. Keeping all observations gives the bandit 4-10x more effective data.

**Implementation in `run-comparison-prompts.py`:**

```python
GA_LLaMEA_Baseline = GA_LLaMEA_Method(
    llm, 
    budget=budget, 
    name="GA-LLAMEA-Improved",
    n_parents=4, 
    n_offspring=8, 
    elitism=True, 
    discount=1.0,     # Changed from 0.9 — no forgetting
    tau_max=0.2       # Keep or reduce (see Change 3)
)
```

**Impact:** Effective sample size per arm increases from ~2-4 to ~19-22
(at budget=100). The bandit can now reliably distinguish arms with mean
reward differences of ~0.002.

**Budget=50 consideration:** Even more critical at budget=50 where only ~46
decisions are made. With discount=0.9, effective counts would be ~1-2 per arm
(useless). With discount=1.0, effective counts are ~11-12 per arm (workable).

### Change 3: Fix Reward Scale (HIGH PRIORITY)

**What:** Reduce `tau_max` from 0.2 to 0.05, and `prior_variance` from 1.0 to
0.01. This makes the bandit's uncertainty proportional to actual reward magnitudes.

**Why:** Rewards are in the range [0, 0.04] with most values at 0.0. A posterior
standard deviation of 0.2 is 5-200x larger than the signal. The bandit cannot
detect operator quality differences until the posterior narrows enough, which
takes too many samples at the current scale.

**Implementation in `run-comparison-prompts.py`:**

```python
GA_LLaMEA_Baseline = GA_LLaMEA_Method(
    llm, 
    budget=budget, 
    name="GA-LLAMEA-Improved",
    n_parents=4, 
    n_offspring=8, 
    elitism=True, 
    discount=1.0,
    tau_max=0.05,     # Changed from 0.2 — matches reward scale
)
```

**Also requires changing `prior_variance` in `core.py`** (line 176-181):

```python
self.bandit = DiscountedThompsonSampler(
    arm_names=["refine", "simplify", "crossover", "random_new"],
    discount=discount,
    tau_max=tau_max,
    prior_variance=0.01,      # Changed from 1.0
    reward_variance=0.01,     # Changed from 1.0
)
```

**Impact:** The posterior std now starts at ~0.1 and narrows to ~0.01-0.005
after 10-20 observations. At this scale, the bandit can distinguish a
0.003 mean reward (simplify) from 0.00002 (crossover) within ~15-20
observations — well within the budget=100 window.

### Change 4: Switch to Binary Reward (HIGH PRIORITY)

**What:** Replace the continuous reward function with a binary signal:
- `+1.0` if child fitness > parent fitness (improvement)
- `0.0` if child fitness <= parent fitness but valid (no improvement)
- `-0.5` if child is invalid (failure penalty)

**Why:** The current continuous reward produces values so small (0.001-0.04)
that they're indistinguishable from noise. Binary rewards give the bandit a
clean, robust signal: "what fraction of the time does each operator produce
improvements?" This is much easier to learn from 20-30 samples.

The failure penalty (-0.5) is critical: it lets the bandit learn that
`random_new` produces broken code 40% of the time, while `simplify` fails
only 20% of the time. Currently both get 0.0, making them indistinguishable.

**Implementation in `utils.py`:**

```python
def calculate_reward(parent_score: float, child_score: float, is_valid: bool) -> float:
    """Calculate reward for bandit update.
    
    Uses binary reward with failure penalty for robust learning
    at small sample sizes (budget=50-100).
    
    Returns:
        +1.0  if improvement over parent
         0.0  if valid but no improvement
        -0.5  if invalid (error, timeout, etc.)
    """
    if not is_valid:
        return -0.5   # Penalize budget-wasting failures
    if child_score > parent_score:
        return 1.0    # Binary: any improvement counts equally
    return 0.0        # Valid but no improvement
```

**Also update `core.py`** to pass `is_valid=False` for failures (lines 355, 364, 387, 393):

```python
# For code extraction failures:
self.bandit.update(operator_name, calculate_reward(0, 0, is_valid=False))

# For validation failures:
self.bandit.update(operator_name, calculate_reward(0, 0, is_valid=False))

# For evaluation failures:
self.bandit.update(operator_name, calculate_reward(0, 0, is_valid=False))
```

**Corresponding scale fix:** With binary rewards in {-0.5, 0.0, 1.0}, update
the bandit parameters:

```python
self.bandit = DiscountedThompsonSampler(
    arm_names=["refine", "simplify", "crossover", "random_new"],
    discount=1.0,
    tau_max=0.5,              # Appropriate for [-0.5, 1.0] reward range
    prior_variance=0.25,      # Moderate prior uncertainty
    reward_variance=0.5,      # Expected variance of binary-ish rewards
)
```

**Impact at budget=100:** With binary rewards, the bandit sees clear signals:
- `refine`: ~40% improvement rate → mean reward ≈ 0.4
- `simplify`: ~25% improvement rate → mean reward ≈ 0.25
- `crossover`: ~5% improvement rate, 30% failure → mean reward ≈ -0.1
- `random_new`: ~10% improvement rate, 40% failure → mean reward ≈ -0.1

These differences (0.4 vs -0.1) are large enough to learn within 15-20
observations per arm. The bandit will quickly favor `refine` and `simplify`.

**Impact at budget=50:** Even with ~11 observations per arm, binary rewards
produce distinguishable signals. The bandit can learn within 2-3 generations.

### Change 5: Add Round-Robin Warm-Up (MEDIUM PRIORITY)

**What:** Force each operator to be used at least N times before the bandit
starts making adaptive selections.

**Why:** Without warm-up, early lucky/unlucky results create bias that persists.
If `random_new` gets lucky on its first call, the bandit may over-select it
for many subsequent turns before accumulating enough failures to correct.

**Implementation in `core.py`, in `_generate_offspring`:**

```python
# At the top of the method, before the offspring loop:
WARMUP_PER_ARM = 3  # Each operator gets 3 forced tries

def _generate_offspring(self, problem: ProblemProtocol) -> List[Any]:
    offspring = []
    
    for _ in range(self.n_offspring):
        if self.llm_calls >= self.budget:
            break

        # Warm-up phase: round-robin through operators
        total_pulls = sum(a.pulls for a in self.bandit.arms.values())
        warmup_needed = len(self.bandit.arm_names) * 3  # 3 per arm = 12 total
        
        if total_pulls < warmup_needed:
            # Force round-robin
            operator_name = self.bandit.arm_names[total_pulls % len(self.bandit.arm_names)]
            theta = 0.0  # No sampling during warm-up
        else:
            # Use bandit selection
            operator_name, theta = self.bandit.select_arm()

        # ... rest of offspring generation unchanged
```

**Impact at budget=100:** Uses 12 of 96 evolutionary calls (12.5%) for
warm-up. Each arm gets 3 guaranteed observations. Remaining 84 calls use
informed bandit selection.

**Impact at budget=50:** Uses 12 of 46 evolutionary calls (26%) for warm-up.
This is a larger fraction, but still leaves 34 calls for adaptive selection.
Consider reducing to `WARMUP_PER_ARM = 2` for budget=50 (8 warm-up calls).

### Change 6: Reduce or Remove Crossover (MEDIUM PRIORITY)

**Option A: Remove crossover entirely.** Reduce to 3 arms: `refine`,
`simplify`, `random_new` — matching LLAMEA-Prompt5's three mutation prompts
exactly.

```python
self.bandit = DiscountedThompsonSampler(
    arm_names=["refine", "simplify", "random_new"],
    # ...
)
```

**Pros:** Fewer arms = more observations per arm = faster learning. Removes
the consistently non-functional operator. Matches the proven LLAMEA-Prompt5
configuration exactly.

**Cons:** Loses the ability for crossover to contribute in principle. If
crossover's prompts were improved, it might eventually work.

**Option B: Keep crossover but cap its selection rate.** Add a maximum
selection frequency (e.g., 15% of the time) so it doesn't waste too much
budget even if the bandit over-selects it.

```python
# In _generate_offspring:
operator_name, theta = self.bandit.select_arm()

# Cap crossover to 15% of decisions
crossover_pulls = self.bandit.arms["crossover"].pulls
total_pulls = sum(a.pulls for a in self.bandit.arms.values())
if operator_name == "crossover" and total_pulls > 0:
    if crossover_pulls / total_pulls > 0.15:
        # Re-sample excluding crossover
        operator_name, theta = self._sample_excluding("crossover")
```

**Option C: Redesign crossover to work at the strategy level.** Instead of
giving the LLM two full code implementations to merge, give it two algorithm
*descriptions* and ask it to design a new algorithm inspired by both — then
generate the code from scratch. This avoids the code-merging problem.

```python
instruction = f"""You are given two algorithm strategies:
Strategy 1 ({parent.fitness:.4f}): {parent.description}
Strategy 2 ({parent2.fitness:.4f}): {parent2.description}

Design a new algorithm that combines the best ideas from both strategies.
Write the implementation from scratch (do not copy-paste code from either)."""
```

**Recommendation for budget=100:** Start with Option A (remove crossover).
This gives the most data per arm and matches the proven LLAMEA baseline.
Test Option C in a separate experiment later.

**Recommendation for budget=50:** Definitely Option A. With only ~46
decisions, 3 arms is already tight. 4 arms would give only ~11 observations
each.

### Change 7: Reduce random_new Reliance (LOW PRIORITY)

**What:** Reduce the baseline fitness threshold for `random_new` to make
it easier for random solutions to earn reward, OR reduce the warmup
allocation for `random_new`.

**Current behavior:** `random_new` baseline = population median. With
median at 0.5-0.7, a random new algorithm must beat the median to get
any reward. Combined with the 40% failure rate, this means `random_new`
almost never earns reward.

**Option:** Set `random_new` baseline to the population minimum (worst
individual) instead of the median. This gives random algorithms a
fighting chance to earn reward.

```python
# In _generate_offspring, the random_new branch:
else:  # random_new
    prompt = self._random_new.build_prompt(problem, self.population)
    parent_ids = []
    if self.population:
        # Use minimum fitness as baseline (easier for random_new to beat)
        baseline_fitness = min(s.fitness for s in self.population)
    else:
        baseline_fitness = -float('inf')
```

**Impact:** With binary rewards (Change 4), this becomes less important
because any valid-but-not-improving solution already gets 0.0. But
adjusting the baseline still helps the bandit learn the true improvement
rate of `random_new`.

---

## Recommended Experiment Configurations

### Configuration A: "Quick Fix" (Minimal Changes)

Changes only the bandit parameters. No code structure changes needed.

```python
# run-comparison-prompts.py
GA_LLaMEA_QuickFix = GA_LLaMEA_Method(
    llm, 
    budget=budget, 
    name="GA-LLAMEA-QuickFix", 
    n_parents=4, 
    n_offspring=8, 
    elitism=True, 
    discount=1.0,     # No forgetting (was 0.9)
    tau_max=0.05      # Matched to reward scale (was 0.2)
)
```

Changes: discount=1.0, tau_max=0.05  
Files modified: `run-comparison-prompts.py` only  
Expected improvement: +0.01-0.02 AOCC (better bandit decisions)

### Configuration B: "Full Fix" (All Recommended Changes)

Adds refine operator, fixes bandit, uses binary reward, removes crossover.

```python
# run-comparison-prompts.py
GA_LLaMEA_FullFix = GA_LLaMEA_Method(
    llm, 
    budget=budget, 
    name="GA-LLAMEA-FullFix", 
    n_parents=4, 
    n_offspring=8, 
    elitism=True, 
    discount=1.0,
    tau_max=0.5,
    # New parameters to add:
    # operators=["refine", "simplify", "random_new"],  # No crossover
    # reward_type="binary",
    # warmup_per_arm=3,
)
```

Changes: All 7 changes above  
Files modified: `operators.py`, `core.py`, `utils.py`, `bandit.py`, `run-comparison-prompts.py`  
Expected improvement: +0.02-0.05 AOCC (closing the gap with LLAMEA-Prompt5)

### Configuration C: "LLAMEA-Equivalent" (Ablation Baseline)

Uses GA-LLAMEA's code structure but exactly matches LLAMEA-Prompt5's
behavior. This creates a clean ablation: if it matches LLAMEA-Prompt5's
performance, the codebase is correct. If it doesn't, there's a
structural bug.

```python
# Three operators matching LLAMEA-Prompt5's three prompts exactly:
# refine   = "Refine the strategy of the selected solution to improve it."
# random_new = "Generate a new algorithm that is different..."
# simplify = "Refine and simplify the selected algorithm to improve it."
# 
# No crossover. Bandit with discount=1.0 for fair comparison.
GA_LLaMEA_Equivalent = GA_LLaMEA_Method(
    llm, 
    budget=budget, 
    name="GA-LLAMEA-Equivalent", 
    n_parents=4, 
    n_offspring=8, 
    elitism=True, 
    discount=1.0,
    tau_max=0.5,
    # operators=["refine", "simplify", "random_new"],
)
```

---

## Implementation Checklist

### Phase 1: Quick Parameter Fixes (no code changes)

- [ ] Set `discount=1.0` in `run-comparison-prompts.py`
- [ ] Set `tau_max=0.05` in `run-comparison-prompts.py`
- [ ] Run 2-3 test runs at budget=50 to verify improvement

### Phase 2: Reward Function Fix

- [ ] Modify `calculate_reward()` in `utils.py` to use binary reward
- [ ] Add `is_valid=False` path for failure cases in `core.py`
- [ ] Update `prior_variance` and `reward_variance` in `core.py` to match binary scale
- [ ] Run 2-3 test runs at budget=50 to verify the bandit learns faster

### Phase 3: Add Refine Operator

- [ ] Create `RefineOperator` class in `operators.py`
- [ ] Register it in the `OPERATORS` dict
- [ ] Add `refine` to the bandit arm names in `core.py`
- [ ] Add the `refine` branch in `_generate_offspring` in `core.py`
- [ ] Run 2-3 test runs at budget=50

### Phase 4: Remove Crossover (Optional)

- [ ] Remove `crossover` from bandit arm names
- [ ] Remove the crossover branch from `_generate_offspring`
- [ ] Run 2-3 test runs at budget=50

### Phase 5: Add Warm-Up

- [ ] Implement round-robin warm-up in `_generate_offspring`
- [ ] Set `WARMUP_PER_ARM=3` for budget=100, `WARMUP_PER_ARM=2` for budget=50
- [ ] Run 2-3 test runs at budget=50

### Phase 6: Full Experiment

- [ ] Run all configurations (QuickFix, FullFix, Equivalent) at budget=100
- [ ] Run with 5 seeds each for statistical significance
- [ ] Compare against LLaMEA-Prompt5 and EoH baselines
- [ ] Analyze bandit arm selection patterns and convergence

---

## Summary Table of All Changes

| # | Change | Parameter/File | Current | Proposed | Priority |
|---|--------|---------------|---------|----------|----------|
| 1 | Add Refine operator | `operators.py`, `core.py` | Missing | Add RefineOperator | HIGH |
| 2 | Disable discounting | `discount` in experiment | 0.9 | 1.0 | HIGH |
| 3 | Fix reward scale | `tau_max`, `prior_variance` | 0.2, 1.0 | 0.05, 0.01 (cont.) or 0.5, 0.25 (binary) | HIGH |
| 4 | Binary reward | `calculate_reward` in utils | Continuous [0,1] | Binary {-0.5, 0, 1} | HIGH |
| 5 | Round-robin warm-up | `_generate_offspring` in core | None | 3 per arm | MEDIUM |
| 6 | Remove crossover | bandit arms in core | 3 arms | 3 arms (swap crossover for refine) | MEDIUM |
| 7 | Lower random_new baseline | `_generate_offspring` in core | Population median | Population minimum | LOW |
