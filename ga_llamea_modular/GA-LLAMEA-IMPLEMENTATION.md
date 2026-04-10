# GA-LLAMEA Implementation Analysis

## Table of Contents
1. [DTS Paper Equations (Qi et al., 2023)](#1-dts-paper-equations)
2. [GA-LLAMEA DTS Implementation](#2-ga-llamea-dts-implementation)
3. [Equation-by-Equation Comparison](#3-equation-by-equation-comparison)
4. [Accuracy Rating](#4-accuracy-rating)
5. [Stagnation Detection Method](#5-stagnation-detection-method)
6. [GA-LLAMEA Prompts](#6-ga-llamea-prompts)
7. [Complete Architecture](#7-complete-architecture)
8. [Reward Formula and 4-Arm Mapping](#8-reward-formula-and-4-arm-mapping)

---

## 1. DTS Paper Equations

The paper "Discounted Thompson Sampling for Non-Stationary Bandit Problems" (Qi et al., 2023) defines DS-TS with Gaussian priors as follows:

### 1.1 Core Definitions

| Symbol | Definition |
|--------|-----------|
| $K$ | Number of arms $\mathcal{A} := \{1, 2, \dots, K\}$ |
| $T$ | Finite time horizon |
| $\gamma$ | Discount factor, $\gamma \in (1 - \frac{1}{e}, 1)$ |
| $\tau_{max}$ | Maximum sampling standard deviation |
| $X_t(i)$ | Reward of arm $i$ at time $t$, bounded in $[0, 1]$ |
| $\mu_t(i)$ | Expected reward $\mathbb{E}[X_t(i)]$ |

### 1.2 Algorithm 1: DS-TS Pseudocode

**Input:** $\gamma \in (1 - \frac{1}{e}, 1)$, $\tau_{max}$

**Initialize:** $\hat{\mu}_1(i) = 0$, $\tilde{\mu}_1(i) = 0$, $N_1(\gamma, i) = 0$, $\tau_1(i) = \tau_{max}$

**For** $t = 1, \dots, T$:

1. **Sample** (Line 5): For each arm $i = 1, \dots, K$:
$$\theta_t(i) \sim \mathcal{N}(\hat{\mu}_t(\gamma, i),\ \tau_t(i)^2)$$

2. **Select** (Line 7): Play arm $i_t = \arg\max_i \theta_t(i)$, observe reward $X_t(i_t)$

3. **Update** (Lines 9-12): For each arm $i = 1, \dots, K$:
   - Line 9 — Discounted cumulative reward:
   $$\tilde{\mu}_{t+1}(\gamma, i) = \gamma \cdot \tilde{\mu}_t(\gamma, i) + \mathbb{1}\{i = i_t\} \cdot X_t(i_t)$$
   - Line 10 — Discounted count:
   $$N_{t+1}(\gamma, i) = \gamma \cdot N_t(\gamma, i) + \mathbb{1}\{i = i_t\}$$
   - Line 11 — Posterior mean (discounted empirical average):
   $$\hat{\mu}_{t+1}(\gamma, i) = \frac{\tilde{\mu}_{t+1}(\gamma, i)}{N_{t+1}(\gamma, i)}$$
   - Line 12 — Posterior variance (capped):
   $$\tau_{t+1}(i) = \min\left\{\frac{1}{\sqrt{N_{t+1}(\gamma, i)}},\ \tau_{max}\right\}$$

### 1.3 Key Properties from the Paper

- **Discount mechanism:** Applies to ALL arms at every step. Unselected arms: $N_{t+1}(\gamma, i) = \gamma \cdot N_t(\gamma, i)$ and $\tilde{\mu}_{t+1}(\gamma, i) = \gamma \cdot \tilde{\mu}_t(\gamma, i)$. This means the posterior mean $\hat{\mu}$ stays unchanged for unselected arms but variance INCREASES (since $N$ decays, $1/\sqrt{N}$ grows).
- **Selected arm:** Gets $+1$ added to count and $+X_t$ to sum, so its variance DECREASES.
- **tau_max purpose:** Prevents variance from going infinite when $N \to 0$, keeping sampling near the mean. The paper recommends $\tau_{max} \geq \frac{1}{12\sqrt{2}}$ (Theorem 1) and empirically uses $\tau_{max} = \mu_{max}/5$.
- **gamma range:** $\gamma \in (1 - \frac{1}{e}, 1) \approx (0.632, 1)$
- **Regret bounds:**
  - Abruptly changing: $\tilde{O}(\sqrt{TB_T})$ with $\gamma = 1 - \sqrt{B_T/T}$
  - Smoothly changing: $\tilde{O}(T^\beta)$ with $\gamma = 1 - 1/T^{1-\beta}$
- **Reward support:** $[0, 1]$ — the paper explicitly assumes bounded support.

### 1.4 Regret Equation (Equation 1)

$$R_T^\pi = \mathbb{E}\left[\sum_{t=1}^{T} (\mu_t(*) - \mu_t(i_t))\right]$$

### 1.5 Theorem 1 (Abruptly Changing)

$$\mathbb{E}[k_T(i)] \le B_T D(\gamma) + (C + 2)L(\gamma) \gamma^{-1/(1-\gamma)} T (1 - \gamma) \log\left(\frac{1}{1 - \gamma}\right)$$

Where:
- $D(\gamma) = \frac{\log((1-\gamma)^2 \log(\frac{1}{1-\gamma}))}{\log \gamma}$
- $L(\gamma) = \frac{144(1+\sqrt{2})^2 \log(\frac{1}{1-\gamma} + e^{25})}{\gamma^{1/(1-\gamma)} (\Delta_T)^2}$
- $C = e^{25} + 12 + \frac{1}{F(\frac{\mu_{max}}{\tau_{max}})}$
- $F(x) = \frac{1}{\sqrt{2\pi}} \frac{x}{1+x^2} e^{-x^2/2}$

### 1.6 Theorem 2 (Smoothly Changing)

$$\mathbb{E}[k_T(i)] \le F \Delta T^\beta + M(\gamma) T (1 - \gamma) \log\left(\frac{1}{1 - \gamma}\right)$$

---

## 2. GA-LLAMEA DTS Implementation

**File:** `ga_llamea_modular/bandit.py`

### 2.1 Data Structure

```python
@dataclass
class ArmState:
    discounted_count: float = 0.0      # N_t(gamma, i)
    discounted_sum: float = 0.0        # mu~_t(gamma, i)
    discounted_sum_sq: float = 0.0     # Extra: sum of squared rewards (not in paper)
    posterior_mean: float = 0.0        # mu_hat_t(gamma, i)
    posterior_var: float = 1.0         # tau_t(i)^2
    last_theta: float = 0.0           # Last sampled theta (for logging)
    pulls: int = 0                     # Total pulls (not discounted, not in paper)
```

### 2.2 Discount Application

```python
def _apply_discount(self) -> None:
    if self.discount == 1.0:
        return
    for arm_state in self.arms.values():
        arm_state.discounted_count *= self.discount    # N *= gamma
        arm_state.discounted_sum *= self.discount      # mu~ *= gamma
        arm_state.discounted_sum_sq *= self.discount   # Extra field
```

### 2.3 Update (after observing reward)

```python
def update(self, arm_name: str, reward: float) -> None:
    self._apply_discount()                          # Discount ALL arms
    arm_state = self.arms[arm_name]
    arm_state.discounted_count += 1.0               # N += 1 (selected arm only)
    arm_state.discounted_sum += reward              # mu~ += X_t (selected arm only)
    arm_state.discounted_sum_sq += reward**2        # Extra
    arm_state.pulls += 1                            # Extra counter
    self._update_posterior(arm_state)
```

### 2.4 Posterior Update

```python
def _update_posterior(self, arm_state: ArmState) -> None:
    count = max(self.epsilon, arm_state.discounted_count)  # epsilon = 1e-6
    arm_state.posterior_mean = arm_state.discounted_sum / count        # Line 11
    arm_state.posterior_var = min(1.0 / count, self.tau_max**2)       # Line 12
```

### 2.5 Arm Selection

```python
def select_arm(self) -> Tuple[str, float]:
    # Phase 1: Burn-in (NOT in paper)
    if self.min_pulls > 0:
        undersampled = [n for n, s in self.arms.items() if s.pulls < self.min_pulls]
        if undersampled:
            return random.choice(undersampled), theta

    # Phase 2: Epsilon-greedy floor (NOT in paper)
    if random.random() < self.epsilon_exploration:  # default 0.4
        return random.choice(self.arm_names), theta

    # Phase 3: Thompson Sampling (Paper Algorithm 1, Line 5-7)
    for arm_name, arm_state in self.arms.items():
        self._update_posterior(arm_state)
        std_dev = math.sqrt(max(self.epsilon, arm_state.posterior_var))
        theta = random.gauss(arm_state.posterior_mean, std_dev)
        # Select argmax theta
```

### 2.6 Default Parameters

| Parameter | GA-LLAMEA Value | Paper Value/Recommendation |
|-----------|----------------|----------------------------|
| `discount` (gamma) | 0.99 | $1 - \sqrt{B_T/T}$ (problem-dependent) |
| `tau_max` | 0.20 | $\mu_{max}/5$ (paper uses 1/5 when $\mu_{max}=1$) |
| `epsilon_exploration` | 0.40 | N/A (not in paper) |
| `min_pulls` | 5 | N/A (not in paper) |
| `epsilon` (numerical) | 1e-6 | N/A |

---

## 3. Equation-by-Equation Comparison

### 3.1 Line 9 — Discounted Cumulative Reward $\tilde{\mu}$

**Paper:**
$$\tilde{\mu}_{t+1}(\gamma, i) = \gamma \cdot \tilde{\mu}_t(\gamma, i) + \mathbb{1}\{i = i_t\} \cdot X_t(i_t)$$

**GA-LLAMEA:**
```python
# In _apply_discount() — applied to ALL arms:
arm_state.discounted_sum *= self.discount       # gamma * mu~_t

# In update() — only for the selected arm:
arm_state.discounted_sum += reward              # + X_t
```

**Verdict: MATCH.** The two-step process (discount all, then add to selected) is mathematically identical to the paper's single-line formulation. For unselected arms, only the discount is applied ($\mathbb{1}\{i = i_t\} = 0$). For the selected arm, discount + reward addition matches exactly.

### 3.2 Line 10 — Discounted Count $N$

**Paper:**
$$N_{t+1}(\gamma, i) = \gamma \cdot N_t(\gamma, i) + \mathbb{1}\{i = i_t\}$$

**GA-LLAMEA:**
```python
arm_state.discounted_count *= self.discount     # gamma * N_t  (all arms)
arm_state.discounted_count += 1.0               # + 1          (selected arm only)
```

**Verdict: MATCH.** Identical logic split across two methods.

### 3.3 Line 11 — Posterior Mean $\hat{\mu}$

**Paper:**
$$\hat{\mu}_{t+1}(\gamma, i) = \frac{\tilde{\mu}_{t+1}(\gamma, i)}{N_{t+1}(\gamma, i)}$$

**GA-LLAMEA:**
```python
arm_state.posterior_mean = arm_state.discounted_sum / count
# where count = max(1e-6, discounted_count)
```

**Verdict: MATCH** (with numerical safeguard). The `max(epsilon, ...)` prevents division by zero when an arm has never been pulled. The paper assumes $N > 0$ implicitly (all arms are initialized with $N = 0$, and division only makes sense after at least one pull). The epsilon is negligibly small (1e-6) and does not affect the algorithm in practice.

### 3.4 Line 12 — Posterior Variance $\tau^2$

**Paper:**
$$\tau_{t+1}(i) = \min\left\{\frac{1}{\sqrt{N_{t+1}(\gamma, i)}},\ \tau_{max}\right\}$$

Note: $\tau$ is the **standard deviation**, so $\tau^2$ is the variance.

**GA-LLAMEA:**
```python
arm_state.posterior_var = min(1.0 / count, self.tau_max**2)
```

Here `posterior_var` stores the **variance** ($\tau^2$), not the standard deviation. Let's verify:

- Paper: $\tau = \min\{1/\sqrt{N}, \tau_{max}\}$, so $\tau^2 = \min\{1/N, \tau_{max}^2\}$
- Code: `posterior_var = min(1/N, tau_max^2)`

**Verdict: MATCH.** The code correctly stores $\tau^2$ (variance) by squaring both terms in the min. When sampling:
```python
std_dev = math.sqrt(max(self.epsilon, arm_state.posterior_var))  # sqrt(tau^2) = tau
theta = random.gauss(arm_state.posterior_mean, std_dev)          # N(mu_hat, tau^2)
```
This correctly recovers $\theta \sim \mathcal{N}(\hat{\mu}, \tau^2)$.

### 3.5 Line 5 — Thompson Sampling

**Paper:**
$$\theta_t(i) \sim \mathcal{N}(\hat{\mu}_t(\gamma, i),\ \tau_t(i)^2)$$

**GA-LLAMEA:**
```python
theta = random.gauss(arm_state.posterior_mean, std_dev)
# where std_dev = sqrt(posterior_var) = tau
```

**Verdict: MATCH.** `random.gauss(mu, sigma)` samples from $\mathcal{N}(\mu, \sigma^2)$, which is exactly $\mathcal{N}(\hat{\mu}, \tau^2)$.

### 3.6 Line 7 — Arm Selection

**Paper:**
$$i_t = \arg\max_i \theta_t(i)$$

**GA-LLAMEA:**
```python
if theta > best_theta:
    best_theta = theta
    best_arm = arm_name
```

**Verdict: MATCH** (when Thompson Sampling path is taken). However, GA-LLAMEA adds two non-paper selection paths before this:
1. **Burn-in phase** (min_pulls): Forces each arm to be tried at least N times
2. **Epsilon-greedy floor** (40%): Random arm selection to prevent extinction

These are **practical extensions**, not part of the original algorithm.

### 3.7 Initialization

**Paper:** $\hat{\mu}_1(i) = 0$, $\tilde{\mu}_1(i) = 0$, $N_1(\gamma, i) = 0$, $\tau_1(i) = \tau_{max}$

**GA-LLAMEA:**
```python
discounted_count: float = 0.0      # N = 0        ✓
discounted_sum: float = 0.0        # mu~ = 0      ✓
posterior_mean: float = 0.0        # mu_hat = 0   ✓
posterior_var: float = 1.0         # tau^2 = 1    ✗ (should be tau_max^2)
```

**Verdict: MINOR MISMATCH.** The initial `posterior_var = 1.0` differs from the paper's $\tau_1 = \tau_{max}$ (which gives $\tau^2 = 0.04$ for `tau_max=0.20`). However, this only matters before the first pull of each arm. After the first `_update_posterior()` call, the variance is immediately recomputed as `min(1/count, tau_max^2)`. With `count = max(1e-6, 0) = 1e-6`, the first `_update_posterior` gives `min(1e6, 0.04) = 0.04 = tau_max^2`. So after any posterior update, it self-corrects. **Practically no impact.**

### 3.8 Discount Timing

**Paper:** Updates happen AFTER selecting and observing reward:
```
For t = 1..T:
    Sample theta → Select arm → Observe reward → Update (lines 9-12)
```
The discount is embedded in the update equations (lines 9-10 multiply by gamma).

**GA-LLAMEA:** Discount is applied at the START of `update()`:
```python
def update(self, arm_name, reward):
    self._apply_discount()        # gamma * all arms
    arm_state.discounted_count += 1.0
    arm_state.discounted_sum += reward
```

**Verdict: MATCH.** The paper's formulation applies gamma to old values before adding new data. The code does the same by discounting first, then adding. The order is: discount → add → compute posterior. This is equivalent to the paper's single-step update equations.

### 3.9 Update Scope

**Paper:** Lines 9-12 loop over ALL arms $i = 1, \dots, K$:
- All arms get their counts/sums multiplied by $\gamma$
- Only the selected arm gets $+1$ to count and $+X_t$ to sum
- ALL arms get posterior recomputed (lines 11-12)

**GA-LLAMEA:**
- `_apply_discount()`: Discounts ALL arms ✓
- `update()`: Only adds to selected arm ✓
- `_update_posterior()`: Only called for the SELECTED arm ✗

**Verdict: MINOR DEVIATION.** The code only recomputes the posterior for the selected arm after update. For unselected arms, the posterior is recomputed lazily during `select_arm()`. This is mathematically equivalent — the posterior values are the same whenever they're needed. It's an optimization (avoid unnecessary computation), not a semantic difference.

---

## 4. Accuracy Rating

### Core Algorithm Fidelity: 9/10

| Aspect | Paper (DS-TS) | GA-LLAMEA | Match? |
|--------|---------------|-----------|--------|
| Discount on counts ($N$) | $\gamma \cdot N + \mathbb{1}$ | `count *= gamma; count += 1` | **Exact** |
| Discount on sums ($\tilde{\mu}$) | $\gamma \cdot \tilde{\mu} + X$ | `sum *= gamma; sum += reward` | **Exact** |
| Posterior mean ($\hat{\mu}$) | $\tilde{\mu} / N$ | `sum / max(eps, count)` | **Exact** (+ safety) |
| Posterior variance ($\tau^2$) | $\min(1/N, \tau_{max}^2)$ | `min(1/count, tau_max^2)` | **Exact** |
| Sampling distribution | $\mathcal{N}(\hat{\mu}, \tau^2)$ | `gauss(mean, sqrt(var))` | **Exact** |
| Arm selection | $\arg\max_i \theta_i$ | Max theta loop | **Exact** |
| Discount scope (all arms) | All arms discounted each step | `_apply_discount()` all arms | **Exact** |
| Initialization | $N=0, \tilde{\mu}=0, \tau=\tau_{max}$ | $N=0, \tilde{\mu}=0, \tau^2=1$ | **Minor diff** |
| Gaussian prior assumption | Implicit Gaussian likelihood | Gaussian sampling | **Match** |
| Reward support $[0, 1]$ | Required by theory | `max(0, min(1, score))` | **Exact** |

### Practical Extensions (not in paper): -1 point

These are additions GA-LLAMEA makes that **deviate from pure DS-TS**:

| Extension | Impact on DS-TS Behavior |
|-----------|-------------------------|
| **Epsilon-greedy floor (40%)** | **Significant.** 40% of selections bypass Thompson Sampling entirely. This overrides the bandit's learned beliefs nearly half the time. The paper relies on TS's natural exploration via variance; adding epsilon-greedy on top dilutes the adaptive signal. |
| **Burn-in phase (min_pulls=5)** | **Moderate.** Forces each arm to be tried 5 times before TS kicks in. Ensures initial statistics are non-trivial. Reasonable but not in the paper. |
| **Stagnation override** | **Moderate.** When triggered, bypasses bandit selection entirely, choosing operators based on posterior means directly (not Thompson sampling). Resets counter, creating periodic forced selections. |
| **discounted_sum_sq** | **None.** Extra tracked field that is never used in selection or update logic. |

### Overall Accuracy Assessment

**The core DS-TS algorithm (equations, update rules, sampling) is faithfully implemented at ~95% accuracy.** The mathematical heart — discounting, posterior computation, and Thompson Sampling — matches the paper exactly.

**The practical wrapper reduces effective DS-TS purity to ~60-65%.** With `epsilon_exploration=0.4`, only 60% of non-burn-in selections use actual Thompson Sampling. The stagnation override further reduces this. In practice, GA-LLAMEA uses DS-TS as one component of a hybrid selection strategy, not as a pure DS-TS implementation.

### Summary Rating

| Dimension | Rating | Notes |
|-----------|--------|-------|
| **Mathematical fidelity** (equations) | **9.5/10** | All update equations match. Only init variance differs trivially. |
| **Algorithmic fidelity** (selection logic) | **6/10** | 40% epsilon-greedy + burn-in + stagnation override dilute pure TS. |
| **Theoretical guarantees preserved** | **3/10** | Paper's regret bounds assume pure DS-TS. Epsilon-greedy and overrides invalidate the theoretical analysis. $\gamma$ is fixed at 0.99 rather than problem-dependent $1-\sqrt{B_T/T}$. |
| **Practical effectiveness** | **8/10** | The extensions are reasonable engineering for low-budget LLM settings where K=4 arms and T~100 evaluations make pure TS unreliable. |

---

## 5. Stagnation Detection Method

**Location:** `ga_llamea_modular/core.py`, lines 216-219 (init), 343-368 (detection), 427-432 (tracking)

### 5.1 Mechanism

Stagnation detection is a **fitness improvement watchdog** that overrides the bandit when the search is stuck.

#### State Variables

```python
self._stagnation_counter = 0                           # Current consecutive no-improvement count
self._stagnation_threshold = 10                        # Trigger override after 10 evaluations with no improvement
self._best_fitness_at_last_improvement = -float('inf') # High-water mark for fitness
```

#### Tracking Logic (per offspring evaluation)

```python
if child.fitness > self._best_fitness_at_last_improvement:
    self._best_fitness_at_last_improvement = child.fitness
    self._stagnation_counter = 0       # RESET: improvement found
else:
    self._stagnation_counter += 1      # INCREMENT: no improvement

# Also incremented on errors/failures:
if child.error:
    self._stagnation_counter += 1
```

#### Override Logic (before each operator selection)

```python
if self._stagnation_counter >= self._stagnation_threshold:  # >= 10
    # Query bandit's current beliefs about operator quality
    refine_arms = [a for a in arm_names if a in ("refine", "simplify")]
    explore_arms = [a for a in arm_names if a in ("crossover", "random_new")]

    refine_mean = max(arms[a].posterior_mean for a in refine_arms)
    explore_mean = max(arms[a].posterior_mean for a in explore_arms)

    if refine_mean >= explore_mean:
        override_pool = refine_arms      # Bandit says: refine is better → exploit
    else:
        override_pool = explore_arms     # Bandit says: explore is better → explore

    operator_name = random.choice(override_pool)
    self._stagnation_counter = 0         # Reset after override
else:
    operator_name, theta = self.bandit.select_arm()  # Normal D-TS selection
```

### 5.2 Design Properties

| Property | Description |
|----------|-------------|
| **Domain-agnostic** | No hardcoded fitness thresholds; uses relative improvement only |
| **Bandit-informed** | Override direction (refine vs explore) is decided by the bandit's learned posterior means, not random |
| **Self-resetting** | Counter resets on improvement AND after override, preventing permanent lock-in |
| **Error-aware** | Failed evaluations (errors, validation failures) increment the counter |
| **Evaluation-level granularity** | Tracks per-evaluation, not per-generation |

### 5.3 When Stagnation Triggers

With default threshold=10 and n_offspring=16, stagnation triggers if:
- 10 consecutive evaluations (possibly less than 1 full generation) produce no fitness improvement
- This includes errors/failures counting as no-improvement

### 5.4 Relationship to DTS Paper

The stagnation mechanism is **entirely absent from the DTS paper**. It is a GA-LLAMEA-specific engineering addition to handle the low-budget, high-variance LLM setting where:
- Only ~4 arms exist (vs. papers testing with K=5-30)
- Total budget is ~100 evaluations (vs. papers testing with T=10,000-100,000)
- Reward distributions are extremely noisy (LLM-generated code quality varies widely)

---

## 6. GA-LLAMEA Prompts

All prompts share a common structure with three BLADE-provided components:
- **role_prompt**: Fixed system prompt defining the LLM's role
- **task_prompt**: Problem-specific description from BLADE
- **format_prompt**: Output format instructions from BLADE

### 6.1 Role Prompt (shared by all operators)

```
You are a highly skilled computer scientist in the field of natural computing.
Your task is to design novel metaheuristic algorithms to solve black box
optimization problems.
```

### 6.2 Population History Block (shared by all operators except init)

```
List of previously generated algorithm names with mean AOCC score:
- AlgorithmA: 0.8234
- AlgorithmB: 0.7891
- AlgorithmC: 0.6542
```

Sorted by fitness (best first).

### 6.3 Initialization Prompt (RandomNewOperator, `is_init=True`)

```
{role_prompt}
{task_prompt}
{example_prompt}

First, describe your new algorithm and main steps in one sentence.
The description must be inside curly braces like this: {Your algorithm description here}.
Next, implement it in Python as a class with __init__(self, budget, dim) and
__call__(self, func) methods.
Do not give additional explanations.

{format_prompt}
```

**Key features:**
- No population history (empty population during init)
- EoH-style structured output with braces for description extraction
- Minimal instructions to avoid constraining creativity

### 6.4 Simplify Prompt

````
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
````

**Key features:**
- Shows parent's full code, name, and fitness
- Instruction: "Refine and **simplify**" — dual objective of improvement + simplification
- Matches LLAMEA's proven Prompt5 verbatim

### 6.5 Refine Prompt

````
{role_prompt}
{task_prompt}
{example_prompt}

{population_history}

Selected algorithm to refine and improve:
Name: {parent.name}
Fitness: {parent.fitness:.4f}
Code:
```python
{parent.code}
```

Refine the strategy of the selected solution to improve it.

{format_prompt}
````

**Key features:**
- Nearly identical to Simplify but without "simplify" instruction
- Focus on strategic refinement rather than code reduction

### 6.6 Crossover Prompt (Guided Concept Transfer)

````
{role_prompt}
{task_prompt}
{example_prompt}

{population_history}

Working Algorithm (fitness: {parent1.fitness:.4f}):
```python
{parent1.code}
```

These are other high-performing solutions discovered during the search.
You may borrow useful ideas, logic, or techniques from them.

Inspiration 1: {insp1.name} (fitness: {insp1.fitness:.4f})
```python
{insp1.code}
```

[Additional inspirations if num_crossover_inspirations > 1...]

Create a new improved solution by combining ideas from the inspiration
solutions while maintaining syntactic correctness.

{format_prompt}
````

**Key features:**
- Shows FULL CODE for both working algorithm and all inspirations
- Parent1 always has highest fitness (enforced by `_select_crossover_parents`)
- "Borrow useful ideas, logic, or techniques" — concept-level guidance
- Inspirations are guaranteed diverse (different code from parent1)

### 6.7 Random New Prompt (Evolution Mode)

````
{role_prompt}
{task_prompt}
{example_prompt}

{population_history}

For correct code structure, follow this template:
```python
import numpy as np

class YourAlgorithm:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        f_opt = np.inf
        x_opt = None
        eval_count = 0

        # Your optimization logic here
        # Use func(x) to evaluate a candidate x (numpy array of shape (dim,))
        # Track eval_count and stop when eval_count >= self.budget

        return f_opt, x_opt
```
Use a DIFFERENT strategy from the algorithms listed above.
This template is only for correct structure and formatting.

Please help me create a new algorithm that has a totally different form
from the given ones.

Generate a completely novel approach that explores a different region of
the algorithm design space.

{format_prompt}
````

**Key features:**
- Provides a **structural template** (reduces format errors) but explicitly states it's only for structure
- Shows population history (to avoid repeating existing approaches)
- Strong instruction to be DIFFERENT and NOVEL

---

## 7. Complete Architecture

### 7.1 Algorithm Flow

```
INITIALIZATION
    └─ For i = 1..n_parents:
        └─ RandomNewOperator(is_init=True) → LLM → code → evaluate → population

EVOLUTION LOOP (while llm_calls < budget)
    └─ For j = 1..n_offspring:
        ├─ STAGNATION CHECK (counter >= 10?)
        │   ├─ YES → Override: pick refine or explore pool based on bandit means
        │   └─ NO  → D-TS Bandit selects operator
        │       ├─ Burn-in? → Force undersampled arm
        │       ├─ Epsilon? → Random arm (40% chance)
        │       └─ Otherwise → Thompson Sampling (sample theta, argmax)
        │
        ├─ OPERATOR DISPATCH
        │   ├─ simplify → select_parent() → SimplifyPrompt
        │   ├─ refine   → select_parent() → RefinePrompt
        │   ├─ crossover → select_crossover_parents() → CrossoverPrompt
        │   └─ random_new → RandomNewPrompt (evolution mode)
        │
        ├─ LLM QUERY → response → extract_code() → validate_code()
        │
        ├─ EVALUATE → problem(child) → fitness
        │
        ├─ REWARD → calculate_reward(fitness, is_valid) → [0, 1]
        │
        ├─ BANDIT UPDATE → bandit.update(operator, reward)
        │
        └─ STAGNATION TRACKING → update counter and high-water mark

    SELECTION
        ├─ Elitism (μ+λ): best from parents + offspring
        └─ Non-elitism (μ,λ): best from offspring only
        └─ Diversity: prefer different code in selected set

    UPDATE BEST → track best_solution across all generations
```

### 7.2 File Map

| File | Purpose |
|------|---------|
| `ga_llamea_modular/core.py` | Main GA_LLaMEA class, evolutionary loop, stagnation |
| `ga_llamea_modular/bandit.py` | DiscountedThompsonSampler (DTS implementation) |
| `ga_llamea_modular/operators.py` | SimplifyOperator, CrossoverOperator, RandomNewOperator, RefineOperator |
| `ga_llamea_modular/utils.py` | calculate_reward, extract_code, validate_code |
| `ga_llamea_modular/interfaces.py` | Protocol definitions (LLM, Problem, Solution) |

### 7.3 Default Configuration

```python
GA_LLaMEA(
    budget=100,              # LLM queries
    n_parents=4,             # Population size (μ)
    n_offspring=16,          # Per generation (λ)
    elitism=True,            # (μ+λ) selection
    discount=0.99,           # D-TS gamma
    tau_max=0.20,            # D-TS max std dev
    epsilon_exploration=0.4, # Random selection floor
    min_pulls_per_arm=5,     # Burn-in per operator
    arm_names=["refine", "simplify", "crossover", "random_new"],
    init_oversample=1,       # No oversampling
)
```

---

## 8. Reward Formula and 4-Arm Mapping

**File:** `ga_llamea_modular/utils.py`, function `calculate_reward`

### 8.1 Base Reward Formula

$$R(c, v) = \begin{cases} \max(0,\ \min(1,\ f_c)) & \text{if } v = \text{True} \\ 0 & \text{if } v = \text{False} \end{cases}$$

Where:
- $f_c$ — child solution's AOCC fitness score after evaluation
- $v$ — validity flag (False on code extraction failure, validation failure, evaluation error, or timeout)

**Code:**
```python
def calculate_reward(child_score: float, is_valid: bool) -> float:
    if not is_valid:
        return 0.0
    return max(0.0, min(1.0, child_score))
```

This is an **absolute fitness reward**: the bandit receives the child's raw fitness as its signal, not a relative improvement over the parent. AOCC is already in $[0, 1]$, so the `clip` is only a safety guard.

### 8.2 Why Absolute (Not Relative) Fitness

An earlier design used `reward = child_score - parent_score`. That caused a systematic bias:

| Operator | Typical parent | Effect |
|----------|---------------|--------|
| `simplify` | Mediocre parent (any) | Small improvement → inflated reward |
| `crossover` | Best parent (forced) | Same improvement → smaller Δ → deflated reward |
| `refine` | Selected parent | Inconsistent baseline |
| `random_new` | No parent | Undefined baseline |

With absolute fitness, all four arms compete on equal footing: the bandit directly learns **"which operator tends to produce the highest-quality solutions on average?"**

### 8.3 The 4 Arms

| Arm | Role | Parent selection | Exploration / Exploitation |
|-----|------|-----------------|---------------------------|
| **`refine`** | Strategic improvement of a single parent — change the search strategy, not just code style | `select_parent()` (fitness-proportional from population) | Exploitation |
| **`simplify`** | Simplify + improve a single parent — reduce complexity while maintaining or improving fitness | `select_parent()` (fitness-proportional from population) | Exploitation |
| **`crossover`** | Concept-level recombination — borrow ideas from 1–2 inspiration solutions into the working algorithm | `select_crossover_parents()` — parent1 = best fitness, inspirations = diverse alternatives | Balanced |
| **`random_new`** | Generate a completely novel algorithm different from all existing population members | No parent (uses population history to avoid repetition) | Exploration |

### 8.4 Reward Flow per Arm

For each offspring evaluation, the reward signal fed to the bandit is identical regardless of which arm was used:

```
Operator selected (arm_name)
    └─ LLM generates child code
        └─ validate_code() → is_valid flag
            └─ evaluate child → child.fitness  (AOCC ∈ [0,1], or -inf on error)
                └─ calculate_reward(child.fitness, is_valid)
                    └─ bandit.update(arm_name, reward)   # D-TS update
```

**Error / timeout cases** always produce `reward = 0.0`, which:
- Decreases the arm's posterior mean (if it was high)
- Increases its variance (via discounting, $1/\sqrt{N}$ grows)
- Counts toward the stagnation counter

### 8.5 Reward Interpretation by Arm

Because rewards are absolute fitness values (not deltas), the bandit's posterior mean $\hat{\mu}$ for each arm converges to the **expected AOCC quality** that arm produces over recent history:

| Typical posterior mean range | Interpretation |
|-----------------------------|----------------|
| $\hat{\mu} \geq 0.8$ | Arm consistently produces high-fitness solutions |
| $0.5 \leq \hat{\mu} < 0.8$ | Arm produces moderate solutions |
| $\hat{\mu} < 0.5$ | Arm struggling (low quality or frequent errors) |
| $\hat{\mu} \approx 0$ | Arm dominated by errors / timeouts |

The stagnation override (Section 5) uses these posterior means to decide whether to force **refine/simplify** (exploit) or **crossover/random_new** (explore) when the search is stuck.
