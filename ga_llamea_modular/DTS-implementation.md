# DTS Implementation Improvements — Walkthrough

## Changes Made

### 1. Paper's Variance Formula — [bandit.py](file:///c:/Users/Kukoy/OneDrive/Documents/GA-LLAMEA-THESIS-NEW/WORKING-GA-LLAMEA/ga_llamea_modular/bandit.py)

Replaced the custom Bayesian posterior update with the paper's Algorithm 1 formula:

```diff
 # OLD: Bayesian posterior (not what the paper specifies)
-observed_var = max(ε, E[X²] - E[X]²)
-inv_prior = 1/σ²_prior
-inv_likelihood = n_eff / observed_var
-posterior_var = min(τ_max², 1/(inv_prior + inv_likelihood))
-posterior_mean = posterior_var × (μ_prior/σ²_prior + x̄ × inv_likelihood)

 # NEW: Paper's Algorithm 1 (lines 11-12)
+posterior_mean = discounted_sum / discounted_count
+posterior_var = min(1.0 / discounted_count, tau_max²)
```

### 2. Epsilon-Greedy Exploration — [bandit.py](file:///c:/Users/Kukoy/OneDrive/Documents/GA-LLAMEA-THESIS-NEW/WORKING-GA-LLAMEA/ga_llamea_modular/bandit.py)

Added `epsilon_exploration` parameter (default 0.1). With probability ε, selects a random arm instead of Thompson Sampling. This prevents permanent arm extinction from the discount-induced death spiral.

### 3. Graduated Reward — [utils.py](file:///c:/Users/Kukoy/OneDrive/Documents/GA-LLAMEA-THESIS-NEW/WORKING-GA-LLAMEA/ga_llamea_modular/utils.py)

Replaced binary `{0.0, 1/3, 1.0}` with graduated `{0.0, 0.1-0.5, 1.0}`:

| Outcome | Old Reward | New Reward |
|---------|-----------|------------|
| Error/timeout | 0.0 | 0.0 |
| Valid, terrible (≈0% of parent) | 0.333 | 0.10 |
| Valid, okay (≈50% of parent) | 0.333 | 0.30 |
| Valid, close (≈90% of parent) | 0.333 | 0.46 |
| Valid, improvement | 1.0 | 1.0 |

### 4. Parameter Passthrough — [core.py](file:///c:/Users/Kukoy/OneDrive/Documents/GA-LLAMEA-THESIS-NEW/WORKING-GA-LLAMEA/ga_llamea_modular/core.py), [ga_llamea.py](file:///c:/Users/Kukoy/OneDrive/Documents/GA-LLAMEA-THESIS-NEW/WORKING-GA-LLAMEA/iohblade/methods/ga_llamea.py)

New defaults: `tau_max=0.15`, `epsilon_exploration=0.1`

### 5. Tuning Script — [run-dts-tuning.py](file:///c:/Users/Kukoy/OneDrive/Documents/GA-LLAMEA-THESIS-NEW/WORKING-GA-LLAMEA/run-dts-tuning.py)

```bash
# Run with new defaults
python run-dts-tuning.py

# Adjust parameters
python run-dts-tuning.py --discount 0.72 --tau_max 0.2 --epsilon_exploration 0.15

# Test without crossover
python run-dts-tuning.py --arms simplify random_new
```

---

## Test Results

All 6 verification tests passed:

| Test | Result |
|------|--------|
| Paper variance formula (mean = sum/count) | ✅ PASS |
| tau_max caps variance correctly | ✅ PASS |
| Epsilon=1.0 selects uniformly (~1000 each) | ✅ PASS |
| Graduated reward edge cases (7 tests) | ✅ PASS |
| Backward compatibility (old params accepted) | ✅ PASS |
| Epsilon prevents lock-in (weak arm > 5/200) | ✅ PASS |

---

## New Default Parameters

| Parameter | Old Default | New Default | Why |
|-----------|------------|-------------|-----|
| [tau_max](file:///c:/Users/Kukoy/OneDrive/Documents/GA-LLAMEA-THESIS-NEW/WORKING-GA-LLAMEA/tests/unit/test_dts_improvements.py#38-46) | 0.1 | **0.15** | Better for graduated rewards (μ_max ≈ 0.5 → τ_max ≈ 0.1-0.15) |
| `epsilon_exploration` | N/A | **0.1** | 10% exploration floor prevents arm extinction |
| [discount](file:///C:/tmp/bandit_simulation.py#40-44) | 0.9 | 0.9 | Unchanged (tune via `--discount` flag) |
