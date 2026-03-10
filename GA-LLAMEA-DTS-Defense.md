# Defense of Discounted Thompson Sampling (DTS) in GA-LLAMEA

This document outlines the theoretical and practical justifications for the specific implementation choices of Discounted Thompson Sampling (DTS) within the GA-LLAMEA framework, specifically addressing the **Minimum Pulls (Burn-in)** strategy and the **Low-Budget ($T=100$)** constraint.

## 1. Defense of the "Minimum Pulls" (Burn-in) Strategy

### The Critique
*Why did you add a forced "burn-in" phase (min_pulls=6) when Algorithm 1 in Qi et al. (2023) does not explicitly include it?*

### The Defense
While Qi et al. (2023) initialize DTS with a fixed prior distribution (Gaussian), we introduced a **Minimum Pulls (Burn-in)** strategy for three critical reasons that align with the paper's underlying principles:

1.  **Addressing "Exploration Loss" (Paper Alignment)**
    *   **Source**: Remark 1 of Qi et al. (2023) states: *"Combining practical experience, too small sampling variance will make Thompson Sampling lose its exploration ability."*
    *   **Argument**: The paper acknowledges that theoretical priors can lead to practical failures in exploration. By enforcing minimum pulls, we ensure the sampling variance is driven by *observed data* rather than an artificial cap ($\tau_{max}$). This is a practical implementation of the paper's own warning, preventing the bandit from prematurely converging to a suboptimal operator due to initial noise.

2.  **Stabilizing Empirical Estimates**
    *   **Logic**: The core update rule (Line 11) is $\hat{\mu}_{t+1} = \hat{\mu}_t / N_{t+1}$. When $N_t \approx 0$, this estimate is highly unstable.
    *   **Argument**: In the volatile landscape of LLM code generation, a single "lucky" generation can skew the posterior mean significantly. A burn-in phase ensures that $\hat{\mu}_t$ represents a statistically valid empirical average before the bandit begins exploiting it. This is analogous to the initialization phase required by UCB algorithms (which must play each arm once to avoid division by zero).

3.  **Fairness with Baselines**
    *   **Context**: The paper compares DTS against **DS-UCB**.
    *   **Argument**: Standard UCB implementations require initializing all arms. To ensure a fair comparison between our adaptive strategy and standard baselines, we adopted a similar robust initialization. This ensures our bandit starts with a comparable information state to standard deterministic strategies.

---

## 2. Defense of DTS in a Low-Budget Setting ($T=100$)

### The Critique
*DTS is typically analyzed in settings with large horizons ($T=100,000$). How can you justify using it for an optimization budget of only $T=100$?*

### The Defense
Discounted Thompson Sampling is uniquely well-suited for this low-budget, high-stakes optimization task for the following reasons:

1.  **Sample Efficiency of Thompson Sampling**
    *   **Theory**: Empirical studies (e.g., Chapelle & Li, 2011) consistently show that Thompson Sampling is more sample-efficient than UCB or $\epsilon$-greedy algorithms in the early stages of learning.
    *   **Relevance**: With a budget of only 100, we cannot wait for "asymptotic convergence." We need an algorithm that learns *fast*. TS quickly identifies promising regions of the operator space, making it ideal for short horizons.

2.  **The "Effective Memory" Matches the Budget**
    *   **Mechanism**: The discount factor $\gamma$ determines the "effective memory" of the bandit, approximated by $1/(1-\gamma)$.
    *   **Implementation**: We use $\gamma=0.99$.
    *   **Calculation**: $\text{Effective Memory} \approx \frac{1}{1 - 0.99} = 100 \text{ steps}$.
    *   **Argument**: This tuning is mathematically perfect for our budget. It means the bandit considers the *entire* history of the run as relevant (it doesn't "forget" too fast), but it still weights recent successes slightly higher. This allows it to act like a standard Bayesian optimizer that is responsive to the latest generation's breakthroughs without discarding early lessons.

3.  **Handling Rapid Non-Stationarity**
    *   **Context**: In Evolutionary Algorithms, the "best" operator changes rapidly.
        *   *Early Game*: "Random New" or "Crossover" might be best to find a valid starting point.
        *   *Mid Game*: "Refine" might be best to optimize logic.
        *   *Late Game*: "Simplify" might be best to reduce code complexity.
    *   **Argument**: A standard stationary bandit (or simple random selection) would fail to track these rapid shifts within 100 steps. DTS is explicitly designed for **non-stationary** environments. Its ability to "forget" allows it to switch strategies dynamically even within a short window, catching the "phase shifts" of the evolutionary process.

4.  **Cost of Exploration vs. Exploitation**
    *   **Logic**: In a massive budget ($T=100k$), you can afford to waste 1,000 steps exploring. In a micro-budget ($T=100$), every step counts.
    *   **Argument**: DTS provides a "soft" exploration. It doesn't waste steps on clearly bad arms (like $\epsilon$-greedy might), but it probabilistically samples arms that *might* be good. This "probability matching" behavior is the most rational strategy when every evaluation is expensive (both in time and API cost).

### Summary Statement for Research Paper

> *"We adapt the Discounted Thompson Sampling (DTS) algorithm (Qi et al., 2023) for the low-budget regime ($T=100$) of Large Language Model evolution. Unlike asymptotic analyses that require $T \to \infty$, our implementation leverages the high sample efficiency of Thompson Sampling to rapidly identify effective operators. We tune the discount factor $\gamma=0.99$ to create an effective memory window matching the experiment horizon ($1/(1-\gamma) \approx 100$), ensuring the algorithm retains global history while remaining responsive to the non-stationary nature of evolutionary progress. Furthermore, we introduce a robust **minimum-pull initialization ($N=6$)** to stabilize posterior estimates against the high variance of LLM outputs, addressing the practical exploration risks noted by Qi et al."*
