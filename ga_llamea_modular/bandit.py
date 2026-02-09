"""
Discounted Thompson Sampling (D-TS) Bandit
==========================================

This module implements the multi-armed bandit used for adaptive operator selection
in GA-LLAMEA. It uses Discounted Thompson Sampling with Gaussian priors.

ALGORITHM OVERVIEW:
    D-TS maintains a posterior distribution for each operator (arm) and samples
    from these distributions to select which operator to use. The "discounted"
    aspect means older observations are weighted less, allowing the bandit to
    adapt to non-stationary reward distributions.

KEY CONCEPTS:
    - Arms: The three genetic operators (mutation, crossover, random_new)
    - Reward: Fitness improvement (child_fitness - baseline_fitness)
    - Discount factor (γ): Controls how fast old observations fade (0.9 = 10% decay per update)
    - Posterior: Normal distribution N(μ, τ²) for each arm's expected reward

MATHEMATICAL DETAILS:
    For each arm, we track:
        - discounted_count: Σᵢ γⁿ⁻ⁱ (effective sample size)
        - discounted_sum: Σᵢ γⁿ⁻ⁱ rᵢ (weighted sum of rewards)
        - discounted_sum_sq: Σᵢ γⁿ⁻ⁱ rᵢ² (for variance estimation)
    
    Bayesian update combines prior with likelihood:
        posterior_var = 1 / (1/prior_var + count/observed_var)
        posterior_mean = posterior_var × (prior_mean/prior_var + sample_mean×count/observed_var)

BLADE INTEGRATION:
    This bandit is independent of BLADE - it only cares about arm names
    and numerical rewards. GA-LLAMEA uses it internally to decide which
    genetic operator to apply at each step.

REFERENCES:
    - "Discounted Thompson Sampling for Non-Stationary Bandit Problems" (Qi et al.)
    - Thompson Sampling: "On the Likelihood that One Unknown Probability Exceeds Another"

USAGE:
    from ga_llamea_modular.bandit import DiscountedThompsonSampler
    
    bandit = DiscountedThompsonSampler(
        arm_names=["mutation", "crossover", "random_new"],
        discount=0.9,  # 10% decay per update
    )
    
    # Select an arm
    arm, theta = bandit.select_arm()
    print(f"Selected: {arm} with θ={theta:.4f}")
    
    # Update with observed reward
    bandit.update(arm, reward=0.5)
    
    # Get current state for logging
    state = bandit.get_state_snapshot()
"""

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class ArmState:
    """Holds the sufficient statistics for a single arm.
    
    These statistics are updated with each observation and used to
    compute the posterior distribution for Thompson Sampling.
    
    Attributes:
        discounted_count: Effective sample size (decayed over time)
        discounted_sum: Weighted sum of observed rewards
        discounted_sum_sq: Weighted sum of squared rewards (for variance)
        posterior_mean: Current estimated mean reward
        posterior_var: Current uncertainty (variance) in mean estimate
        last_theta: Most recent sampled θ value (for logging)
        pulls: Total number of times this arm was selected (not discounted)
    """
    discounted_count: float = 0.0
    discounted_sum: float = 0.0
    discounted_sum_sq: float = 0.0
    posterior_mean: float = 0.0
    posterior_var: float = 1.0
    last_theta: float = 0.0
    pulls: int = 0


class DiscountedThompsonSampler:
    """D-TS bandit with Gaussian priors for adaptive operator selection.
    
    This bandit learns which operator (mutation, crossover, random_new) produces
    the best offspring over time, adapting to non-stationary reward distributions
    through exponential discounting.
    
    The algorithm:
        1. For each arm, maintain a posterior distribution N(μ, τ²)
        2. Sample θ ~ N(μ, τ²) for each arm
        3. Select the arm with highest θ (Thompson Sampling)
        4. After observing reward, update statistics with discount
    
    Args:
        arm_names: List of operator names (e.g., ["mutation", "crossover", "random_new"])
        discount: Exponential discount factor γ ∈ (0, 1]. Lower = faster adaptation.
                  Default 0.9 means each observation loses 10% weight per new update.
        prior_mean: Prior mean for arm rewards (μ₀). Default 0.0.
        prior_variance: Prior variance for arm rewards (σ₀²). Default 1.0.
        reward_variance: Expected variance of observed rewards (σ²). Default 1.0.
        tau_max: Maximum posterior standard deviation (caps exploration). Default 5.0.
        epsilon: Small constant for numerical stability. Default 1e-6.
    
    Example:
        >>> bandit = DiscountedThompsonSampler(
        ...     arm_names=["mutation", "crossover", "random_new"],
        ...     discount=0.9,
        ... )
        >>> arm, theta = bandit.select_arm()
        >>> # ... apply operator and observe reward ...
        >>> bandit.update(arm, reward=0.3)
    """

    def __init__(
        self,
        arm_names: List[str],
        discount: float = 0.9,
        prior_mean: float = 0.0,
        prior_variance: float = 1.0,
        reward_variance: float = 1.0,
        tau_max: float = 5.0,
        epsilon: float = 1e-6,
    ):
        # Validate parameters
        if not 0 < discount <= 1:
            raise ValueError("discount must be in (0, 1].")
        if prior_variance <= 0:
            raise ValueError("prior_variance must be positive.")
        if tau_max <= 0:
            raise ValueError("tau_max must be positive.")

        self.arm_names = arm_names
        self.discount = discount
        self.prior_mean = prior_mean
        self.prior_variance = prior_variance
        self.reward_variance = reward_variance
        self.tau_max = tau_max
        self.epsilon = epsilon

        # Initialize arm states
        self.arms: Dict[str, ArmState] = {name: ArmState() for name in arm_names}
        self.total_pulls = 0

    def select_arm(self) -> Tuple[str, float]:
        """Sample an arm according to Thompson Sampling.
        
        For each arm, samples θ from the posterior distribution and
        selects the arm with the highest sampled value.
        
        Returns:
            Tuple of (selected_arm_name, sampled_theta_value)
            
        Note:
            The returned theta value is useful for logging to understand
            which arm the bandit believed was best at decision time.
        """
        best_arm = self.arm_names[0]
        best_theta = float("-inf")

        for arm_name, arm_state in self.arms.items():
            # Update posterior statistics before sampling
            self._update_posterior(arm_state)

            # Sample θ ~ N(μ̂, τ²)
            std_dev = math.sqrt(max(self.epsilon, arm_state.posterior_var))
            std_dev = min(std_dev, self.tau_max)  # Cap exploration
            theta = random.gauss(arm_state.posterior_mean, std_dev)

            arm_state.last_theta = theta

            # Select arm with maximum sampled value
            if theta > best_theta:
                best_theta = theta
                best_arm = arm_name

        self.total_pulls += 1
        return best_arm, best_theta

    def update(self, arm_name: str, reward: float) -> None:
        """Update the bandit with the observed reward.
        
        This applies discounting to all arms (exponential forgetting)
        and then adds the new observation to the selected arm.
        
        Args:
            arm_name: The arm that was selected (must be in arm_names)
            reward: The observed reward (fitness improvement, should be >= 0)
        """
        # Apply discount to ALL arms (exponential forgetting)
        self._apply_discount()

        # Update selected arm with new observation
        arm_state = self.arms[arm_name]
        arm_state.discounted_count += 1.0
        arm_state.discounted_sum += reward
        arm_state.discounted_sum_sq += reward**2
        arm_state.pulls += 1

        # Recompute posterior
        self._update_posterior(arm_state)

    def get_state_snapshot(self) -> Dict[str, Dict[str, float]]:
        """Return a serializable view of the current posterior statistics.
        
        Useful for logging, debugging, and analyzing which operators
        the bandit prefers.
        
        Returns:
            Dict mapping arm names to their statistics:
                - count: Effective sample size (discounted)
                - mean: Posterior mean reward estimate
                - var: Posterior variance
                - std: Posterior standard deviation
                - theta: Last sampled θ value
                - pulls: Total number of selections (not discounted)
        """
        snapshot: Dict[str, Dict[str, float]] = {}
        for arm_name, arm_state in self.arms.items():
            snapshot[arm_name] = {
                "count": arm_state.discounted_count,
                "mean": arm_state.posterior_mean,
                "var": arm_state.posterior_var,
                "std": math.sqrt(max(self.epsilon, arm_state.posterior_var)),
                "theta": arm_state.last_theta,
                "pulls": arm_state.pulls,
            }
        return snapshot

    def _apply_discount(self) -> None:
        """Apply exponential discount to all arm statistics.
        
        This implements the "forgetting" mechanism that allows the bandit
        to adapt to non-stationary reward distributions.
        """
        if self.discount == 1.0:
            return  # No discounting

        for arm_state in self.arms.values():
            arm_state.discounted_count *= self.discount
            arm_state.discounted_sum *= self.discount
            arm_state.discounted_sum_sq *= self.discount

    def _update_posterior(self, arm_state: ArmState) -> None:
        """Update posterior mean and variance using Bayesian update.
        
        Combines the prior distribution with observed data to compute
        the posterior distribution for the arm's expected reward.
        """
        count = max(self.epsilon, arm_state.discounted_count)
        sample_mean = arm_state.discounted_sum / count

        # Compute observed variance
        if arm_state.discounted_count <= self.epsilon:
            observed_var = self.reward_variance
        else:
            mean_sq = sample_mean**2
            moment = arm_state.discounted_sum_sq / count
            observed_var = max(
                self.epsilon,
                moment - mean_sq if moment > mean_sq else self.reward_variance,
            )

        # Bayesian update: posterior = prior + likelihood
        inv_prior = 1.0 / self.prior_variance
        inv_likelihood = count / max(self.epsilon, observed_var)
        posterior_var = 1.0 / max(self.epsilon, inv_prior + inv_likelihood)
        posterior_var = min(self.tau_max**2, posterior_var)
        posterior_mean = posterior_var * (
            self.prior_mean * inv_prior + sample_mean * inv_likelihood
        )

        arm_state.posterior_mean = posterior_mean
        arm_state.posterior_var = posterior_var
