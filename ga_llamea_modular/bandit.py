"""
Discounted Thompson Sampling (D-TS) Bandit
==========================================

This module implements the multi-armed bandit used for adaptive operator selection
in GA-LLAMEA. It uses Discounted Thompson Sampling with Gaussian priors,
following Qi et al. (2023) Algorithm 1.

ALGORITHM (Qi et al., Algorithm 1):
    For each arm i, we track:
        - discounted_count N_t(gamma, i) = gamma * N_{t-1} + 1{arm i selected}
        - discounted_sum   mu~_t(gamma, i) = gamma * mu~_{t-1} + 1{arm i selected} * X_t
    
    Posterior:
        mean = mu~_t / N_t                         (discounted empirical average)
        var  = min(1/N_t, tau_max^2)                (simple 1/n variance, capped)
    
    Selection:
        Sample theta_t(i) ~ N(mean_i, var_i) for each arm
        Select arm with highest theta
        With probability epsilon, select a random arm instead (exploration floor)

KEY CONCEPTS:
    - Arms: Genetic operators (simplify, crossover, random_new, etc.)
    - Reward: Fitness-based signal bounded in [0, 1]
    - Discount factor (gamma): Controls how fast old observations fade
    - tau_max: Caps sampling variance to prevent over-exploration
    - epsilon_exploration: Random selection probability to prevent arm extinction

REFERENCES:
    - "Discounted Thompson Sampling for Non-Stationary Bandit Problems" (Qi et al., 2023)

USAGE:
    from ga_llamea_modular.bandit import DiscountedThompsonSampler
    
    bandit = DiscountedThompsonSampler(
        arm_names=["simplify", "crossover", "random_new"],
        discount=0.95,
        tau_max=0.10,
        epsilon_exploration=0.1,  # 10% random arm selection
    )
    
    arm, theta = bandit.select_arm()
    bandit.update(arm, reward=0.5)
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
    """D-TS bandit following Qi et al. (2023) Algorithm 1.
    
    Uses Discounted Thompson Sampling with Gaussian priors for adaptive
    operator selection. Includes an epsilon-greedy exploration floor to
    prevent permanent arm extinction in low-budget settings.
    
    The algorithm:
        1. With probability epsilon, select a random arm (exploration floor)
        2. Otherwise, for each arm sample theta ~ N(mu_hat, tau^2)
           where mu_hat = discounted_sum / discounted_count
           and tau = min(1/sqrt(discounted_count), tau_max)
        3. Select the arm with highest theta (Thompson Sampling)
        4. After observing reward, update statistics with discount
    
    Args:
        arm_names: List of operator names
        discount: Exponential discount factor gamma in (0, 1]. Lower = faster adaptation.
                 Default 0.95 (effective window ~20 observations).
        tau_max: Maximum sampling standard deviation (caps exploration).
                 Paper recommends tau_max ~ mu_max/5. Default 0.10.
        epsilon_exploration: Probability of selecting a random arm instead of
                           using Thompson Sampling. Prevents arm extinction.
                           0.0 = pure TS, 1.0 = fully random. Default 0.1.
        epsilon: Small constant for numerical stability.
    
    Example:
        >>> bandit = DiscountedThompsonSampler(
        ...     arm_names=["simplify", "crossover", "random_new"],
        ...     discount=0.9,
        ...     tau_max=0.15,
        ...     epsilon_exploration=0.1,
        ... )
        >>> arm, theta = bandit.select_arm()
        >>> bandit.update(arm, reward=0.3)
    """

    def __init__(
        self,
        arm_names: List[str],
        discount: float = 0.95,
        tau_max: float = 0.10,
        epsilon_exploration: float = 0.1,
        min_pulls: int = 3,
        epsilon: float = 1e-6,
        # Backward-compatible: accept but ignore old Bayesian params
        prior_mean: float = 0.0,
        prior_variance: float = 1.0,
        reward_variance: float = 1.0,
        **kwargs,
    ):
        # Validate parameters
        if not 0 < discount <= 1:
            raise ValueError("discount must be in (0, 1].")
        if tau_max <= 0:
            raise ValueError("tau_max must be positive.")
        if not 0 <= epsilon_exploration <= 1:
            raise ValueError("epsilon_exploration must be in [0, 1].")
        if min_pulls < 0:
            raise ValueError("min_pulls must be non-negative.")

        self.arm_names = arm_names
        self.discount = discount
        self.tau_max = tau_max
        self.epsilon_exploration = epsilon_exploration
        self.min_pulls = min_pulls
        self.epsilon = epsilon

        # Initialize arm states
        self.arms: Dict[str, ArmState] = {name: ArmState() for name in arm_names}
        self.total_pulls = 0

    def select_arm(self) -> Tuple[str, float]:
        """Select an arm using Thompson Sampling with epsilon-greedy floor.
        
        Prioritizes arms that haven't met the min_pulls threshold (Burn-in).
        Then, with probability epsilon_exploration, selects a random arm.
        Otherwise, samples theta from each arm's posterior and picks
        the arm with the highest sampled value.
        
        Returns:
            Tuple of (selected_arm_name, sampled_theta_value)
        """
        # 1. Burn-in Phase: Force exploration of arms with insufficient data
        if self.min_pulls > 0:
            # Find arms that haven't been pulled enough
            undersampled_arms = [
                name for name, state in self.arms.items() 
                if state.pulls < self.min_pulls
            ]
            if undersampled_arms:
                # Deterministically pick the first one (or random, doesn't matter much)
                # Random is better to avoid order bias in a single generation
                chosen_arm = random.choice(undersampled_arms)
                
                # We still need to return a theta for logging. 
                # Since we have no data, sample from the prior (mean=0, var=tau_max^2)
                # or just return 0.0. Let's sample from current belief to be consistent.
                arm_state = self.arms[chosen_arm]
                self._update_posterior(arm_state)
                std_dev = math.sqrt(max(self.epsilon, arm_state.posterior_var))
                arm_state.last_theta = random.gauss(arm_state.posterior_mean, std_dev)
                
                self.total_pulls += 1
                return chosen_arm, arm_state.last_theta

        # 2. Epsilon-greedy exploration floor
        if self.epsilon_exploration > 0 and random.random() < self.epsilon_exploration:
            chosen_arm = random.choice(self.arm_names)
            # Still compute posteriors and sample theta for logging
            for arm_name, arm_state in self.arms.items():
                self._update_posterior(arm_state)
                std_dev = math.sqrt(max(self.epsilon, arm_state.posterior_var))
                arm_state.last_theta = random.gauss(arm_state.posterior_mean, std_dev)
            self.total_pulls += 1
            return chosen_arm, self.arms[chosen_arm].last_theta

        # Thompson Sampling: sample theta for each arm, pick highest
        best_arm = self.arm_names[0]
        best_theta = float("-inf")

        for arm_name, arm_state in self.arms.items():
            self._update_posterior(arm_state)

            # Sample theta ~ N(mu_hat, tau^2) per Algorithm 1 line 5
            std_dev = math.sqrt(max(self.epsilon, arm_state.posterior_var))
            theta = random.gauss(arm_state.posterior_mean, std_dev)

            arm_state.last_theta = theta

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
        """Update posterior mean and variance using the paper's formula.
        
        Following Qi et al. Algorithm 1 (lines 11-12):
            mu_hat = discounted_sum / discounted_count
            tau    = min(1/sqrt(discounted_count), tau_max)
        """
        count = max(self.epsilon, arm_state.discounted_count)

        # Paper Algorithm 1, line 11: mu_hat = mu~/N
        arm_state.posterior_mean = arm_state.discounted_sum / count

        # Paper Algorithm 1, line 12: tau = min(1/sqrt(N), tau_max)
        arm_state.posterior_var = min(1.0 / count, self.tau_max**2)
