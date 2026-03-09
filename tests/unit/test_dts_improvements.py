"""Unit tests for D-TS improvements.

Tests the paper's variance formula, epsilon-greedy exploration,
and graduated reward function.
"""

import math
import random

import pytest

from ga_llamea_modular.bandit import DiscountedThompsonSampler
from ga_llamea_modular.utils import calculate_reward


class TestPaperVarianceFormula:
    """Test that the bandit uses the paper's Algorithm 1 formula."""

    def test_posterior_mean_is_discounted_average(self):
        """After one update, posterior_mean = reward (N=1, sum=reward)."""
        bandit = DiscountedThompsonSampler(
            arm_names=["a", "b"], discount=0.9, tau_max=1.0
        )
        bandit.update("a", 0.7)
        state = bandit.get_state_snapshot()
        assert abs(state["a"]["mean"] - 0.7) < 1e-6

    def test_posterior_var_is_one_over_count(self):
        """After one update, posterior_var = min(1/1, tau_max^2) = 1.0 (if tau_max=1)."""
        bandit = DiscountedThompsonSampler(
            arm_names=["a"], discount=0.9, tau_max=1.0
        )
        bandit.update("a", 0.5)
        state = bandit.get_state_snapshot()
        # count=1.0, so var = min(1/1, 1.0^2) = 1.0
        assert abs(state["a"]["var"] - 1.0) < 1e-6

    def test_tau_max_caps_variance(self):
        """Variance should never exceed tau_max^2."""
        bandit = DiscountedThompsonSampler(
            arm_names=["a"], discount=0.9, tau_max=0.15
        )
        # Before any updates, count ≈ 0, so 1/count is huge — should be capped
        bandit._update_posterior(bandit.arms["a"])
        assert bandit.arms["a"].posterior_var <= 0.15**2 + 1e-6

    def test_variance_decreases_with_more_samples(self):
        """More observations should decrease variance (1/N effect)."""
        bandit = DiscountedThompsonSampler(
            arm_names=["a"], discount=0.99, tau_max=10.0
        )
        bandit.update("a", 0.5)
        var_after_1 = bandit.arms["a"].posterior_var

        bandit.update("a", 0.5)
        var_after_2 = bandit.arms["a"].posterior_var

        assert var_after_2 < var_after_1

    def test_discounting_decays_count(self):
        """Unselected arm's count should decay, increasing its variance."""
        bandit = DiscountedThompsonSampler(
            arm_names=["a", "b"], discount=0.9, tau_max=10.0
        )
        # Update arm a, then update arm b multiple times
        bandit.update("a", 0.5)
        var_a_after_pull = bandit.arms["a"].posterior_var

        # Many pulls of b — arm a's count decays
        for _ in range(10):
            bandit.update("b", 0.5)

        bandit._update_posterior(bandit.arms["a"])
        var_a_after_decay = bandit.arms["a"].posterior_var

        assert var_a_after_decay > var_a_after_pull

    def test_backward_compatible_with_old_params(self):
        """Old-style constructor with prior_variance/reward_variance should not crash."""
        bandit = DiscountedThompsonSampler(
            arm_names=["a", "b"],
            discount=0.9,
            tau_max=0.15,
            prior_variance=0.25,
            reward_variance=0.5,
        )
        arm, theta = bandit.select_arm()
        assert arm in ["a", "b"]


class TestEpsilonGreedyExploration:
    """Test the epsilon-greedy exploration floor."""

    def test_epsilon_zero_is_pure_ts(self):
        """With epsilon=0, should always use Thompson Sampling."""
        bandit = DiscountedThompsonSampler(
            arm_names=["a", "b", "c"],
            discount=0.9,
            tau_max=0.15,
            epsilon_exploration=0.0,
        )
        # Just verify it runs without error
        arm, theta = bandit.select_arm()
        assert arm in ["a", "b", "c"]

    def test_epsilon_one_selects_uniformly(self):
        """With epsilon=1.0, every selection should be random."""
        random.seed(42)
        bandit = DiscountedThompsonSampler(
            arm_names=["a", "b", "c"],
            discount=0.9,
            tau_max=0.15,
            epsilon_exploration=1.0,
        )
        # Give arm "a" overwhelmingly good rewards
        for _ in range(20):
            bandit.update("a", 1.0)

        # With epsilon=1.0, all arms should be selected roughly equally
        counts = {"a": 0, "b": 0, "c": 0}
        for _ in range(3000):
            arm, _ = bandit.select_arm()
            counts[arm] += 1

        # Each arm should get roughly 1000 selections (±200)
        for arm, count in counts.items():
            assert 600 < count < 1400, f"Arm {arm} got {count} selections, expected ~1000"

    def test_epsilon_prevents_lock_in(self):
        """With epsilon=0.1, even a dominated arm should get some selections."""
        random.seed(42)
        bandit = DiscountedThompsonSampler(
            arm_names=["dominant", "weak"],
            discount=0.9,
            tau_max=0.1,
            epsilon_exploration=0.1,
        )
        # Make "dominant" overwhelmingly good
        for _ in range(20):
            bandit.update("dominant", 1.0)
            bandit.update("weak", 0.0)

        # Over 200 selections, "weak" should get at least some (from ε exploration)
        weak_count = sum(
            1 for _ in range(200) if bandit.select_arm()[0] == "weak"
        )
        assert weak_count > 5, f"Weak arm only selected {weak_count} times out of 200"


class TestAbsoluteFitnessReward:
    """Test the absolute fitness reward function."""

    def test_invalid_gives_zero(self):
        """Invalid solutions should always get 0.0 reward."""
        assert calculate_reward(0.9, is_valid=False) == 0.0
        assert calculate_reward(0.0, is_valid=False) == 0.0

    def test_valid_returns_fitness_directly(self):
        """Valid solutions should return their fitness as the reward."""
        assert calculate_reward(0.85, is_valid=True) == 0.85
        assert calculate_reward(0.45, is_valid=True) == 0.45
        assert calculate_reward(0.0, is_valid=True) == 0.0

    def test_higher_fitness_gives_higher_reward(self):
        """Higher fitness should produce higher reward."""
        r_low = calculate_reward(0.3, is_valid=True)
        r_mid = calculate_reward(0.6, is_valid=True)
        r_high = calculate_reward(0.9, is_valid=True)
        assert r_low < r_mid < r_high

    def test_reward_clipped_to_unit_interval(self):
        """Reward should be clipped to [0, 1] even for out-of-range inputs."""
        assert calculate_reward(1.5, is_valid=True) == 1.0
        assert calculate_reward(-0.5, is_valid=True) == 0.0

    def test_reward_bounded_zero_to_one(self):
        """Reward should always be in [0, 1]."""
        test_cases = [
            (0.7, True),
            (0.3, True),
            (0.0, True),
            (0.5, True),
            (0.99, True),
            (0.9, False),
            (0.501, True),
            (0.55, True),
            (0.75, True),
        ]
        for child, valid in test_cases:
            r = calculate_reward(child, valid)
            assert 0.0 <= r <= 1.0, f"Reward {r} out of bounds for ({child}, {valid})"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
