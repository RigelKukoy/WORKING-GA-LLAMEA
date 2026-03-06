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


class TestGraduatedReward:
    """Test the graduated reward function."""

    def test_invalid_gives_zero(self):
        """Invalid solutions should always get 0.0 reward."""
        assert calculate_reward(0.5, 0.9, is_valid=False) == 0.0
        assert calculate_reward(0.0, 0.0, is_valid=False) == 0.0

    def test_large_improvement_gives_one(self):
        """A 10%+ improvement over parent should get 1.0 reward."""
        assert calculate_reward(0.5, 0.55, is_valid=True) == 1.0  # +10%
        assert calculate_reward(0.5, 0.60, is_valid=True) == 1.0  # +20%

    def test_small_improvement_gives_graduated(self):
        """Small improvements should get reward in [0.6, 1.0)."""
        r = calculate_reward(0.5, 0.51, is_valid=True)  # +2%
        assert 0.6 < r < 1.0, f"Small improvement reward {r} not in (0.6, 1.0)"

    def test_small_improvement_less_than_large(self):
        """Larger improvements should get higher rewards."""
        r_small = calculate_reward(0.5, 0.51, is_valid=True)   # +2%
        r_medium = calculate_reward(0.5, 0.525, is_valid=True)  # +5%
        r_large = calculate_reward(0.5, 0.55, is_valid=True)    # +10%
        assert r_small < r_medium < r_large

    def test_improvement_from_zero_parent(self):
        """Improvement from zero parent should get high reward."""
        r = calculate_reward(0.0, 0.3, is_valid=True)
        assert r > 0.1, f"Improvement from 0 got too low reward: {r}"

    def test_zero_parent_no_improvement_gives_small_reward(self):
        """Valid but parent_score <= 0 and no improvement should get 0.1."""
        assert calculate_reward(0.0, 0.0, is_valid=True) == 0.1
        assert calculate_reward(-1.0, -2.0, is_valid=True) == 0.1

    def test_graduated_regression_scale(self):
        """Regression reward should scale with child/parent ratio."""
        # 90% of parent → 0.1 + 0.4×0.9 = 0.46
        r = calculate_reward(1.0, 0.9, is_valid=True)
        assert abs(r - 0.46) < 1e-6

        # 50% of parent → 0.1 + 0.4×0.5 = 0.30
        r = calculate_reward(1.0, 0.5, is_valid=True)
        assert abs(r - 0.30) < 1e-6

        # 0% of parent → 0.1 + 0.4×0.0 = 0.10
        r = calculate_reward(1.0, 0.0, is_valid=True)
        assert abs(r - 0.10) < 1e-6

    def test_reward_bounded_zero_to_one(self):
        """Reward should always be in [0, 1]."""
        test_cases = [
            (0.5, 0.7, True),
            (0.5, 0.3, True),
            (0.5, 0.0, True),
            (0.5, 0.5, True),
            (0.0, 0.0, True),
            (1.0, 0.99, True),
            (0.5, 0.9, False),
            (0.5, 0.501, True),   # tiny improvement
            (0.5, 0.55, True),    # 10% improvement
            (0.5, 0.75, True),    # 50% improvement
        ]
        for parent, child, valid in test_cases:
            r = calculate_reward(parent, child, valid)
            assert 0.0 <= r <= 1.0, f"Reward {r} out of bounds for ({parent}, {child}, {valid})"

    def test_close_to_parent_gets_higher_reward(self):
        """Closer to parent (regression) should give higher reward than far from parent."""
        r_close = calculate_reward(0.5, 0.48, is_valid=True)  # close
        r_far = calculate_reward(0.5, 0.1, is_valid=True)     # far
        assert r_close > r_far


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
