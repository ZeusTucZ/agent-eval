import pytest

from evalkit.stats import bootstrap_mean, mcnemar_exact

@pytest.mark.parametrize("improved, regressed, expected", [
    (6, 1, 0.125),         # 2 * (C(7,0) + C(7,1)) / 2^7 = 2 * 8 / 128
    (8, 0, 2 / 256),       # 2 * 1 / 2^8
    (5, 5, 1.0),           # an even split can't be evidence of anything
    (0, 0, 1.0),           # no disagreements at all
    (1, 0, 1.0),           # a single flip proves nothing
    (10, 2, 2 * (1 + 12 + 66) / 4096),
])
def test_mcnemar_matches_hand_calculations(improved, regressed, expected):
    assert mcnemar_exact(improved, regressed) == pytest.approx(expected)

def test_mcnemar_is_symmetric():
    assert mcnemar_exact(6, 1) == mcnemar_exact(1, 6)

def test_bootstrap_of_identical_values_has_no_spread():
    interval = bootstrap_mean([0.5] * 20)
    assert (interval.mean, interval.low, interval.high) == (0.5, 0.5, 0.5)
    assert interval.excludes_zero

def test_bootstrap_interval_contains_the_mean_and_is_reproducible():
    values = [0, 0, 0, 1, 1, 0, 1, 0, 0, 1]
    interval = bootstrap_mean(values)
    assert interval.low < interval.mean < interval.high
    assert interval.mean == 0.4
    assert interval == bootstrap_mean(values)

def test_bootstrap_matches_the_normal_approximation():
    # Half 0s and half 1s: the standard error is 0.5 / sqrt(400) = 0.025, so the
    # 95% interval should be about 0.5 +/- 1.96 * 0.025 = 0.451 to 0.549.
    interval = bootstrap_mean([0.0, 1.0] * 200)
    assert interval.low == pytest.approx(0.451, abs=0.01)
    assert interval.high == pytest.approx(0.549, abs=0.01)

def test_small_samples_give_wide_intervals():
    small = bootstrap_mean([0.0, 1.0] * 5)
    large = bootstrap_mean([0.0, 1.0] * 200)
    assert small.high - small.low > 3 * (large.high - large.low)

def test_an_interval_straddling_zero_does_not_exclude_it():
    assert not bootstrap_mean([-1.0, 1.0] * 20).excludes_zero
    assert bootstrap_mean([1.0, 2.0] * 20).excludes_zero
