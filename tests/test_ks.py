import torch

from llm_sleep.ks import gjsd, lev_reward


def test_lev_reward():
    assert lev_reward("abc", "abc") == 1.0
    assert lev_reward("abc", "xyz", z0=0.5) == 0.0
    assert abs(lev_reward("abcd", "abcx", z0=0.5) - 0.75) < 1e-9


def test_gjsd_zero_when_equal_and_positive_otherwise():
    t = torch.randn(1, 5, 11)
    mask = torch.ones(1, 5)
    for beta in (0.0, 0.5, 1.0):
        assert gjsd(t, t, mask, beta).abs() < 1e-6
        assert gjsd(torch.randn(1, 5, 11), t, mask, beta) > 0
