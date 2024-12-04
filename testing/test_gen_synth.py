import pytest
from conv_imputer import gen_synth
import numpy as np

def test_weekly_daily_signal():
    signal = gen_synth.weekly_daily_signal(num_weeks=4, start_date='2023-01-02')
    assert(len(signal) == 672) # 4*7*24

def test_uniform_random_missing_patterns():
    arr = np.zeros((4,168))
    patterns = gen_synth.uniform_random_missing_patterns(arr, num_patterns=5, num_missings=30)
    for pat in patterns:
        assert(np.sum(pat) > 0)