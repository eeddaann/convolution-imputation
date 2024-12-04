import pytest
from conv_imputer import gen_synth, impute_weekly
import numpy as np
from astropy.convolution import Gaussian2DKernel

def test_impute_weekly():
    data = gen_synth.weekly_daily_signal()
    missing_pattern  = gen_synth.uniform_random_missing_patterns(impute_weekly.get_pivoted_weekly(data).iloc[:,1:],
                                                                  num_patterns=1, num_missings=30)
    data.iloc[missing_pattern[0].flatten().nonzero()] = np.nan # set missing values
    x_size,y_size=11, 5
    x_stddev,y_stddev=1.28,0.46
    kernel  = Gaussian2DKernel(x_stddev = x_stddev,y_stddev = y_stddev,x_size = x_size,y_size = y_size)
    result = impute_weekly.impute(data, kernel)
    assert(result.is_imputed.sum() > 0)