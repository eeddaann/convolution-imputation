import pandas as pd
import numpy as np
from statsmodels.tsa.deterministic import DeterministicProcess

def weekly_daily_signal(num_weeks=4, start_date='2023-01-02'):
    date_range = pd.date_range(start=start_date, periods=24*7*num_weeks, freq='H')

    daily_proc = DeterministicProcess(date_range, constant=True, period=24, fourier=1).in_sample().to_numpy()
    daily = pd.Series(np.sum(daily_proc,axis=1), index=date_range)

    weekly_proc = DeterministicProcess(date_range, constant=True, period=7*24, fourier=1).in_sample().to_numpy()
    weekly = pd.Series(np.sum(weekly_proc,axis=1), index=date_range)


    full_proc = np.sum(daily_proc + weekly_proc,axis=1)
    full = pd.Series(full_proc, index=date_range)
    return full

def uniform_random_missing_patterns(arr, num_patterns=100, num_missings=30):
    patterns = []
    if arr.shape[1] != 168:
        print("INFO: row size expected to be number of hours in a week (168)")
    for i in range(num_patterns):
        mask = np.zeros_like(arr)
        rows = np.random.randint(0, mask.shape[0],num_missings)
        cols = np.random.randint(0, mask.shape[1],num_missings)
        
        indexarray = np.array([rows,cols])
        rows, cols = indexarray[0], indexarray[1]


        mask[rows, cols] = 1

        patterns.append(mask)
    return patterns