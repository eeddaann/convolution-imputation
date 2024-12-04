import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from astropy.convolution import interpolate_replace_nans

def get_pivoted_weekly(data):
    df = pd.DataFrame(data)
    df["updatetime"]  = data.index
    df["week_start"] = df["updatetime"].dt.to_period('W-SUN').apply(lambda r: r.start_time)
    df["tow"] = df['updatetime'] - df['week_start']
    pivoted = pd.pivot_table(df, values = 0, index=['week_start'], columns = 'tow').reset_index()
    return pivoted.replace(-666,np.nan)

def _impute_conv_array(data, kernel):
    pivoted = get_pivoted_weekly(data).drop(["week_start"],axis=1).values.copy()
    pad_width = max(kernel.shape[0],kernel.shape[1]) //2
    padded = np.pad(pivoted.copy(),mode='edge',pad_width=pad_width)
    interpolated = interpolate_replace_nans(padded, kernel)
    return interpolated[pad_width:-pad_width,pad_width:-pad_width]

def impute(data, kernel):
    pivoted = get_pivoted_weekly(data)
    restored = pivoted.set_index('week_start').unstack([0]).reset_index()
    restored.index = pd.to_datetime(restored.week_start) + pd.to_timedelta(restored['tow'])
    restored = restored.sort_index()
    restored["mean_hr"] = _impute_conv_array(data, kernel).flatten()
    restored["is_imputed"] = restored["mean_hr"] - restored[0]
    restored["is_imputed"] = restored["is_imputed"].replace(np.nan,True)
    restored = restored.drop([0,"tow","week_start"],axis=1)
    return restored

def plot(data, kernel):
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(22, 5))
    pivoted = get_pivoted_weekly(data)
    ax1.imshow(pivoted.iloc[:,1:], aspect='auto')
    ax1.set_title('Before')
    ax1.set_xticks(list(range(0,168,24)),["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
    ax1.set_yticks(list(pivoted.iloc[::1].index),list(pivoted.iloc[::1].week_start.apply(lambda x: x.strftime('%Y-%m-%d'))))
    imputed = _impute_conv_array(data, kernel)
    ax2.imshow(imputed, aspect='auto')
    ax2.set_title('After')
    ax2.set_xticks(list(range(0,168,24)),["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
    ax2.set_yticks(list(pivoted.iloc[::1].index),list(pivoted.iloc[::1].week_start.apply(lambda x: x.strftime('%Y-%m-%d'))))
    fig.suptitle('Data Imputation', fontsize=16)
    plt.figure()