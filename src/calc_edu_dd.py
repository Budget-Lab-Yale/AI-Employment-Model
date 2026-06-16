import pandas
import numpy
import os
import psutil
import gc
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pyarrow as pa
import pyarrow.parquet as pq

from pandas import read_csv, DataFrame, to_datetime 
from parsing import month_helper
from datetime import date
from calc_other import calc_avg2

def build_mis_edu(ID: str, start: str, end: str, freq = 1, occ_subset = '', time = False):
    """
    Builds a time series of education/recency-of-graduation dissimilarity scores.

    `start` is the FIRST month that appears in the output. For each target month
    the score is a `freq`-month *trailing* moving average: the window covers the
    target month and the (freq-1) months immediately before it. The look-back is
    handled internally, so the caller no longer needs to offset `start` to leave
    room for the window (e.g. starting at "2020.11" just to get a 3-month average
    at January 2021). Pass the actual first month you want and freq does the rest.

    Parameters
    ----------
    ID         : str – Subdirectory identifying the model/data variant.
    start      : str – First target month in "YYYY.MM" format, inclusive.
    end        : str – Last target month in "YYYY.MM" format, inclusive.
    freq       : int – Window size in months (1 = monthly, 3 = 3-month trailing avg).
    occ_subset : str – Optional "first.last" occupation-code filter, '' for all.
    """

    # Parse start and end dates into iterable format
    start = start.split('.')
    end   = end.split('.')

    yr_dex = int(start[0])
    mn_dex = int(start[1])

    # Loop runs through `end` inclusive; set the boundary to the month *after* end
    # so the while condition below stops once `end` has been computed.
    end_yr = int(end[0])
    end_mn = int(end[1]) + 1
    if end_mn > 12:
        end_mn -= 12
        end_yr += 1

    # October 2025 has no real data; its own data point is reported as NaN, while
    # it is skipped when it falls inside another month's trailing window.
    oct_25 = "202510"
    
    # Loop through each month of a given time period
    while((yr_dex != end_yr) or (mn_dex != end_mn)):
        # Combine current month and year to get data stamp
        month = str(yr_dex) + str(mn_dex).zfill(2)

        if "months" not in locals():
            months = calc_mis_edu_month(ID, month, freq, occ_subset, oct_25)
        else: 
            months = pandas.concat([months,calc_mis_edu_month(ID,month,freq,occ_subset,oct_25)])

        # Iterate Month/Year
        mn_dex += 1
        if mn_dex > 12:
            mn_dex = mn_dex % 12
            yr_dex += 1

    data = months.reset_index().rename(
        columns = {
            "index": "MONTH",
            0: "dissimilarity"
        }
    )
    deltas = pandas.Series()
    deltas = data
    deltas.loc[deltas["MONTH"] == oct_25, "dissimilarity"] = numpy.nan
    
    if time:
        deltas = deltas.assign(
            time = lambda x: to_datetime(x["MONTH"], format='%Y%m').dt.strftime('%Y-%m-02'))
    else:
        deltas = deltas.assign(time = lambda x: range(1, len(x) + 1))
    
    deltas = deltas[["time", "dissimilarity"]]

    return deltas


def calc_mis_edu_month(ID: str, this_month: str, freq: int, occ_subset: str, oct_25: str):
    
    if this_month == oct_25:
        out = pandas.Series({this_month: float('nan')})
        return out
    
    # Get the target month data (the most recent month of the trailing window)
    month = get_month_edu_dd(ID, this_month, occ_subset)
    
    # Build a freq-month *trailing* moving average for occupation counts by
    # looking BACK (freq-1) months from this_month, skipping 202510 if encountered.
    for i in range(freq-1):
        yr_dex = int(this_month[:4])
        mn_dex = int(this_month[4:6]) - (i + 1)   # look back instead of forward
        # Normalize when the look-back crosses a year boundary (month <= 0).
        yr_dex += (mn_dex - 1) // 12
        mn_dex  = (mn_dex - 1) % 12 + 1
        prev_month = str(yr_dex) + str(mn_dex).zfill(2)

        # Skip 202510 if we encounter it (no real data for that month)
        if prev_month == "202510":
            continue
        
        month = month.merge(
            get_month_edu_dd(ID, prev_month, occ_subset),
            how = 'outer',
            on = ['OCC', 'recent_grad'],
            suffixes = (None, str(i+1))
        )
    
    # Calculate mean across all WTFINL columns
    month['WTFINL'] = month.filter(regex='^WTFINL').mean(axis=1, skipna=True)
    month = month[['OCC', 'WTFINL', 'recent_grad']]    

    month = month.pivot(
        index = 'OCC',
        columns = 'recent_grad',
        values = 'WTFINL'
    ).fillna(0).rename_axis(columns=None).reset_index()

    # calculate dissimilarity 
    young_all = month['young'].sum()
    old_all = month['old'].sum()
    month[this_month] = abs((month['old'] / old_all) - (month['young'] / young_all)) * 100 / 2

    out = month.agg({this_month:'sum'})

    return out


def get_month_edu_dd(ID: str, this_month: str, occ_subset: str):
    if this_month == "202510":
        sep = get_month_edu_dd(ID, "202509", occ_subset)
        nov = get_month_edu_dd(ID, "202511", occ_subset)
        out = pandas.concat([sep, nov], ignore_index = True).groupby(["OCC","recent_grad"]).agg({"WTFINL": 'mean'}).reset_index()


    else:
        full_month = pandas.read_parquet(
            os.path.join("/nfs/roberts/project/pi_nrs36/shared/model_data/AI-Employment-Model", ID, "cps_" + this_month + ".parquet"), engine = 'pyarrow'
        )
        
        # subset for college grads and define age groups
        full_month = full_month[full_month["OCC"]!="0000"]
        full_month = full_month[full_month["EDUC"]>=91]
        full_month = full_month[full_month["AGE"].between(20,34)]
        full_month['recent_grad'] = numpy.where(full_month['AGE']<25, 'young', 'old')
        count      = full_month["WTFINL"].sum()

        out = full_month.groupby(["OCC","recent_grad"]).agg({"WTFINL": 'sum'}).reset_index()

        # subset by occupation
        if occ_subset:
            occ_subset = occ_subset.split('.')
            out = out[(pandas.to_numeric(out['OCC']) >= int(occ_subset[0])) & (pandas.to_numeric(out['OCC']) <= int(occ_subset[1]))].reset_index()

    return out