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

def build_mismatch(ID: str, start: str, end: str, freq = 0):

    # Parse start and end dates into iterable format
    start = start.split('.')
    end   = end.split('.')

    yr_dex = int(start[0])
    mn_dex = int(start[1])
    
    end_yr = int(end[0])
    end_mn = int(end[1]) + 1

    if end_mn > 12:
        end_yr += 1
        end_mn = end_mn % 12 

    # Loop through each month of a given time period
    while((yr_dex != end_yr) or (mn_dex != end_mn)):
        # Combine current month and year to get data stamp
        month = str(yr_dex) + str(mn_dex).zfill(2)

        # Begin calculating mismatch
        if "months" not in locals():
            # Get baseline month against which future months will be compared
            months = calc_mis_month(ID, month)
        else: 
            # Calculated mismatch for a new month and attach to the running count
            # At this point, months (DataFrame) is long in occupations and wide in dissimilarity by month
            this_month = calc_mis_month(ID, month, months[["OCC", "base"]])
            months = months.merge(
                this_month[["OCC", month]],
                how = 'left',
                on  = 'OCC' 
            )
        # Iterate Month/Year
        mn_dex += 1
        if mn_dex > 12:
            mn_dex = mn_dex % 12
            yr_dex += 1

    # With all months calculated, we tidy up the dataframe a bit and aggregate across occupations
    # The 'sum(axis=0)' sums each column to get the month's dissimilarity
    # On the 'loc[1:,:]' line, switch the '1' to a '2' if you want to drop the base month with dissimilarity = -
    data = months.drop(["OCC", "base"], axis=1).sum(axis=0).reset_index().rename(
        columns = {
            "index": "MONTH",
            0: "dissimilarity"
        }
    ).loc[1:,:]
    print(data)
    
    # Calculate 'freq' month moving average as necessary
    # If 0, skip calculation
    deltas = pandas.Series()
    
    if freq == 0:
        deltas = data     
    else:
        # Select months 'freq' months after the start date
        months = data["MONTH"].unique()[freq:]
        for month in months:
            # Calculate each month's 'freq' monthly moving average for 'freq' preceeding months
            curr = calc_avg2(data, month, freq, groups=None)
            deltas = pandas.concat([deltas, pandas.Series(curr)])
    
        deltas = DataFrame({"MONTH": months, "dissimilarity": deltas})
    
    deltas = deltas.assign(MONTH = lambda x: to_datetime(x["MONTH"], format='%Y%M'))

    # This graph is mainly for testing and is not useful for an actual write up
    locator = mdates.YearLocator()
    fmt     = mdates.DateFormatter("%Y")

    sns.lineplot(
        data = deltas,
        x = "MONTH",
        y = "dissimilarity"
    )
    plt.ylim((0,10))
    X = plt.gca().xaxis
    X.set_major_locator(locator)
    X.set_major_formatter(fmt)
    plt.xlabel("Time")
    plt.show()
    
    # Prep for output
    # Months_gone is a useful metric for graphing multiple timespans on the same chart, with it as the x-axis
    deltas = deltas.reset_index()
    deltas["months_gone"] = deltas.index +1
    deltas = deltas.drop(['index', "MONTH"], axis=1)
    
    return deltas

def calc_mis_month(ID: str, this_month: str, base = None):
    # This funciton calculates the dissimilarity between the same occupations across two months

    if base is None:
        # If a base month isn't provided, it assumes it needs to calcualte one, 
        # and assigns both the current month and base to be this month
        base = get_month_dd(ID, this_month)
        month = base
        base = base.rename(columns = {'WTFINL' : "base"})
    else:
        # Otherwise, it just fetches the current month
        month = get_month_dd(ID, this_month)
    
    # Attaches the current month to the base month by occupation
    month = base.merge(
        month,
        how = 'left',
        on = 'OCC'
    )
    
    base_all = month['base'].sum()
    month_all = month['WTFINL'].sum()
    # Calculates the dissimilarity between the occupations
    # This is measured as the absolute difference of an occupation's percent composition of the overall workforce
    # The 100 puts it in the percentage point terms and the /2 is part of the given methodology: https://www.hiringlab.org/wp-content/uploads/2018/09/Mismatch-methodology.pdf
    month[this_month] = abs((month['WTFINL'] / month_all) - (month['base'] / base_all)) * 100 / 2
    
    return month


def get_month_dd(ID: str, this_month: str):
    # This function simply reads in data, filters it to the population we want (working adults), 
    # and sums weights for each occupation

    full_month = pandas.read_parquet(
        os.path.join(os.path.dirname(__file__), "../cps/processed", ID, "cps_" + this_month + ".parquet"), engine = 'pyarrow'
    )
    
    full_month = full_month[full_month["AGE"] > 15]
    full_month = full_month[full_month["OCC"]!="0000"]
    count      = full_month["WTFINL"].sum()
    
    out = full_month.groupby("OCC").agg({"WTFINL": 'sum'}).reset_index()

    return out
