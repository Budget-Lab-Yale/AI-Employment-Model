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

def build_mis_edu(ID: str, start: str, end: str, freq = 1, occ_subset = ''):

    # Parse start and end dates into iterable format
    start = start.split('.')
    end   = end.split('.')

    yr_dex = int(start[0])
    mn_dex = int(start[1])
    
    end_yr = int(end[0])
    # -(freq-1) to account for the moving average and +1 to make exclusive
    end_mn = int(end[1]) - freq + 2
    if (end_mn > 12):
        end_yr += 1
        end_mn = end_mn % 12
    elif (end_mn < 1):
        end_yr -= 1
        end_mn = 12 + end_mn

    # Loop through each month of a given time period
    while((yr_dex != end_yr) or (mn_dex != end_mn)):
        # Combine current month and year to get data stamp
        month = str(yr_dex) + str(mn_dex).zfill(2)

        if "months" not in locals():
            months = calc_mis_edu_month(ID, month, freq, occ_subset)
        else: 
            months = pandas.concat([months,calc_mis_edu_month(ID,month,freq,occ_subset)])

        
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
    print(data)
    
    deltas = pandas.Series()
    deltas = data     
        
    deltas = deltas.assign(MONTH = lambda x: to_datetime(x["MONTH"], format='%Y%m'))

    # This graph is mainly for testing and is not useful for an actual write up
    locator = mdates.YearLocator()
    fmt     = mdates.DateFormatter("%Y")

    sns.lineplot(
        data = deltas,
        x = "MONTH",
        y = "dissimilarity"
    )
    X = plt.gca().xaxis
    X.set_major_locator(locator)
    X.set_major_formatter(fmt)
    plt.xlabel("Time")
    plt.show()
        
    return deltas


def calc_mis_edu_month(ID: str, this_month: str, freq: int, occ_subset: str):
    
    month = get_month_edu_dd(ID, this_month, occ_subset)
    # calculate a freq-month moving average for occupation counts
    for i in range(freq-1):
        yr_dex = int(this_month[:4])
        mn_dex = int(this_month[4:6]) + i+1
        if mn_dex > 12:
            yr_dex += 1
            mn_dex = mn_dex % 12 
        next_month = str(yr_dex) + str(mn_dex).zfill(2)
        month = month.merge(
            get_month_edu_dd(ID, next_month, occ_subset),
            how = 'outer',
            on = ['OCC', 'recent_grad'],
            suffixes = (None, str(i+1))
        )
    month['WTFINL'] = month.filter(regex='^WTFINL').mean(axis=1)
    month = month[['OCC', 'WTFINL', 'recent_grad']]    

    month = month.pivot(
        index = 'OCC',
        columns = 'recent_grad',
        values = 'WTFINL'
    ).fillna(0).rename_axis(columns=None).reset_index()

    # calculate dissimiarlity 
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
            os.path.join("/gpfs/gibbs/project/sarin/shared/model_data/AI-Employment-Model", ID, "cps_" + this_month + ".parquet"), engine = 'pyarrow'
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