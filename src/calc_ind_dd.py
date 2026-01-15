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

def build_mis_ind(ID: str, start: str, end: str, freq = 1):

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
            base = calc_mis_ind_month(ID, month, freq)
            months = base.groupby('Industry').agg({month:'sum'}).reset_index()
        else: 
            this_month = calc_mis_ind_month(ID, month, freq, base[["Industry", 'OCC', "base"]])
            this_month = this_month.groupby('Industry').agg({month:'sum'}).reset_index()
            months = months.merge(
                this_month,
                how = 'outer',
                on  = 'Industry'
            )        

        # Iterate Month/Year
        mn_dex += 1
        if mn_dex > 12:
            mn_dex = mn_dex % 12
            yr_dex += 1

    deltas = months.set_index('Industry').T.reset_index(names = 'MONTH').rename_axis(columns=None)

    deltas["months_gone"] = deltas.index
    deltas = deltas.loc[:, ['months_gone', 'Natural Resources and Mining', 'Construction', 'Manufacturing', 'Trade, Transportation, and Utilities', 'Information', 'Financial Activities', 'Professional and Business Services', 'Education and Health Services', 'Leisure and Hospitality', 'Other Services']]
        
    return deltas


def calc_mis_ind_month(ID: str, this_month: str, freq: int, base = None):
    
    month = get_month_ind_dd(ID, this_month)
    # calculate a freq-month moving average for occupation counts
    for i in range(freq-1):
        yr_dex = int(this_month[:4])
        mn_dex = int(this_month[4:6]) + i+1
        if mn_dex > 12:
            yr_dex += 1
            mn_dex = mn_dex % 12 
        next_month = str(yr_dex) + str(mn_dex).zfill(2)
        month = month.merge(
            get_month_ind_dd(ID, next_month),
            how = 'outer',
            on = ['OCC', 'Industry'],
            suffixes = (None, str(i+1))
        )
    month['WTFINL'] = month.filter(regex='^WTFINL').mean(axis=1)
    month = month[['OCC', 'WTFINL', 'Industry']]    

    if base is None:
        # If a base month isn't provided, it assumes it needs to calcualte one, 
        # and assigns both the current month and base to be this month
        base = month.rename(columns = {'WTFINL' : "base"})

    month = base.merge(
        month,
        how = 'outer',
        on = ['OCC', 'Industry'],
    )

    month['base_all'] = month['base'].groupby(month['Industry']).transform('sum')
    month['month_all'] = month['WTFINL'].groupby(month['Industry']).transform('sum')

    month[this_month] = abs((month['WTFINL'] / month['month_all']) - (month['base'] / month['base_all'])) * 100 / 2
    
    out = month[['OCC', 'Industry', 'base', this_month]]

    return out


def get_month_ind_dd(ID: str, this_month: str):

    if this_month == "202510":
        sep = get_month_ind_dd(ID, "202509")
        nov = get_month_ind_dd(ID, "202511")
        out = pandas.concat([sep, nov], ignore_index = True).groupby(["OCC","Industry"]).agg({"WTFINL": 'mean'}).reset_index()

    else:
        full_month = pandas.read_parquet(
            os.path.join("/gpfs/gibbs/project/sarin/shared/model_data/AI-Employment-Model/dissimilarity", ID, "cps_" + this_month + ".parquet"), engine = 'pyarrow'
        )
        
        full_month = full_month[full_month["AGE"] > 15]
        full_month = full_month[full_month["OCC"]!="0000"]

        full_month["IND"] = pandas.to_numeric(full_month['IND'])
        full_month['Industry'] = numpy.select(
            [
                full_month['IND'].ge(100) & full_month['IND'].le(560),
                full_month['IND'].ge(770) & full_month['IND'].le(1060),
                full_month['IND'].ge(1070) & full_month['IND'].le(4060),
                (full_month['IND'].ge(4070) & full_month['IND'].le(6390)) | (full_month['IND'].ge(570) & full_month['IND'].le(760)),
                full_month['IND'].ge(6470) & full_month['IND'].le(6860),
                full_month['IND'].ge(6870) & full_month['IND'].le(7260),
                full_month['IND'].ge(7270) & full_month['IND'].le(7790),
                full_month['IND'].ge(7860) & full_month['IND'].le(8470),
                full_month['IND'].ge(8560) & full_month['IND'].le(8690),
                full_month['IND'].ge(8770) & full_month['IND'].le(9290)
            ],
            ['Natural Resources and Mining', 'Construction', 'Manufacturing', 'Trade, Transportation, and Utilities', 'Information', 'Financial Activities', 'Professional and Business Services', 'Education and Health Services', 'Leisure and Hospitality', 'Other Services'],
            default=''
        )
        full_month = full_month[full_month['Industry']!='']
        
        out = full_month.groupby(["OCC","Industry"]).agg({"WTFINL": 'sum'}).reset_index()

    return out