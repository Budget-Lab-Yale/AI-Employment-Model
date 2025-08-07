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

from pandas import read_csv, DataFrame
from parsing import month_helper
from datetime import date

def build_std_output(ID: str, start: str, end: str, metric = "mean_rating_human_beta", freq = 3):

    locator = mdates.YearLocator()
    fmt     = mdates.DateFormatter("%Y")

    # This is incredibly inefficient but is easier to understand for humans
    # When doing 3mma or less, runtime is less important. Is a HUGE pain for 12mma
    # See calc_other.py for uglier but more efficient code
    for g in ["EMPSTAT", "DURUNEMP_cat", "SEX", "major_occ", "AGE"]:
        # Iterate over grouping variablese

        if g == "DURUNEMP_cat":
            # For Duration of Unemployment grouping, there are a few codes (basically employed and NILF) that we don't want in our output
            # Extra verbosity here notes what to filter out
            block = calc_block(ID, start, end, freq = freq, group = g, filter_group = "DURUNEMP", filter_conditions = [0,999])
        elif g == "EMPSTAT":
            # Post-processing, we only want employed people, so we manually filter them out
            block = calc_block(ID, start, end, freq = freq, group = g)
        elif g == "AGE":
            # Here, exposure is our grouping variable while AGE is our output variable.
            # Verbosity notes the "axis" switch
            block = calc_block(ID, start, end, freq = freq, group = g, exposure_cat = True)
        else:
            # Anything else is run normally (currently just SEX)
            block = calc_block(ID, start, end, freq = freq, group = g)

        # Make sure output looks sensible (YYYYMM, group subcode, exposure)
        print(block.head())


        # Basic seaborn / matplot plotting (nothing fancy, just for use in analytics)
        # Is better to output data to something like PowerBI or excel to make prettier graphics
        # But as "engineers", it's fine to have ugly stuff to squint at
        plt.figure()

        if g == "AGE":
            sns.lineplot(
            data = block,
            x   = "MONTH",
            y   = "exposure",
            hue = "exposed"
            )    
            plt.ylabel("Average Age")
        else:
            sns.lineplot(
                data = block,
                x   = "MONTH",
                y   = "exposure",
                hue = g
            )
            plt.ylabel("Percent Exposure")
        plt.ylim((0,100))
        X = plt.gca().xaxis
        X.set_major_locator(locator)
        X.set_major_formatter(fmt)
        plt.xlabel("Time")
        plt.show()       
    

def calc_block(ID: str, start: str, end: str, freq = 3, group = None, metric = "mean_rating_human_beta", filter_group = None, filter_conditions = None, exposure_cat = False):
    
    # Collapse microdata into (YYYYMM, group subcode, exposure) format with no mma
    data = collapse_months(ID, start, end, group, metric, filter_group, filter_conditions, exposure_cat)
    
    # Early exit if we don't need to calculate mma
    if freq == 1:
        return data.rename(columns = {"tasks_exposed_pct": "exposure"})

    # Gets a list of every month of data beginning FREQ months from the start period
    months = data["MONTH"].unique()[freq-1:]
    
    # Calculate the FREQ mma for the period ENDING in this_month
    for month in months:
        curr = calc_avg(data, month, freq, group, exposure_cat)

        if "out" not in locals():
            out = curr
        else:
            out = pandas.concat([out, curr], ignore_index = True, sort = False)
    
    return out

def collapse_months(ID: str, start: str, end: str, group: str, metric: str, filter_group: str, filter_conditions, exposure_cat: bool):

    # Same year/month iteration process as before
    start = start.split('.')
    end   = end.split('.')

    yr_dex = int(start[0])
    mn_dex = int(start[1])
    
    end_yr = int(end[0])
    end_mn = int(end[1]) + 1
    if end_mn > 12:
        end_yr += 1
        end_mn = end_mn % 12 

    while((yr_dex != end_yr) or (mn_dex != end_mn)):
        month = str(yr_dex) + str(mn_dex).zfill(2)

        # Collapses this_month into grouped summary
        curr = get_month(ID, month, group, metric, filter_group, filter_conditions, exposure_cat)
        
        if "collapsed" not in locals():
            collapsed = curr
        else:
            collapsed = pandas.concat([collapsed, curr], ignore_index = True, sort = False)

        # Month/year iteration
        mn_dex += 1
        if mn_dex > 12:
            mn_dex = mn_dex % 12
            yr_dex += 1
    
    print(collapsed.head())

    return collapsed



def calc_avg(data: DataFrame, this_month: str, freq: int, group: str, exposure_cat: bool):  

    months = []
    # Here we go from our starting month and go backwards in time FREQ periods
    for i in range(0, freq):
        mn_dex = int(this_month[4:]) - i
        yr_dex = int(this_month[:4])
        if mn_dex < 1:
            mn_dex = month_helper(mn_dex)
            yr_dex -= 1
        month = str(yr_dex) + str(mn_dex).zfill(2)
        months.append(month)

    # Filter to collapsed months just in our mma range
    # These are in the format [YYYYMM, group, metric]
    out = data[data["MONTH"].isin(months)]
    out = out.assign(MONTH = lambda x: x["MONTH"].astype(str))

    # Pivot data wider so that rows are subgroups (employed, unemployed, NILF etc), columns are months, and column values are exposure
    # Then calculate average exposure across months and drop the unaggregated months

    # Slightly different method for 
    if exposure_cat:
        out = out.pivot(index= ["exposed"], columns = "MONTH", values = ("avg_" + group)).assign(
            exposure = lambda x: x[months].mean(axis=1)
        ).drop(months, axis=1).reset_index()

    else:
        out = out.pivot(index= [group], columns = "MONTH", values = "tasks_exposed_pct").assign(
            exposure = lambda x: x[months].mean(axis=1)
        ).drop(months, axis=1).reset_index()
    
    # Insert 0th column of YYYYMM of this_month repeated n(subgroups) times
    out.insert(0, "MONTH", date(int(this_month[:4]), int(this_month[4:]),1))
    
    return out


def get_month(ID: str, this_month: str, group: str, metric: str, filter_group: str, filter_conditions, exposure_cat: bool):

    # Read in microdata
    month = pandas.read_parquet(
        os.path.join(os.path.dirname(__file__), "../cps/processed", ID, "cps_" + this_month + ".parquet"), engine = 'pyarrow'
    )

    # Want "working age" adults and people in CPS OCC universe
    # (Civilians age 16+ who were employed, on layoff, unemployed but had worked in the past, or not in labor force but had worked in the past year)
    # Good universe for what we're trying to measure here
    month = month[month["AGE"] > 15]
    month = month[month["OCC"]!="0000"]
    
    if group == "DURUNEMP_cat":
        month = month[month["EMPSTAT"]!="1"]
    else:
        month = month[month["EMPSTAT"]=="1"]

    # MAKE IT JUST EMPLOYED PEOPLE UNLESS DOING DURUNEMP

    # Apply additional filter conditions if they exist
    if filter_conditions is not None:
        for c in filter_conditions:
            month = month[month[filter_group] != c]

    if exposure_cat:
        # Exposure here is a binary grouping variable
        month = month.assign(
            exposed = lambda x: numpy.where(x[metric] > 0, True, False)
        )

        # Aggregate weighted average of output metric by exposure binary
        month = month.groupby("exposed").apply(lambda x: numpy.average(x[group], weights=x["WTFINL"]))
        month  = month.to_frame().reset_index()
        month.insert(0, "MONTH", [this_month] * len(month))
        month = month.rename(columns = {0: "avg_" + group})

    else:
        # Aggregated weighted average exposure (pct level, not decimal)
        month = month.groupby(group).apply(lambda x: numpy.sum(x[metric] * x["WTFINL"]) * 100) / month.groupby(group)["WTFINL"].sum() # IMPORTANT TO MAKE SURE I'M RIGHT OR NOT MARTHA APPROVED 2025041414
        month  = month.to_frame().reset_index()
        month.insert(0, "MONTH", [this_month] * len(month))
        month = month.rename(columns = {0: "tasks_exposed_pct"})
    
    return month

def do_benchmark(metric: str):

    # This was important early, don't worry about it now
    cps = pandas.read_parquet(
        os.path.join(os.path.dirname(__file__), "../cps/processed/benchmark/cps_benchmark.parquet"), engine = 'pyarrow'
    )

    bench = read_csv(os.path.join(os.path.dirname(__file__), "../resources/brookings_benchmark.csv"))

    major = cps.groupby("major_occ").apply(lambda x: numpy.sum(x[metric] * x["ASECWT"])) * 100 / cps.groupby("major_occ")["ASECWT"].sum()
    
    major = major.to_frame().reset_index().rename(columns = {0: "tasks_exposed_pct"}).sort_values(by = "tasks_exposed_pct", ascending = False).merge(
        bench,
        how = 'left',
        on = 'major_occ'
    ).assign(
        gap = lambda x: x["tasks_exposed_pct"] - x["brookings"]
    )

    sns.barplot(major, x = 'gap', y = 'major_occ', hue = 'major_occ', legend = False)

    return