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

def build_other_output(ID: str, start: str, end: str, out_path: str, metric = "mean_rating_human_beta", freq = 3):
    locator = mdates.YearLocator()
    fmt     = mdates.DateFormatter("%Y")
    this_dir = os.path.join(out_path, metric)
    os.makedirs(this_dir)
    
    # Here we minimize number of I/O operations rather than reducing frame complexity
    # As such, subfunctions are significantly more complex, but it runs (a bit) faster
    # We also pack output so that the grouping categories can remain seperate
    # Generally speaking, objects that include "_e" or "e_" use exposure as a grouping characteristic rather than as an output metric
    block, e_cat_block = calc_block2(ID = ID, start = start, end = end, groups = ["EMPSTAT", "DURUNEMP_cat", "SEX"], e_vars = ["AGE"], freq = freq,  metric = metric) 


    block.to_csv(os.path.join(this_dir, "exposure.csv"), index = False)


    # cleaning for comparison of OpenAI to Anthropic measures
    full_month = pandas.read_parquet(
        os.path.join(os.path.dirname(__file__), "../cps/processed", ID, "cps_" + '202301' + ".parquet"), engine = 'pyarrow'
    )

    full_month = full_month[full_month["AGE"] > 15]
    full_month = full_month[full_month["OCC"]!="0000"]
    check = full_month.groupby("OCC").agg({metric:'mean', 'filtered':'mean', 'total':'mean', 'pct_of_convs':'mean'}).reset_index()

    check['not_filtered'] = check['total'] - check['filtered']
    check['not_filtered_pct'] = check['not_filtered'] / check['total']

    socs = read_csv("/gpfs/gibbs/project/sarin/jmk263/Repositories/AI-Employment-Model/resources/crosswalk.csv").dropna(subset = 'cps_code').assign(
        # Processing values to facilitate easier merging later on
        cps_code  = lambda x: x['cps_code'].astype(int).astype(str).str.zfill(4),
        major_code = lambda x: x["Code"].str[:2].astype(int),
    )
    oes = read_csv('../resources/national_M2024_dl.csv')[['OCC_CODE', 'TOT_EMP', 'OCC_TITLE', 'O_GROUP']]
    oes['Code'] = oes['OCC_CODE'] + '.00'
    socs = socs.merge(
        oes,
        how = 'left',
        on = 'Code'
    )
    socs = socs.groupby(['cps_code', 'major_code']).agg({'TOT_EMP':'sum'}).reset_index()

    majors = read_csv("/gpfs/gibbs/project/sarin/jmk263/Repositories/AI-Employment-Model/resources/soc_major.csv")
    socs = socs.merge(
        majors,
        how = "left",
        on = "major_code"
    )
    check = check.merge(
        socs,
        how = "left",
        left_on = "OCC",
        right_on = "cps_code"
    ).assign(
        quadrant = lambda x: numpy.where(
            x[metric] >= .5,
            numpy.where(x['not_filtered_pct'] >= .5, "Q1", "Q4"),
            numpy.where(x['not_filtered_pct'] >= .5, "Q2", "Q3"),
        )
    )

    check.to_csv(os.path.join(this_dir, "openai_vs_anthropic.csv"), index = False)


    check['weighted_exposure'] = check[metric] * check['TOT_EMP']

    def get_wp(var, total, weights):
        # Weighted sum of the metric at hand
        w_sum = (var * weights).sum()
        # Total weight (magnified by n(tasks) for each occupation in the usage data)
        w_total = (weights * total * ~var.isna()).sum()

        return (w_sum / w_total) * 100 if w_total > 0 else 0

    out = check.groupby('major_occ').agg(
        # Weighted exposure (just give the total var in get_wp() a vector of 1s)
        average_exposure = pandas.NamedAgg(
            column = metric,
            aggfunc = lambda x, w=check['TOT_EMP']: get_wp(x, 1, w.loc[x.index])
        ),
        # THIS IS WEIGHTED BY EMPLOYMENT SHARE
        average_usage = pandas.NamedAgg(
            column = 'not_filtered_pct',
            aggfunc = lambda x, w=check['TOT_EMP']: get_wp(x, 1, w.loc[x.index])
        ),
        weighted_exposure = pandas.NamedAgg(
            column = 'weighted_exposure',
            aggfunc = 'sum'
        ),
        # not_filtered = pandas.NamedAgg(
        #     column = 'not_filtered',
        #     aggfunc = 'sum'
        # ),
        # total = pandas.NamedAgg(
        #     column = 'total',
        #     aggfunc = 'sum'
        # ),
        observed_usage = pandas.NamedAgg(
            column = 'pct_of_convs',
            aggfunc = 'sum'
        ),
        TOT_EMP = pandas.NamedAgg(
            column = 'TOT_EMP',
            aggfunc = 'sum'
        )
    ).reset_index()
    out['emp_share'] = out['TOT_EMP']/out['TOT_EMP'].sum() * 100
    out['observed_usage'] = out['observed_usage']/out['observed_usage'].sum() * 100
    out['expected_usage'] = out['weighted_exposure']/out['weighted_exposure'].sum() * 100
    # out['observed_usage'] = out['total']/out['total'].sum() * 100
    # out['observed_usage_nf'] = (out['total'] - out['not_filtered'])/(out['total'] - out['not_filtered']).sum() * 100

    # out = out[['major_occ', 'average_exposure', 'average_usage', 'emp_share', 'expected_usage', 'observed_usage', 'observed_usage_nf']]
    out = out[['major_occ', 'average_exposure', 'average_usage', 'emp_share', 'expected_usage', 'observed_usage']]
    out.to_csv(os.path.join(this_dir, "molly.csv"), index = False)


def calc_block2(ID: str, start: str, end: str, groups, e_vars, freq = 3, metric = "mean_rating_human_beta"):
    
    # Collapse microdata into aggregated no mma monthly data
    data, data_e = collapse_months2(ID, start, end, groups, metric, e_vars)
    
    if freq == 1:
        return data, data_e

    months = data["MONTH"].unique()[freq-1:]    
    for month in months:
        curr = calc_avg2(data, month, freq, groups=groups)
        curr_e = calc_avg2(data_e, month, freq, groups=e_vars)
        if "out" not in locals():
            out = curr
        else:
            out = pandas.concat([out, curr], ignore_index = True, sort = False)
        if "out_e" not in locals():
            out_e = curr_e
        else:
            out_e = pandas.concat([out_e, curr_e], ignore_index = True, sort = False)
    
    return out, out_e


def collapse_months2(ID: str, start: str, end: str, groups, metric: str, e_vars):
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
        #if exposure:
        #    curr = get_month_exposure_cat(ID, month, groups, metric)
        #else:
        #    curr = get_month2(ID, month, groups, metric)   
        
        curr, curr_e = get_month(ID, month, groups, e_vars, metric)

        if "collapsed" not in locals():
            collapsed = curr
        else:
            collapsed = pandas.concat([collapsed, curr], ignore_index = True, sort = False)

        if "collapsed_e" not in locals():
            collapsed_e = curr_e
        else:
            collapsed_e = pandas.concat([collapsed_e, curr_e], ignore_index = True, sort = False)

        mn_dex += 1
        if mn_dex > 12:
            mn_dex = mn_dex % 12
            yr_dex += 1
    
    return collapsed, collapsed_e

def calc_avg2(data: DataFrame, this_month: str, freq: int, groups):  
    
    months = []
    for i in range(0, freq):
        mn_dex = int(this_month[4:]) - i
        yr_dex = int(this_month[:4])
        if mn_dex < 1:
            mn_dex = month_helper(mn_dex)
            yr_dex -= 1
        month = str(yr_dex) + str(mn_dex).zfill(2)
        months.append(month)

    these_months = data[data["MONTH"].isin(months)].drop(["MONTH"], axis=1)
    
    if groups is None:
        return these_months.agg('mean')
    else:
        these_months = these_months.groupby(["group", "group_val"], as_index=True).agg('mean')
        these_months.reset_index(inplace=True)
        these_months.insert(0, "MONTH", date(int(this_month[:4]), int(this_month[4:]),1))

    return these_months

def get_month(ID: str, this_month: str, groups, e_vars, metric: str):
    full_month = pandas.read_parquet(
        os.path.join(os.path.dirname(__file__), "../cps/processed", ID, "cps_" + this_month + ".parquet"), engine = 'pyarrow'
    )

    full_month = full_month.assign(
        # exposed = lambda x: numpy.where(x[metric] > 0, 1, 0),
        # highly_exposed = lambda x: numpy.where(x[metric] > .75, 1, 0),
        mildly_exposed = lambda x: numpy.where(numpy.logical_and(x[metric] > 0, x[metric] < .4), 1, 0),
        moderately_exposed = lambda x: numpy.where(numpy.logical_and(x[metric] >= .4, x[metric] < .8), 1, 0),
        highly_exposed = lambda x: numpy.where(x[metric] >= .8, 1, 0),
        lowest_exposed = lambda x: numpy.where(numpy.logical_and(x[metric] > 0, x[metric] <= 0.2687269), 1, 0),
        middle_exposed = lambda x: numpy.where(numpy.logical_and(x[metric] > 0.2687269, x[metric] <= 0.5307864), 1, 0),
        highest_exposed = lambda x: numpy.where(x[metric] > 0.5307864, 1, 0),
        automated = lambda x: numpy.where(x['automation']/x['total'] > .5, 1, 0),
        augmented = lambda x: numpy.where(x['augmentation']/x['total'] > .5, 1, 0)
    )
    
    # Want "working age" adults and people in CPS OCC universe
    # (Civilians age 15+ who were employed, on layoff, unemployed but had worked in the past, or not in labor force but had worked in the past year)
    # Good universe for what we're trying to measure here
    full_month = full_month[full_month["AGE"] > 15]
    full_month = full_month[full_month["OCC"]!="0000"]
    
    for g in groups:  
        # First, we define a local aggregation function to get weighted percents (where we also normalize the usage metrics)
        def get_wp(var, total, weights):
            # Weighted sum of the metric at hand
            w_sum = (var * weights).sum()
            # Total weight (magnified by n(tasks) for each occupation in the usage data)
            w_total = (weights * total * ~var.isna()).sum()

            return (w_sum / w_total) * 100 if w_total > 0 else 0

        # Then we apply bespoke filtering conditions
        if g == "DURUNEMP_cat":
            month = full_month[full_month["EMPSTAT"]!="1"]
            month = month[~month["DURUNEMP"].isin([0,999])]

        else:
            month = full_month[full_month["EMPSTAT"]=="1"]

        # Aggregate micro data into:
        temp = month.groupby(g).agg(
            # Weighted exposure (just give the total var in get_wp() a vector of 1s)
            tasks_exposed_pct = pandas.NamedAgg( # This should be identical to the output of calc.py line 220
                column = metric,
                aggfunc = lambda x, w=month['WTFINL']: get_wp(x, 1, w.loc[x.index])
            ),

            # percent_exposed = pandas.NamedAgg( 
            #     column = 'exposed',
            #     aggfunc = lambda x, w=month['WTFINL']: get_wp(x, 1, w.loc[x.index])
            # ),

            # percent_highly_exposed = pandas.NamedAgg( 
            #     column = 'highly_exposed',
            #     aggfunc = lambda x, w=month['WTFINL']: get_wp(x, 1, w.loc[x.index])
            # ),
            percent_mildly_exposed = pandas.NamedAgg( 
                column = 'mildly_exposed',
                aggfunc = lambda x, w=month['WTFINL']: get_wp(x, 1, w.loc[x.index])
            ),

            percent_moderately_exposed = pandas.NamedAgg( 
                column = 'moderately_exposed',
                aggfunc = lambda x, w=month['WTFINL']: get_wp(x, 1, w.loc[x.index])
            ),

            percent_highly_exposed = pandas.NamedAgg( 
                column = 'highly_exposed',
                aggfunc = lambda x, w=month['WTFINL']: get_wp(x, 1, w.loc[x.index])
            ),

            percent_lowest_exposed = pandas.NamedAgg( 
                column = 'lowest_exposed',
                aggfunc = lambda x, w=month['WTFINL']: get_wp(x, 1, w.loc[x.index])
            ),

            percent_middle_exposed = pandas.NamedAgg( 
                column = 'middle_exposed',
                aggfunc = lambda x, w=month['WTFINL']: get_wp(x, 1, w.loc[x.index])
            ),

            percent_highest_exposed = pandas.NamedAgg( 
                column = 'highest_exposed',
                aggfunc = lambda x, w=month['WTFINL']: get_wp(x, 1, w.loc[x.index])
            ),


            # Percent of conversations indicating task automation
            automation = pandas.NamedAgg(
                column = 'automation',
                aggfunc = lambda x, t=month['total'], w=month['WTFINL']: get_wp(x, t.loc[x.index], w.loc[x.index])
            ),
            # Percent of conversations indicating task augmentation
            augmentation = pandas.NamedAgg(
                column = 'augmentation',
                aggfunc = lambda x, t=month['total'], w=month['WTFINL']: get_wp(x, t.loc[x.index], w.loc[x.index])
            ),
            # Percent of conversations where task is filtered (make sure you know what the filtering is TODO)
            filtered = pandas.NamedAgg(
                column = 'filtered',
                aggfunc = lambda x, t=month['total'], w=month['WTFINL']: get_wp(x, t.loc[x.index], w.loc[x.index])
            ),

            # percent of occupations that are automated/augmented
            percent_automated = pandas.NamedAgg( 
                column = 'automated',
                aggfunc = lambda x, w=month['WTFINL']: get_wp(x, 1, w.loc[x.index])
            ),
            percent_augmented = pandas.NamedAgg( 
                column = 'augmented',
                aggfunc = lambda x, w=month['WTFINL']: get_wp(x, 1, w.loc[x.index])
            )

        ).reset_index().rename(columns = {g: "group_val"}).assign(group = g)
        
        # Just get the vars we need
        temp = temp[['group', 'group_val', 'tasks_exposed_pct', 'percent_lowest_exposed', 'percent_middle_exposed', 'percent_highest_exposed', 'percent_mildly_exposed', 'percent_moderately_exposed', 'percent_highly_exposed', 'automation', 'augmentation', 'filtered', 'percent_automated', 'percent_augmented']]
        
        if "out" not in locals():
            out = temp
        else:
            out = pandas.concat([out, temp], ignore_index = True, sort = False)
    
    out.insert(0, "MONTH", [this_month] * len(out))

    del temp

    for v in e_vars:
        month = full_month[full_month["EMPSTAT"]=="1"]

        for e in ["lowest_exposed", "middle_exposed", "highest_exposed", 'mildly_exposed', 'moderately_exposed', 'highly_exposed']:
            this = month.groupby(e).agg(
                    var = pandas.NamedAgg(
                        column = v,
                        aggfunc = lambda x: numpy.average(x, weights = month.loc[x.index, "WTFINL"])
                    )
            ).reset_index().rename(columns = {e: "group_val"}).assign(group = e)  
            
            if "temp" not in locals():
                temp = this
            else:
                temp = pandas.concat([temp, this], ignore_index = True, sort = False)
        temp = temp.rename(columns = {"var": v}).assign(
            group_val = lambda x: numpy.where(x["group_val"]==0, False, True)
        )

        if "out_e" not in locals():
            out_e = temp
        else:
            out_e = out_e.merge(temp)

    out_e.insert(0, "MONTH", [this_month] * len(out_e))

    return out, out_e


def get_month2(ID: str, this_month: str, groups, metric: str):
    full_month = pandas.read_parquet(
        os.path.join(os.path.dirname(__file__), "../cps/processed", ID, "cps_" + this_month + ".parquet"), engine = 'pyarrow'
    )
    
    full_month = full_month.assign(
        # exposed = lambda x: numpy.where(x[metric] > 0, 1, 0),
        # highly_exposed = lambda x: numpy.where(x[metric] > .75, 1, 0),
        mildly_exposed = lambda x: numpy.where(numpy.logical_and(x[metric] > 0, x[metric] < .4), 1, 0),
        moderately_exposed = lambda x: numpy.where(numpy.logical_and(x[metric] >= .4, x[metric] < .8), 1, 0),
        highly_exposed = lambda x: numpy.where(x[metric] >= .8, 1, 0),
        lowest_exposed = lambda x: numpy.where(numpy.logical_and(x[metric] > 0, x[metric] <= 0.26801243), 1, 0),
        middle_exposed = lambda x: numpy.where(numpy.logical_and(x[metric] > 0.2680124, x[metric] <= 0.5333915), 1, 0),
        highest_exposed = lambda x: numpy.where(x[metric] > 0.5333915, 1, 0),
    )
    
    # Want "working age" adults and people in CPS OCC universe
    # (Civilians age 15+ who were employed, on layoff, unemployed but had worked in the past, or not in labor force but had worked in the past year)
    # Good universe for what we're trying to measure here
    full_month = full_month[full_month["AGE"] > 15]
    full_month = full_month[full_month["OCC"]!="0000"]
    
    for g in groups:  

        # First, we define a local aggregation function to get weighted percents (where we also normalize the usage metrics)
        def get_wp(var, total, weights):
            # Weighted sum of the metric at hand
            w_sum = (var * weights).sum()
            # Total weight (magnified by n(tasks) for each occupation in the usage data)
            w_total = (weights * total * ~var.isna()).sum()

        return (w_sum / w_total) * 100 if w_total > 0 else 0
        
        # Then we apply bespoke filtering conditions
        if g == "DURUNEMP_cat":
            month = full_month[full_month["EMPSTAT"]!="1"]
            month = month[~month["DURUNEMP"].isin([0,999])]

        else:
            month = full_month[full_month["EMPSTAT"]=="1"]

        # Aggregate micro data into:
        temp = month.groupby(g).agg(
            # Weighted exposure (just give the total var in get_wp() a vector of 1s)
            tasks_exposed_pct = pandas.NamedAgg( # This should be identical to the output of calc.py line 220
                column = metric,
                aggfunc = lambda x, w=month['WTFINL']: get_wp(x, 1, w.loc[x.index])
            ),

            # percent_exposed = pandas.NamedAgg( 
            #     column = 'exposed',
            #     aggfunc = lambda x, w=month['WTFINL']: get_wp(x, 1, w.loc[x.index])
            # ),

            # percent_highly_exposed = pandas.NamedAgg( 
            #     column = 'highly_exposed',
            #     aggfunc = lambda x, w=month['WTFINL']: get_wp(x, 1, w.loc[x.index])
            # ),
            
            percent_mildly_exposed = pandas.NamedAgg( 
                column = 'mildly_exposed',
                aggfunc = lambda x, w=month['WTFINL']: get_wp(x, 1, w.loc[x.index])
            ),

            percent_moderately_exposed = pandas.NamedAgg( 
                column = 'moderately_exposed',
                aggfunc = lambda x, w=month['WTFINL']: get_wp(x, 1, w.loc[x.index])
            ),

            percent_highly_exposed = pandas.NamedAgg( 
                column = 'highly_exposed',
                aggfunc = lambda x, w=month['WTFINL']: get_wp(x, 1, w.loc[x.index])
            ),


            percent_lowest_exposed = pandas.NamedAgg( 
                column = 'lowest_exposed',
                aggfunc = lambda x, w=month['WTFINL']: get_wp(x, 1, w.loc[x.index])
            ),

            percent_middle_exposed = pandas.NamedAgg( 
                column = 'middle_exposed',
                aggfunc = lambda x, w=month['WTFINL']: get_wp(x, 1, w.loc[x.index])
            ),

            percent_highest_exposed = pandas.NamedAgg( 
                column = 'highest_exposed',
                aggfunc = lambda x, w=month['WTFINL']: get_wp(x, 1, w.loc[x.index])
            ),


            # Percent of conversations indicating task automation
            automation = pandas.NamedAgg(
                column = 'automation',
                aggfunc = lambda x, t=month['total'], w=month['WTFINL']: get_wp(x, t.loc[x.index], w.loc[x.index])
            ),
            # Percent of conversations indicating task augmentation
            augmentation = pandas.NamedAgg(
                column = 'augmentation',
                aggfunc = lambda x, t=month['total'], w=month['WTFINL']: get_wp(x, t.loc[x.index], w.loc[x.index])
            ),
            # Percent of conversations where task is filtered (make sure you know what the filtering is TODO)
            filtered = pandas.NamedAgg(
                column = 'filtered',
                aggfunc = lambda x, t=month['total'], w=month['WTFINL']: get_wp(x, t.loc[x.index], w.loc[x.index])
            )
        ).reset_index().rename(columns = {g: "group_val"}).assign(group = g)
        
        # Just get the vars we need
        temp = temp[['group', 'group_val', 'tasks_exposed_pct', 'percent_lowest_exposed', 'percent_middle_exposed', 'percent_highest_exposed', 'percent_mildly_exposed', 'percent_moderately_exposed', 'percent_highly_exposed', 'automation', 'augmentation', 'filtered']]
        
        if "out" not in locals():
            out = temp
        else:
            out = pandas.concat([out, temp], ignore_index = True, sort = False)
    
    out.insert(0, "MONTH", [this_month] * len(out))
    
    return out

def get_month_exposure_cat(ID: str, this_month: str, vars, metric: str):
    full_month = pandas.read_parquet(
        os.path.join(os.path.dirname(__file__), "../cps/processed", ID, "cps_" + this_month + ".parquet"), engine = 'pyarrow'
    )
    
    full_month = full_month.assign(
        exposed = lambda x: numpy.where(x[metric] > 0, 1, 0),
        highly_exposed = lambda x: numpy.where(x[metric] > .75, 1, 0)
    )
    
    # Want "working age" adults and people in CPS OCC universe
    # (Civilians age 15+ who were employed, on layoff, unemployed but had worked in the past, or not in labor force but had worked in the past year)
    # Good universe for what we're trying to measure here
    full_month = full_month[full_month["AGE"] > 15]
    full_month = full_month[full_month["OCC"]!="0000"]

    for v in vars:
        month = full_month[full_month["EMPSTAT"]=="1"]

        for e in ["exposed", "highly_exposed"]:
            this = month.groupby(e).agg(
                    var = pandas.NamedAgg(
                        column = v,
                        aggfunc = lambda x: numpy.average(x, weights = month.loc[x.index, "WTFINL"])
                    )
            ).reset_index().rename(columns = {e: "group_val"}).assign(group = e)  
            
            if "temp" not in locals():
                temp = this
            else:
                temp = pandas.concat([temp, this], ignore_index = True, sort = False)
        temp = temp.rename(columns = {"var": v}).assign(
            group_val = lambda x: numpy.where(x["group_val"]==0, False, True)
        )

        if "out" not in locals():
            out = temp
        else:
            out = out.merge(temp)

    out.insert(0, "MONTH", [this_month] * len(out))

    return out