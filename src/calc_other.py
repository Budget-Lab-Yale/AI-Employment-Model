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

def build_other_output(ID: str, start: str, end: str, write_path: str, metric = "mean_rating_human_beta", freq = 3):
    locator = mdates.YearLocator()
    fmt     = mdates.DateFormatter("%Y")
    write_dir = os.path.join(write_path, ID)
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)
    
    # Here we minimize number of I/O operations rather than reducing frame complexity
    # As such, subfunctions are significantly more complex, but it runs (a bit) faster
    # We also pack output so that the grouping categories can remain seperate
    # Generally speaking, objects that include "_e" or "e_" use exposure as a grouping characteristic rather than as an output metric
    block = calc_block2(ID = ID, start = start, end = end, groups = ["EMPSTAT", "DURUNEMP_cat", "SEX"], freq = freq,  metric = metric) 

    block.to_csv(os.path.join(write_dir, "exposure.csv"), index = False)

    # cleaning for comparison of OpenAI to Anthropic measures
    if ID == "indexed" or ID == "indexed_fm":
        ID = "november_fm"
    full_month = pandas.read_parquet(
        os.path.join("/nfs/roberts/project/pi_nrs36/shared/model_data/AI-Employment-Model/input", ID,  "cps_" + '202301' + ".parquet"), engine = 'pyarrow'
    )

    full_month = full_month[full_month["AGE"] > 15]
    full_month = full_month[full_month["OCC"]!="0000"]

    is_job_exposure = (ID == "job_exposure")

    if is_job_exposure:
        check = full_month.groupby("OCC").agg({metric: 'mean', 'observed_exposure': 'mean'}).reset_index()
    else:
        check = full_month.groupby("OCC").agg({metric:'mean', 'automation':'mean', 'augmentation':'mean', 'total':'mean', 'pct_of_convs':'mean'}).reset_index()
        check['not_filtered'] = check['automation'] + check['augmentation']
        check['not_filtered_pct'] = check['not_filtered'] / check['total']

    socs = read_csv("/nfs/roberts/project/pi_nrs36/jmk263/Repositories/AI-Employment-Model/resources/crosswalk.csv").dropna(subset = 'cps_code').assign(
        cps_code  = lambda x: x['cps_code'].astype(int).astype(str).str.zfill(4),
        major_code = lambda x: x["Code"].str[:2].astype(int),
    )
    oes = read_csv('../resources/national_M2024_dl.csv')[['OCC_CODE', 'TOT_EMP', 'OCC_TITLE', 'O_GROUP']]
    oes['Code'] = oes['OCC_CODE'] + '.00'
    socs = socs.merge(oes, how = 'left', on = 'Code')
    socs = socs.groupby(['cps_code', 'major_code']).agg({'TOT_EMP':'sum'}).reset_index()

    majors = read_csv("/nfs/roberts/project/pi_nrs36/jmk263/Repositories/AI-Employment-Model/resources/soc_major.csv")
    socs = socs.merge(majors, how = "left", on = "major_code")

    if is_job_exposure:
        check = check.merge(socs, how = "left", left_on = "OCC", right_on = "cps_code")
    else:
        check = check.merge(socs, how = "left", left_on = "OCC", right_on = "cps_code").assign(
            quadrant = lambda x: numpy.where(
                x[metric] >= .5,
                numpy.where(x['not_filtered_pct'] >= .5, "Q1", "Q4"),
                numpy.where(x['not_filtered_pct'] >= .5, "Q2", "Q3"),
            )
        )

    check.to_csv(os.path.join(write_dir, "openai_vs_anthropic.csv"), index = False)

    check['weighted_exposure'] = check[metric] * check['TOT_EMP']

    def get_wp(var, total, weights):
        w_sum = (var * weights).sum()
        w_total = (weights * total * ~var.isna()).sum()
        return (w_sum / w_total) * 100 if w_total > 0 else 0

    if is_job_exposure:
        out = check.groupby('major_occ').agg(
            average_exposure = pandas.NamedAgg(
                column = metric,
                aggfunc = lambda x, w=check['TOT_EMP']: get_wp(x, 1, w.loc[x.index])
            ),
            average_usage = pandas.NamedAgg(
                column = 'observed_exposure',
                aggfunc = lambda x, w=check['TOT_EMP']: get_wp(x, 1, w.loc[x.index])
            ),
            weighted_exposure = pandas.NamedAgg(column = 'weighted_exposure', aggfunc = 'sum'),
            TOT_EMP = pandas.NamedAgg(column = 'TOT_EMP', aggfunc = 'sum')
        ).reset_index()
        out['emp_share'] = out['TOT_EMP'] / out['TOT_EMP'].sum() * 100
        out['expected_usage'] = out['weighted_exposure'] / out['weighted_exposure'].sum() * 100
        out = out[['major_occ', 'average_exposure', 'average_usage', 'emp_share', 'expected_usage']]
    else:
        out = check.groupby('major_occ').agg(
            average_exposure = pandas.NamedAgg(
                column = metric,
                aggfunc = lambda x, w=check['TOT_EMP']: get_wp(x, 1, w.loc[x.index])
            ),
            average_usage = pandas.NamedAgg(
                column = 'not_filtered_pct',
                aggfunc = lambda x, w=check['TOT_EMP']: get_wp(x, 1, w.loc[x.index])
            ),
            weighted_exposure = pandas.NamedAgg(column = 'weighted_exposure', aggfunc = 'sum'),
            observed_usage = pandas.NamedAgg(column = 'pct_of_convs', aggfunc = 'sum'),
            TOT_EMP = pandas.NamedAgg(column = 'TOT_EMP', aggfunc = 'sum')
        ).reset_index()
        out['emp_share'] = out['TOT_EMP'] / out['TOT_EMP'].sum() * 100
        out['observed_usage'] = out['observed_usage'] / out['observed_usage'].sum() * 100
        out['expected_usage'] = out['weighted_exposure'] / out['weighted_exposure'].sum() * 100
        out = out[['major_occ', 'average_exposure', 'average_usage', 'emp_share', 'expected_usage', 'observed_usage']]

    out.to_csv(os.path.join(write_dir, "molly.csv"), index = False)


def calc_block2(ID: str, start: str, end: str, groups, freq = 3, metric = "mean_rating_human_beta"):
    
    data = collapse_months2(ID, start, end, groups, metric)
    
    if freq == 1:
        return data

    months = data["MONTH"].unique()[freq-1:]    
    for month in months:
        curr = calc_avg2(data, month, freq, groups=groups)
        if "out" not in locals():
            out = curr
        else:
            out = pandas.concat([out, curr], ignore_index = True, sort = False)
    out = out.rename(columns = {"MONTH": "time"})
    return out


def collapse_months2(ID: str, start: str, end: str, groups, metric: str):
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

        curr = get_month(ID, month, groups, metric, ID == "indexed")

        if "collapsed" not in locals():
            collapsed = curr
        else:
            collapsed = pandas.concat([collapsed, curr], ignore_index = True, sort = False)

        mn_dex += 1
        if mn_dex > 12:
            mn_dex = mn_dex % 12
            yr_dex += 1
    
    return collapsed

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
        return these_months.agg(lambda x: x.mean(skipna = (this_month != "202510")))
    else:
        these_months = these_months.groupby(["group", "group_val"], as_index=True).agg(lambda x: x.mean(skipna = (this_month != "202510")))
        these_months.reset_index(inplace=True)
        these_months.insert(0, "MONTH", date(int(this_month[:4]), int(this_month[4:]),2))

    return these_months

def get_month(ID: str, this_month: str, groups, metric: str, index_adoption = False):
    is_job_exposure = (ID == "job_exposure")

    if this_month == "202510":
        out = get_month(ID, "202509", groups, metric, ID == "indexed")
        to_agg = ['tasks_exposed_pct', 'percent_lowest_exposed', 'percent_middle_exposed', 'percent_highest_exposed', 'percent_mildly_exposed', 'percent_moderately_exposed', 'percent_highly_exposed', 'automation', 'augmentation', 'percent_automated', 'percent_augmented']
        out[to_agg] = float("nan")
        out["MONTH"] = "202510"

    else:
        if ID == "indexed" or ID == "indexed_fm":
            base_id = "november"
            ID = base_id + "_fm" if ID == "indexed_fm" else base_id

        full_month = pandas.read_parquet(
            os.path.join("/nfs/roberts/project/pi_nrs36/shared/model_data/AI-Employment-Model/input", ID, "cps_" + this_month + ".parquet"), 
            engine = 'pyarrow'
        )
        
        full_month = full_month.assign(
            mildly_exposed     = lambda x: numpy.where(numpy.logical_and(x[metric] > 0, x[metric] < .4), 1, 0),
            moderately_exposed = lambda x: numpy.where(numpy.logical_and(x[metric] >= .4, x[metric] < .8), 1, 0),
            highly_exposed     = lambda x: numpy.where(x[metric] >= .8, 1, 0),
            lowest_exposed     = lambda x: numpy.where(numpy.logical_and(x[metric] > 0, x[metric] <= 0.2687269), 1, 0),
            middle_exposed     = lambda x: numpy.where(numpy.logical_and(x[metric] > 0.2687269, x[metric] <= 0.5307864), 1, 0),
            highest_exposed    = lambda x: numpy.where(x[metric] > 0.5307864, 1, 0),
            automated = lambda x: numpy.where(
                x['observed_exposure'] > .5, 1, 0
            ) if is_job_exposure else numpy.where(
                x['automation'] / x['total'] > .5, 1, 0
            ),
            # augmented is skipped for job_exposure
            **({} if is_job_exposure else {
                'augmented': lambda x: numpy.where(x['augmentation'] / x['total'] > .5, 1, 0)
            })
        )
        
        full_month = full_month[full_month["AGE"] > 15]
        full_month = full_month[full_month["OCC"]!="0000"]
        
        if index_adoption:
            adoption_all = pandas.read_csv("../resources/adoption_index_monthly.csv")
            adoption = adoption_all[adoption_all["Survey Date"] == int(this_month)].melt(
                id_vars=["Survey Date"], 
                var_name="major_occ", 
                value_name="rate"
            )

            if len(adoption) == 0:
                raise ValueError(f"No adoption data found for month {this_month}")

            full_month = full_month.merge(adoption[['major_occ', 'rate']], on='major_occ', how='left')

            missing_occs = full_month[full_month['rate'].isna()]['major_occ'].unique()
            if len(missing_occs) > 0:
                print(f"Warning: No adoption rate found for occupations: {missing_occs}")
                print("Setting adoption rate to 0 for these occupations")
                full_month['rate'] = full_month['rate'].fillna(0)

            numpy.random.seed(20221130)
            full_month['random_draw'] = numpy.random.uniform(0, 1, size=len(full_month))
            full_month['uses_ai'] = numpy.where(full_month['random_draw'] < full_month['rate'], 1, 0)
            full_month = full_month.drop(columns=['rate', 'random_draw'])
        else:
            full_month["uses_ai"] = 1
        

        for g in groups:  
            def get_wp(var, total, weights):
                w_sum = (var * weights).sum()
                w_total = (weights * total * ~var.isna()).sum()
                return (w_sum / w_total) * 100 if w_total > 0 else 0

            if g == "DURUNEMP_cat":
                month = full_month[full_month["EMPSTAT"]!="1"]
                month = month[~month["DURUNEMP"].isin([0,999])]
            else:
                month = full_month[full_month["EMPSTAT"]=="1"]

            agg_dict = {
                'tasks_exposed_pct': pandas.NamedAgg(
                    column = metric,
                    aggfunc = lambda x, w=month['WTFINL']: get_wp(x, 1, w.loc[x.index])
                ),
                'percent_mildly_exposed': pandas.NamedAgg(
                    column = 'mildly_exposed',
                    aggfunc = lambda x, w=month['WTFINL']: get_wp(x, 1, w.loc[x.index])
                ),
                'percent_moderately_exposed': pandas.NamedAgg(
                    column = 'moderately_exposed',
                    aggfunc = lambda x, w=month['WTFINL']: get_wp(x, 1, w.loc[x.index])
                ),
                'percent_highly_exposed': pandas.NamedAgg(
                    column = 'highly_exposed',
                    aggfunc = lambda x, w=month['WTFINL']: get_wp(x, 1, w.loc[x.index])
                ),
                'percent_lowest_exposed': pandas.NamedAgg(
                    column = 'lowest_exposed',
                    aggfunc = lambda x, w=month['WTFINL']: get_wp(x, 1, w.loc[x.index])
                ),
                'percent_middle_exposed': pandas.NamedAgg(
                    column = 'middle_exposed',
                    aggfunc = lambda x, w=month['WTFINL']: get_wp(x, 1, w.loc[x.index])
                ),
                'percent_highest_exposed': pandas.NamedAgg(
                    column = 'highest_exposed',
                    aggfunc = lambda x, w=month['WTFINL']: get_wp(x, 1, w.loc[x.index])
                ),
                'percent_automated': pandas.NamedAgg(
                    column = 'automated',
                    aggfunc = lambda x, ai=month['uses_ai'], w=month['WTFINL']: get_wp(x * ai.loc[x.index], 1, w.loc[x.index])
                ),
            }

            if is_job_exposure:
                agg_dict['automation'] = pandas.NamedAgg(
                    column = 'observed_exposure',
                    aggfunc = lambda x, ai=month['uses_ai'], w=month['WTFINL']: get_wp(x * ai.loc[x.index], 1, w.loc[x.index])
                )
            else:
                agg_dict['automation'] = pandas.NamedAgg(
                    column = 'automation',
                    aggfunc = lambda x, t=month['total'], ai=month['uses_ai'], w=month['WTFINL']: get_wp(x * ai.loc[x.index], t.loc[x.index], w.loc[x.index])
                )
                agg_dict['augmentation'] = pandas.NamedAgg(
                    column = 'augmentation',
                    aggfunc = lambda x, t=month['total'], ai=month['uses_ai'], w=month['WTFINL']: get_wp(x * ai.loc[x.index], t.loc[x.index], w.loc[x.index])
                )
                agg_dict['percent_augmented'] = pandas.NamedAgg(
                    column = 'augmented',
                    aggfunc = lambda x, ai=month['uses_ai'], w=month['WTFINL']: get_wp(x * ai.loc[x.index], 1, w.loc[x.index])
                )

            temp = month.groupby(g).agg(**agg_dict).reset_index().rename(columns={g: "group_val"}).assign(group=g)

            keep_cols = ['group', 'group_val', 'tasks_exposed_pct', 'percent_lowest_exposed', 'percent_middle_exposed',
                         'percent_highest_exposed', 'percent_mildly_exposed', 'percent_moderately_exposed',
                         'percent_highly_exposed', 'automation', 'percent_automated']
            if not is_job_exposure:
                keep_cols += ['augmentation', 'percent_augmented']
            temp = temp[keep_cols]
            
            if "out" not in locals():
                out = temp
            else:
                out = pandas.concat([out, temp], ignore_index = True, sort = False)
        
        out.insert(0, "MONTH", [this_month] * len(out))
        del temp
    
    return out


def get_month2(ID: str, this_month: str, groups, metric: str):
    full_month = pandas.read_parquet(
        os.path.join("/nfs/roberts/project/pi_nrs36/shared/model_data/AI-Employment-Model/input", ID, "cps_" + this_month + ".parquet"), 
    )
    
    full_month = full_month.assign(
        mildly_exposed     = lambda x: numpy.where(numpy.logical_and(x[metric] > 0, x[metric] < .4), 1, 0),
        moderately_exposed = lambda x: numpy.where(numpy.logical_and(x[metric] >= .4, x[metric] < .8), 1, 0),
        highly_exposed     = lambda x: numpy.where(x[metric] >= .8, 1, 0),
        lowest_exposed     = lambda x: numpy.where(numpy.logical_and(x[metric] > 0, x[metric] <= 0.2687269), 1, 0),
        middle_exposed     = lambda x: numpy.where(numpy.logical_and(x[metric] > 0.2687269, x[metric] <= 0.5307864), 1, 0),
        highest_exposed    = lambda x: numpy.where(x[metric] > 0.5307864, 1, 0),
    )
    
    full_month = full_month[full_month["AGE"] > 15]
    full_month = full_month[full_month["OCC"]!="0000"]
    
    for g in groups:  

        def get_wp(var, total, weights):
            w_sum = (var * weights).sum()
            w_total = (weights * total * ~var.isna()).sum()
            return (w_sum / w_total) * 100 if w_total > 0 else 0
        
        if g == "DURUNEMP_cat":
            month = full_month[full_month["EMPSTAT"]!="1"]
            month = month[~month["DURUNEMP"].isin([0,999])]
        else:
            month = full_month[full_month["EMPSTAT"]=="1"]

        temp = month.groupby(g).agg(
            tasks_exposed_pct = pandas.NamedAgg(
                column = metric,
                aggfunc = lambda x, w=month['WTFINL']: get_wp(x, 1, w.loc[x.index])
            ),
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
            automation = pandas.NamedAgg(
                column = 'automation',
                aggfunc = lambda x, t=month['total'], w=month['WTFINL']: get_wp(x, t.loc[x.index], w.loc[x.index])
            ),
            augmentation = pandas.NamedAgg(
                column = 'augmentation',
                aggfunc = lambda x, t=month['total'], w=month['WTFINL']: get_wp(x, t.loc[x.index], w.loc[x.index])
            ),
            filtered = pandas.NamedAgg(
                column = 'filtered',
                aggfunc = lambda x, t=month['total'], w=month['WTFINL']: get_wp(x, t.loc[x.index], w.loc[x.index])
            )
        ).reset_index().rename(columns = {g: "group_val"}).assign(group = g)
        
        temp = temp[['group', 'group_val', 'tasks_exposed_pct', 'percent_lowest_exposed', 'percent_middle_exposed', 'percent_highest_exposed', 'percent_mildly_exposed', 'percent_moderately_exposed', 'percent_highly_exposed', 'automation', 'augmentation', 'filtered']]
        
        if "out" not in locals():
            out = temp
        else:
            out = pandas.concat([out, temp], ignore_index = True, sort = False)
    
    out.insert(0, "MONTH", [this_month] * len(out))
    
    return out