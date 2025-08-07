import pandas
import numpy
import os
import psutil
import gc
import pyarrow as pa
import pyarrow.parquet as pq

from pandas import read_csv, DataFrame
from ipumspy import readers, ddi

def build_linked_dataset(ID: str, start: str, end: str, occ_code = "OCC", simple = False):
    
    # Path to non-CPS data
    rsc_path = os.path.join(os.path.dirname(__file__), "..", "resources")

    # SOC major group codes and descriptions
    soc_major = read_csv(os.path.join(rsc_path, "soc_major.csv"))
    
    # Crosswalk between SOC codes and CPS codes
    crosswalk = read_csv(os.path.join(rsc_path, "crosswalk.csv")).dropna(subset = 'cps_code').assign(
        # Processing values to facilitate easier merging later on
        cps_code  = lambda x: x['cps_code'].astype(int).astype(str).str.zfill(4),
        major_code = lambda x: x["Code"].str[:2].astype(int)
    )
    
    # Processes AI exposure data
    exposure = collapse_exposure(rsc_path)

    # Attaches disparate data together. This data is not time sensitive, though some of it requires qualification
    external = crosswalk.merge(
        soc_major,
        how = 'left',
        left_on = 'major_code',
        right_on = 'major_code'
    ).merge(
        exposure, 
        how = 'left',
        left_on = 'Code', 
        right_on = 'O*NET-SOC Code' 
    ).drop(["Job Zone", "O*NET-SOC Code", "Occupation", "Data-level", 'major_code'], axis = 1).groupby(
        'cps_code').agg({
        'mean_rating_human_alpha': 'mean',
        'mean_rating_human_beta': 'mean',
        'mean_rating_human_gamma': 'mean',
        'gpt4_rubric1_alpha': 'mean',
        'gpt4_rubric1_beta': 'mean',
        'gpt4_rubric1_gamma': 'mean',
        'gpt4_rubric2_beta': 'mean',
        'gpt4_automation': 'mean',
        'augmentation': 'mean',
        'automation': 'mean',
        'filtered': 'mean',
        'total': 'mean'
    })

    cps_path = os.path.join(os.path.dirname(__file__), "..", "cps")

    if not os.path.exists(os.path.join(cps_path, "processed", ID)):
        os.makedirs(os.path.join(cps_path, "processed", ID))

    ddi = readers.read_ipums_ddi(os.path.join(cps_path, "raw", ID, "cps.xml"))
    # Processing CPS data to combine with exposure data
    if simple:
            cps = readers.read_microdata(ddi, os.path.join(cps_path, "raw", ID, "cps.dat.gz")).assign(
            OCC      = lambda x: x[occ_code].astype(str).str.zfill(4),
            IND      = lambda x: x["IND"].astype(str).str.zfill(4),
        )

    else:
        cps = readers.read_microdata(ddi, os.path.join(cps_path, "raw", ID, "cps.dat.gz")).assign(
            OCC      = lambda x: x[occ_code].astype(str).str.zfill(4),
            IND      = lambda x: x["IND"].astype(str).str.zfill(4),
            SEX      = lambda x: numpy.where(x['SEX'] == 1, "Male", "Female"),
            EMPSTAT  = lambda x: x["EMPSTAT"].astype(str).str.zfill(2).str[:1],
            LABFORCE = lambda x: x["LABFORCE"] - 1,
            DURUNEMP_cat = lambda x: pandas.cut(
                x["DURUNEMP"],
                [0,6,12,float('inf')],
                right = False,
                include_lowest = True,
                labels = ["1-6 Weeks", "6-12 Weeks", "12+ Weeks"]
            ).astype(str)
        )

    if ID == "benchmark":
        linked = cps.merge(
                external,
                how = 'left',
                left_on = 'OCC', 
                right_on = 'cps_code'
        )
        linked = linked[linked["EMPSTAT"] == "1"]
        
        pq.write_table(pa.Table.from_pandas(linked), os.path.join(cps_path, "processed", ID, "cps_" + ID + ".parquet"), compression='snappy')
    
    else:
        start = start.split('.')
        end   = end.split('.')
        # Year and month iterators
        yr_dex = int(start[0])
        mn_dex = int(start[1])
        # Year and month end conditions
        end_yr = int(end[0])
        end_mn = int(end[1]) + 1

        # END parameter is inclusive, so we extend the month by one and check to see if we need to iterate years
        if end_mn > 12:
            end_yr += 1
            end_mn = end_mn % 12 

        
        while((yr_dex != end_yr) or (mn_dex != end_mn)):
            # Year-Month stamp in YYYYMM format
            stamp = str(yr_dex) + str(mn_dex).zfill(2)

            # Get CPS data associated with the month at hand and attach exposure data
            linked = cps[(cps['MONTH'] == mn_dex) & (cps['YEAR'] == yr_dex)].merge(
                    external,
                    how = 'left',
                    left_on = 'OCC', 
                    right_on = 'cps_code'
            )
            # Write out data as a smaller .parquet file. We don't need to look at the micro outside of these scripts so this make I/0 WAY faster
            pq.write_table(pa.Table.from_pandas(linked), os.path.join(cps_path, "processed", ID, "cps_" + stamp  + ".parquet"), compression='snappy')

            # Iterate month
            mn_dex += 1
            # Check if we need to iterate year
            if mn_dex > 12:
                mn_dex = mn_dex % 12
                yr_dex += 1
            
            # Clean gargabe to avoid accidental memory leak (these are big dataframes in local memory and can easily exceed RAM if not careful)
            del linked
            collected_count = gc.collect()
    
    return

def collapse_exposure(rsc_path: str):

    ### OPENAI EXPOSURE DATA ###

    # The OpenAI exposure data, containing:
    # O*NET SOC code, the associated occupation in text, a task, and its various exposure metrics coded in range(0,1)
    # See https://arxiv.org/pdf/2303.10130 for details
    exposure_raw = read_csv(os.path.join(rsc_path, "exposure.csv")).assign(
        Task = lambda x: x["Task"].str.lower().str.strip()
    )
    exposure_columns = ["mean_rating_human_alpha", "mean_rating_human_beta", "mean_rating_human_gamma", "gpt4_rubric1_alpha", \
        "gpt4_rubric1_beta", "gpt4_rubric1_gamma", "gpt4_rubric2_beta", "gpt4_automation"]

    # Read in and attach task weights if we ever get them 
    # Tasks are not weighted in exposure data according to how important they are to the occupation
    onet_tasks = read_csv(os.path.join(rsc_path, 'onet_task_statements.csv'))[["O*NET-SOC Code", "Task", "Task ID", "Task Type"]].assign(
        t_wgt = lambda x: numpy.where(x["Task Type"]=="Core", 1, .5)
    ).drop(["Task Type"], axis=1)

    exposure_raw = exposure_raw.merge(
        onet_tasks[["Task ID", "t_wgt"]],
        how = 'left',
        on = 'Task ID'
    )
    
    exposure_raw[exposure_columns] = exposure_raw[exposure_columns].apply(lambda x: x * exposure_raw["t_wgt"])

    # Anthropic data for a task's percent of all Claude queries
    """ task_pct = read_csv(os.path.join(rsc_path, "onet_task_mappings.csv")).rename(
        columns = {"pct": "pct_of_convs"}
    )
    
    exposure_raw = exposure_raw.merge(
        task_pct,
        left_on="Task",
        right_on="task_name",
        how="left"
    ) """
    # Start a new dataframe where the lhs is all unique SOC codes
    collapsed = DataFrame({"O*NET-SOC Code": exposure_raw["O*NET-SOC Code"].unique()})

    # Iterate over metrics and collapse tasks' exposure to a single value for the 
    for m in exposure_columns: 
        temp = DataFrame()      
        temp[m] = exposure_raw.groupby("O*NET-SOC Code")[m].sum() / exposure_raw.groupby("O*NET-SOC Code")["t_wgt"].sum()
        
        # Reset index and attach collapsed metric to our output frame SOC code-wise
        temp = temp.reset_index()        
        collapsed = collapsed.merge(
            temp,
            how = 'left',
            on = "O*NET-SOC Code"
        )
    
    ### ANTHROPIC USAGE DATA ###

    # Usage metrics for a tasks' propensity to have Claude augment or automate a task. Usage sums to 1 for a task. 
    # If a task has filtered = 1, then the task is neither automated nor augmented with Claude (i.e. "apply caulk, sealants, or other agents to installed surfaces.")
    # A filtered task is present in the data because someone with an occupation who does that task used Claude, but the conversations did not pertain to that specific task
    usage_raw  = read_csv(os.path.join(rsc_path, "automation_vs_augmentation_by_task.csv"))

    # Another frame of ONET tasks and occupation codes
    #onet_tasks = read_csv(os.path.join(rsc_path, 'onet_task_statements.csv'))[["O*NET-SOC Code", "Task"]]
    onet_tasks["Task"] = onet_tasks["Task"].str.lower().str.strip()
    onet_tasks = onet_tasks[["O*NET-SOC Code", "Task", 't_wgt']]

    # Might be useful depending on future paramaterization, not in the current one
    #usage_raw = usage_raw[usage_raw["filtered"] != 1]

    # Attach usage to the task descriptions/codes, maintaining the tasks rather than the usage
    onet_tasks = onet_tasks.merge(
        usage_raw,
        how = 'left',
        left_on = "Task",
        right_on = "task_name"
    ).drop(["task_name"], axis = 1 )

    # Pivot long
    onet_tasks = pandas.melt(
        onet_tasks, 
        id_vars = ['O*NET-SOC Code', "Task"],
        value_vars=["validation", "task_iteration", "learning", "directive", "feedback_loop", "filtered", "t_wgt"]
    )

    onet_tasks[onet_tasks["value"].isnull()].to_csv("null_check.csv")

    # Drop null values. Some O*NET tasks are not present in the Claude data. As such, they have null values for the Claude variables, which disrupts aggregation
    onet_tasks = onet_tasks[onet_tasks["value"].notnull()]
    
    # Pivot wide to original format
    onet_tasks = onet_tasks.pivot(index = ['O*NET-SOC Code', "Task"], columns = "variable", values = 'value').reset_index()

    # Sum types of usage into either augmentation or automation for a TASK
    onet_tasks = onet_tasks.assign(
        augmentation = lambda x: x["validation"] + x["task_iteration"] + x["learning"],
        automation   = lambda x: x["directive"] + x["feedback_loop"]
    ).drop(["validation", "task_iteration", "learning", "directive", "feedback_loop"], axis = 1)

    # Aggregate task sub-sums of usage to OCCUPATION level
    # Note that we don't calculate percentages or normalize during this pre-processing step. As such, values are typically range(0,n(tasks_per_occ))
    # We keep them un-normalized so that we can do normalization only once we aggregate microdata individuals into groups during processing
    onet_tasks[["augmentation", "automation", "filtered"]] = onet_tasks[["augmentation", "automation", "filtered"]].apply(lambda x: x * onet_tasks['t_wgt'])

    temp = onet_tasks.groupby("O*NET-SOC Code").agg({
        'augmentation': 'sum',
        'automation':   'sum',
        'filtered' :    'sum'
    }).reset_index().assign(
        total = lambda x: x['augmentation'] + x['automation'] + x['filtered'],
        augmentation = lambda x: x['augmentation'],
        automation = lambda x: x['automation'],
        filtered = lambda x: x['filtered']
    )
    # Finally attach all OCCUPATION level metrics of exposure and usage
    collapsed = collapsed.merge(
        temp,
        how = "left",
        on = "O*NET-SOC Code"
    )
    
    return collapsed


def month_helper(month: int):
    # Just a helper function to do one-line month iteration
    if (month % 12) == 0:
        return 12
    else:
        return month % 12

