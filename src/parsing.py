import pandas
import numpy
import os
import gc
import pyarrow as pa
import pyarrow.parquet as pq

from pandas import read_csv, DataFrame
from ipumspy import readers, ddi

def build_linked_dataset(ID: str, start: str, end: str, out_path: str, build_map = False, occ_code = "OCC", simple = False, fill_missing_tasks = True):
    
    # Path to non-CPS data
    rsc_path = os.path.join(os.path.dirname(__file__), "..", "resources")

    # Attaches disparate data together.
    if build_map:
        external = collapse_exposure(rsc_path, fill_missing_tasks)
 
    #cps_path = os.path.join(os.path.dirname(__file__), "..", "cps")
    cps_path = os.path.join("/gpfs/gibbs/project/sarin/shared/raw_data/CPS-Monthly", ID)

    #if not os.path.exists(os.path.join(cps_path, "processed", ID)):
    #    os.makedirs(os.path.join(cps_path, "processed", ID))

    ddi = readers.read_ipums_ddi(os.path.join(cps_path, "cps.xml"))
    # Processing CPS data to combine with exposure data
    if simple:
            cps = readers.read_microdata(ddi, os.path.join(cps_path, "cps.dat.gz")).assign(
            OCC      = lambda x: x[occ_code].astype(str).str.zfill(4),
            IND      = lambda x: x["IND"].astype(str).str.zfill(4),
        )

    else:
        cps = readers.read_microdata(ddi, os.path.join(cps_path, "cps.dat.gz")).assign(
            OCC      = lambda x: x[occ_code].astype(str).str.zfill(4),
            IND      = lambda x: x["IND"].astype(str).str.zfill(4),
            SEX      = lambda x: numpy.where(x['SEX'] == 1, "Male", "Female"),
            EMPSTAT  = lambda x: x["EMPSTAT"].astype(str).str.zfill(2).str[:1],
            LABFORCE = lambda x: x["LABFORCE"] - 1,
            DURUNEMP_cat = lambda x: pandas.cut(
                x["DURUNEMP"],
                [0,5,15,27,float('inf')],
                right = False,
                include_lowest = True,
                labels = ["1-5 Weeks", "5-14 Weeks", "15-26 Weeks", "27+ Weeks"]
            ).astype(str),
            AGE_cat = lambda x: pandas.cut(
                x["AGE"],
                [0, 22, 26, 31, 35, 41, 50, float('inf')],
                right = False,
                include_lowest = True,
                labels = ["Junior (<22)", "Early Career 1 (22-25)", "Early Career 2 (26-30)", "Developing (31-34)", \
                    "Mid-Career 1 (35-40)", "Mid-Career 2 (41-49)", "Senior (50+)"]
            )
        )

    start = start.split('.')
    end   = end.split('.')

    #for m in ["march", "march_fm", "august_claude", "august_claude_fm", "august", "august_fm"]:
    for m in ["march", "august_claude", "august", "november_claude", "november"]:
        for fm in [False, True]:
            map_paths = {}
            #print(f"Building dataset for: {m}")
            if occ_code == "OCC2010":
                sub_dir = "occ2010/input"
                map_paths["Industry"] = os.path.join(out_path, "dissimilarity/Industry") 

            else:
                sub_dir = "input"
                map_paths["AI"] = os.path.join(out_path, "dissimilarity/AI")

            if fm:
                map_paths["monthly"] = os.path.join(out_path, sub_dir, m + "_fm")
            else:
                map_paths["monthly"] = os.path.join(out_path, sub_dir, m)

            external = collapse_exposure(rsc_path, m, fm)

            #external = read_csv(os.path.join(rsc_path, "full_map_" + m + ".csv")).assign(
            #    cps_code = lambda x: x["cps_code"].astype(str).str.zfill(4)
            #)
            print(map_paths)
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
                for key in map_paths:
                    if not os.path.exists(map_paths[key]):
                        os.makedirs(map_paths[key])
                    this_path = os.path.join(map_paths[key], "cps_" + stamp  + ".parquet")
                    if not os.path.exists(this_path):
                        pq.write_table(pa.Table.from_pandas(linked), this_path, compression='snappy')

                # Iterate month
                mn_dex += 1
                # Check if we need to iterate year
                if mn_dex > 12:
                    mn_dex = mn_dex % 12
                    yr_dex += 1
                
                # Clean garbage to avoid accidental memory leak (these are big dataframes in local memory and can easily exceed RAM if not careful)
                del linked
                collected_count = gc.collect()
    
    return

def collapse_exposure(rsc_path: str, samp: str, fill_missing_tasks: bool):

    # SOC major group codes and descriptions
    soc_major = read_csv(os.path.join(rsc_path, "soc_major.csv"))
    
    # Crosswalk between SOC codes and CPS codes
    oes = read_csv(os.path.join(rsc_path, 'national_M2024_dl.csv'))[['OCC_CODE', 'TOT_EMP', 'OCC_TITLE', 'O_GROUP']]
    
    crosswalk = read_csv(os.path.join(rsc_path, "crosswalk.csv")).dropna(subset = 'cps_code').assign(
        # Processing values to facilitate easier merging later on
        cps_code  = lambda x: x['cps_code'].astype(int).astype(str).str.zfill(4),
        major_code = lambda x: x["Code"].str[:2].astype(int),
        no_detail = lambda x: x["Code"].str[:7]
    )
    
    crosswalk = crosswalk.merge(
        soc_major,
        how = 'left',
        left_on = 'major_code',
        right_on = 'major_code'
    )

    # Start a new dataframe where the lhs is all unique SOC codes
    collapsed = DataFrame({"O*NET-SOC Code": crosswalk["Code"].unique()})

    ### OPENAI EXPOSURE DATA ###

    # The OpenAI exposure data, containing:
    # O*NET SOC code, the associated occupation in text, a task, and its various exposure metrics coded in range(0,1)
    # See https://arxiv.org/pdf/2303.10130 for details

    exposure_raw = read_csv(os.path.join(rsc_path, "exposure.csv")).assign(
        Task = lambda x: x["Task"].str.lower().str.strip()
    )

    exposure_columns = ["mean_rating_human_alpha", "mean_rating_human_beta", "mean_rating_human_gamma", "gpt4_rubric1_alpha", \
        "gpt4_rubric1_beta", "gpt4_rubric1_gamma", "gpt4_rubric2_beta", "gpt4_automation"]
    

    # Core tasks get 1 non core get .5
    onet_tasks = read_csv(os.path.join(rsc_path, 'onet_task_statements_old.csv'))[["O*NET-SOC Code", "Task", "Task ID", "Task Type", "Title"]].assign(
        t_wgt = lambda x: numpy.where(x["Task Type"]=="Core", 1, .5)
    ).drop(["Task Type"], axis=1)
    onet_tasks["Task"] = onet_tasks["Task"].str.lower().str.strip()
    onet_tasks["n_occurrences"] = onet_tasks.groupby("Task")["Title"].transform("nunique")
    onet_tasks["n_occurrences"] = onet_tasks["n_occurrences"].fillna(1)
    
    old_new = read_csv(os.path.join(rsc_path, "2010_to_2019_Crosswalk.csv"))
    
    onet_tasks = onet_tasks.merge(
        old_new,
        how = 'left',
        left_on ='O*NET-SOC Code',
        right_on = 'O*NET-SOC 2010 Code'
    ).drop(["O*NET-SOC Code", 'Title'], axis=1).rename(
        columns = {
            'O*NET-SOC 2019 Code':"O*NET-SOC Code",
            "O*NET-SOC 2019 Title": "Title"
        }
    ) 

    # Better weight mapping for exposure
    onet_v2 = read_csv(os.path.join(rsc_path, "onet_tasks_v2.csv")).assign(
        t_wgt = lambda x: numpy.where(x["Task Type"]=="Core", 1, .5)
    )
    
    exposure_raw = exposure_raw.merge(
        onet_v2[["Task ID", "t_wgt"]],
        how = 'left', 
        on = 'Task ID'
    ).assign(
        t_wgt = lambda x: numpy.where(x['t_wgt'].isna(), .75, x['t_wgt'])
    )
    
    exposure_raw[exposure_columns] = exposure_raw[exposure_columns].apply(lambda x: x * exposure_raw["t_wgt"])
    
    # Iterate over metrics and collapse tasks' exposure to a single value for the 
    for m in exposure_columns:       
        mask = exposure_raw[m].notna()
        num = exposure_raw.loc[mask].groupby("O*NET-SOC Code")[m].sum()
        den = exposure_raw.loc[mask].groupby("O*NET-SOC Code")["t_wgt"].sum()
        temp = (num / den).rename(m).reset_index()

        collapsed = collapsed.merge(
            temp,
            how = 'left',
            on = "O*NET-SOC Code"
        )
    
    ### ANTHROPIC USAGE DATA ###

    # March Release:  automation_vs_augmentation_by_task.csv
    # August Release: automation_vs_augmentation_by_task_v2.csv
    # August (Claude) Release: automation_vs_augmentation_by_task_v2_claude.csv
    # November Release: automation_vs_augmentation_by_task_v3.csv
    # November (Claude) Release: automation_vs_augmentation_by_task_v3_claude.csv
    if samp == "march":
        usage_data_source = "automation_vs_augmentation_by_task.csv"
    elif samp == "august":
        usage_data_source = "automation_vs_augmentation_by_task_v2.csv"
    elif samp == "august_claude":
        usage_data_source = "automation_vs_augmentation_by_task_v2_claude.csv"
    elif samp == "november":
        usage_data_source = "automation_vs_augmentation_by_task_v3.csv"
    else:
        usage_data_source = "automation_vs_augmentation_by_task_v3_claude.csv"


    # Percent of Conversations is included in the August releases (pre-processing)
    if usage_data_source == "automation_vs_augmentation_by_task.csv":
        # Anthropic data for a task's percent of all Claude queries
        
        task_pct = read_csv(os.path.join(rsc_path, "task_pct_v2.csv"))
        
        onet_tasks = onet_tasks.merge(
            task_pct,
            left_on = "Task",
            right_on = "task_name",
            how = 'left'
        )
    
    # Usage metrics for a tasks' propensity to have Claude augment or automate a task. Usage sums to 1 for a task. 
    # As per convention in Anthropic's reports, we drop filtered and renormalize the remaining categories.
    usage_raw  = read_csv(os.path.join(rsc_path, usage_data_source))
    usage_raw["task_name"] = usage_raw["task_name"].str.lower().str.strip()

    # Normalize the distribution for usage tasks after dropping "filtered" conversations
    usage_raw = usage_raw.drop('filtered', axis = 1)
    cols = ['validation', 'task_iteration', 'learning', 'feedback_loop', 'directive']
    new_total = usage_raw[cols].sum(axis=1)
    for col in cols:
        usage_raw[col] = usage_raw[col] / new_total
    

    # Attach usage to the task descriptions/codes
    # It's unclear which is the better choice methodologically regarding filling missing tasks. 
    
    if fill_missing_tasks:
        # Doing it this way implies that since the tasks aren't in the data, they aren't being used.
        # This provides fuller data but assumes that tasks that are similar to ones that are in the data are not comparable.
        onet_tasks = onet_tasks.merge(
            usage_raw,
            how = 'left',
            left_on = "Task",
            right_on = "task_name"
        ).assign(
            validation     = lambda x: x["validation"].fillna(0),
            task_iteration = lambda x: x["task_iteration"].fillna(0),
            learning       = lambda x: x["learning"].fillna(0),
            directive      = lambda x: x["directive"].fillna(0),
            feedback_loop  = lambda x: x["feedback_loop"].fillna(0),
            pct            = lambda x: x["pct"].fillna(0)
        )
    
    else:
        onet_tasks = usage_raw.merge(
            onet_tasks,
            how = 'left',
            left_on = "task_name",
            right_on = "Task"
        )

    # Sum types of usage into either augmentation or automation for a TASK
    onet_tasks = onet_tasks.assign(
        augmentation = lambda x: x["validation"] + x["task_iteration"] + x["learning"],
        automation   = lambda x: x["directive"] + x["feedback_loop"],
        pct_of_convs = lambda x: 100 * (x["pct"] / x["n_occurrences"]) / (x["pct"] / x["n_occurrences"]).sum()
    ).drop(["validation", "task_iteration", "learning", "directive", "feedback_loop", "pct"], axis = 1)
    
    # Aggregate task sub-sums of usage to OCCUPATION level
    # Note that we don't calculate percentages or normalize during this pre-processing step. As such, values are typically range(0,n(tasks_per_occ))
    # We keep them un-normalized so that we can do normalization only once we aggregate microdata individuals into groups during processing
    # If we are filling missing tasks, then every task should be counted and we fill totals with 1
    # Otherwise, it's simply the sum of the normalized automation and augmentation metrics
    onet_tasks['total'] = 1 if fill_missing_tasks else onet_tasks['automation'] + onet_tasks['augmentation']
    onet_tasks[["augmentation", "automation", "total"]] = onet_tasks[["augmentation", "automation", 'total']].apply(lambda x: x * onet_tasks['t_wgt'])

    temp = onet_tasks.groupby("O*NET-SOC Code").agg({
        'augmentation': 'sum',
        'automation':   'sum',
        'total' :    'sum',
        'pct_of_convs': 'sum'
    }).reset_index().assign(
        augmentation = lambda x: x['augmentation'],
        automation   = lambda x: x['automation'],
        total = lambda x: x['total'],
        pct_of_convs = lambda x: x['pct_of_convs']
    )
    
    # Sets tasks with 0s in augmentation, automation, and total to NaN so that they are ignored in future calculations.
    # This is only important for the method in which we do not fill tasks. In that configuration, 0s across the board (except for total) is a valid entry
    # In this configuration, it would only serve to confound the future aggregations
    if not fill_missing_tasks:
        temp[['augmentation','automation','total']] = temp[['augmentation','automation','total']].mask(temp[['augmentation','automation','total']].eq(0).all(axis=1))

    # Finally attach all OCCUPATION level metrics of exposure and usage
    collapsed = collapsed.merge(
        temp,
        how = "left",
        on = "O*NET-SOC Code"
    )
    
    socs = read_csv(os.path.join(rsc_path, "crosswalk.csv")).dropna(subset = 'cps_code').assign(
        # Processing values to facilitate easier merging later on
        cps_code  = lambda x: x['cps_code'].astype(int).astype(str).str.zfill(4),
        major_code = lambda x: x["Code"].str[:2].astype(int),
    )
    majors = read_csv(os.path.join(rsc_path, "soc_major.csv"))
    socs = socs.merge(
        majors,
        how = "left",
        on = "major_code"
    )[["cps_code", "major_occ"]].drop_duplicates() 
    
    external = crosswalk.merge(
        collapsed,
        how = "left",
        left_on = "Code",
        right_on = "O*NET-SOC Code"
    )
    
    external = external.groupby('no_detail').agg({
        'cps_code': lambda x: x.mode()[0] if len(x.mode()) > 0 else np.nan,
        'mean_rating_human_alpha': 'mean',
        'mean_rating_human_beta': 'mean',
        'mean_rating_human_gamma': 'mean',
        'gpt4_rubric1_alpha': 'mean',
        'gpt4_rubric1_beta': 'mean',
        'gpt4_rubric1_gamma': 'mean',
        'gpt4_rubric2_beta': 'mean',
        'gpt4_automation': 'mean',
        'pct_of_convs': 'sum',
        'augmentation': 'mean',
        'automation': 'mean',
        'total': 'mean'
    }).reset_index().merge(
        oes,
        how = 'left',
        left_on = 'no_detail',
        right_on = 'OCC_CODE'
    )
    
    def get_wp(var, total, weights):
            # Weighted sum of the metric at hand
            w_sum = (var * weights).sum()
            # Total weight (magnified by n(tasks) for each occupation in the usage data)
            w_total = (weights * total * ~var.isna()).sum()

            return (w_sum / w_total) if w_total > 0 else pandas.NA 

    external = external.groupby('cps_code').agg(
        mean_rating_human_alpha = pandas.NamedAgg(
            column = 'mean_rating_human_alpha',
            aggfunc = lambda x, w=external["TOT_EMP"]: get_wp(x, 1, w.loc[x.index])
        ),
        mean_rating_human_beta = pandas.NamedAgg(
            column = 'mean_rating_human_beta',
            aggfunc = lambda x, w=external["TOT_EMP"]: get_wp(x, 1, w.loc[x.index])
        ),
        mean_rating_human_gamma = pandas.NamedAgg(
            column = 'mean_rating_human_gamma',
            aggfunc = lambda x, w=external["TOT_EMP"]: get_wp(x, 1, w.loc[x.index])
        ),
        gpt4_rubric1_alpha = pandas.NamedAgg(
            column = 'gpt4_rubric1_alpha',
            aggfunc = lambda x, w=external["TOT_EMP"]: get_wp(x, 1, w.loc[x.index])
        ),
        gpt4_rubric1_beta = pandas.NamedAgg(
            column = 'gpt4_rubric1_beta',
            aggfunc = lambda x, w=external["TOT_EMP"]: get_wp(x, 1, w.loc[x.index])
        ),
        gpt4_rubric1_gamma = pandas.NamedAgg(
            column = 'gpt4_rubric1_gamma',
            aggfunc = lambda x, w=external["TOT_EMP"]: get_wp(x, 1, w.loc[x.index])
        ),
        gpt4_rubric2_beta = pandas.NamedAgg(
            column = 'gpt4_rubric2_beta',
            aggfunc = lambda x, w=external["TOT_EMP"]: get_wp(x, 1, w.loc[x.index])
        ),
        gpt4_automation = pandas.NamedAgg(
            column = 'gpt4_automation',
            aggfunc = lambda x, w=external["TOT_EMP"]: get_wp(x, 1, w.loc[x.index])
        ),
        pct_of_convs = pandas.NamedAgg(
            column = 'pct_of_convs',
            aggfunc = lambda x: x.sum()
        ),
        augmentation = pandas.NamedAgg(
            column = 'augmentation',
            aggfunc = lambda x, w=external["TOT_EMP"]: get_wp(x, 1, w.loc[x.index])
        ),
        automation = pandas.NamedAgg(
            column = 'automation',
            aggfunc = lambda x, w=external["TOT_EMP"]: get_wp(x, 1, w.loc[x.index])
        ),
        total = pandas.NamedAgg(
            column = 'total',
            aggfunc = lambda x, w=external["TOT_EMP"]: get_wp(x, 1, w.loc[x.index])
        ),
        employment_count = pandas.NamedAgg(
            column = "TOT_EMP",
            aggfunc = lambda x: x.sum()
        )
    ).reset_index().merge(
        socs,
        how = "left",
        left_on = "cps_code",
        right_on = "cps_code"
    )

    if fill_missing_tasks:
        external.to_csv(os.path.join(rsc_path, "full_map_" + samp + "_fm" + ".csv"), index = False)
    else:
        external.to_csv(os.path.join(rsc_path, "full_map_" + samp + ".csv"), index = False)
    
    return external

def month_helper(month: int):
    # Just a helper function to do one-line month iteration
    if (month % 12) == 0:
        return 12
    else:
        return month % 12