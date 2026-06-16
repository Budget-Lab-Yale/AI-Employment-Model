import numpy as np
import pandas as pd
import seaborn as sns
import os
import subprocess
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from scipy import stats as scipy_stats
from scipy.stats import spearmanr, pearsonr, t

def current_git_branch(path):
    # Branch of the repo containing `path`. `git -C` walks up to the enclosing
    # .git, so this resolves the tracker repo even though `path` is a subdir of
    # it (and a different repo from the one this code lives in).
    try:
        result = subprocess.run(
            ["git", "-C", path, "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

def clean_crosswalk(rsc_path: str, tracker_path: str):
    branch = current_git_branch(tracker_path)

    if branch == "main":
        print("=" * 72)
        print("ERROR: the tracker repository is on the 'main' branch -- aborting.")
        print(f"  tracker_path: {tracker_path}")
        print("  No outputs were written. Switch the tracker repo to a working/")
        print("  staging branch and re-run, e.g.:")
        print(f"    git -C {tracker_path} switch <branch>")
        print("=" * 72)
        return

    if branch is None:
        print(f"WARNING: could not determine the git branch of the tracker repo at {tracker_path}")
    else:
        print(f"Writing outputs to tracker repo on git branch: '{branch}'  ({tracker_path})")

    oes = pd.read_csv(os.path.join(rsc_path, 'national_M2024_dl.csv'))[['OCC_CODE', 'TOT_EMP', 'OCC_TITLE', 'O_GROUP']]

    crosswalk = pd.read_csv(os.path.join(rsc_path, "crosswalk.csv")).dropna(subset = 'cps_code')[["cps_code", "Code"]].assign(
        # Processing values to facilitate easier merging later on
        cps_code  = lambda x: x['cps_code'].astype(int).astype(str).str.zfill(4),
        #major_code = lambda x: x["Code"].str[:2].astype(int),
        soc2018 = lambda x: x["Code"].str[:7]
    )

    usage = clean_usage(rsc_path)
    master = clean_master(rsc_path)

    onet = usage.merge(
        master,
        on = "soc2018",
        how = "outer"
    )

    cps = crosswalk.groupby('soc2018').agg({
        'cps_code': lambda x: x.mode()[0] if len(x.mode()) > 0 else np.nan
    }).reset_index().merge(
        oes,
        how = 'left',
        left_on = 'soc2018',
        right_on = 'OCC_CODE'
    ).merge(
        onet.drop(columns='pca'),
        how = 'left',
        on = 'soc2018'
    )

    def get_wp(var, total, weights):
        # Weighted sum of the metric at hand
        w_sum = (var * weights).sum()
        # Total weight (magnified by n(tasks) for each occupation in the usage data)
        w_total = (weights * total * ~var.isna()).sum()

        return (w_sum / w_total) if w_total > 0 else pd.NA 

    def wp_agg(col, w=cps["TOT_EMP"]):
        return pd.NamedAgg(column=col, aggfunc=lambda x, w=w: get_wp(x, 1, w.loc[x.index]))

    agg_dict = {
        'AIOE':                   wp_agg('AIOE'),
        'dv_rating_beta':         wp_agg('dv_rating_beta'),
        'human_rating_beta':      wp_agg('human_rating_beta'),
        'genaiexp_estz_total':    wp_agg('genaiexp_estz_total'),
        'genaiexp_estz_core':     wp_agg('genaiexp_estz_core'),
        'ai_applicability_score': wp_agg('ai_applicability_score'),
        #'pct_of_convs':           pd.NamedAgg(column = 'pct_of_convs', aggfunc =  lambda x: x.sum()),
        'augmentation':           wp_agg('augmentation'),
        'automation':             wp_agg('automation'),
        'total':                  wp_agg('total'),
        'total_fm':               wp_agg('total_fm'),
        'employment_count':       pd.NamedAgg(column="TOT_EMP", aggfunc=lambda x: x.sum()),
    }

    cps = cps.groupby('cps_code').agg(**agg_dict).reset_index()

    measures = ['AIOE', 'dv_rating_beta', 'human_rating_beta', 'genaiexp_estz_total', 'genaiexp_estz_core', 'ai_applicability_score']

    cps_no_null = cps[['cps_code'] + measures].dropna()

    cps = cps.merge(
        do_pca(cps_no_null, 'cps_code', measures),
        on  = 'cps_code',
        how = 'left'
    )

    os.makedirs(tracker_path, exist_ok=True)
    onet.to_csv(os.path.join(tracker_path, "soc_data.csv"), index=False)
    cps.to_csv(os.path.join(tracker_path, "cps_2020_data.csv"), index=False)

    return

def clean_usage(rsc_path: str):
    onet_tasks = pd.read_csv(os.path.join(rsc_path, 'onet_task_statements_old.csv'))[["O*NET-SOC Code", "Task", "Task ID", "Task Type", "Title"]].assign(
        t_wgt = lambda x: np.where(x["Task Type"]=="Core", 1, .5)
    ).drop(["Task Type"], axis=1)
    onet_tasks["Task"] = onet_tasks["Task"].str.lower().str.strip()
    onet_tasks["n_occurrences"] = onet_tasks.groupby("Task")["Title"].transform("nunique")
    onet_tasks["n_occurrences"] = onet_tasks["n_occurrences"].fillna(1)
        
    old_new = pd.read_csv(os.path.join(rsc_path, "2010_to_2019_Crosswalk.csv"))
        
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

    usage_raw = pd.read_csv(os.path.join(rsc_path, "automation_vs_augmentation_by_task_v4.csv"))
    usage_raw["task_name"] = usage_raw["task_name"].str.lower().str.strip()

    usage_raw = usage_raw.drop('filtered', axis = 1)
    cols = ['validation', 'task_iteration', 'learning', 'feedback_loop', 'directive']
    new_total = usage_raw[cols].sum(axis=1)
    for col in cols:
        usage_raw[col] = usage_raw[col] / new_total
    
    frames = []

    for fill in [False, True]:
        if fill:
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
        onet_tasks = onet_tasks.assign(
            augmentation = lambda x: x["validation"] + x["task_iteration"] + x["learning"],
            automation   = lambda x: x["directive"] + x["feedback_loop"],
           # pct_of_convs = lambda x: 100 * (x["pct"] / x["n_occurrences"]) / (x["pct"] / x["n_occurrences"]).sum()
        ).drop(["validation", "task_iteration", "learning", "directive", "feedback_loop", "pct"], axis = 1)

        onet_tasks['total'] = 1 if fill else onet_tasks['automation'] + onet_tasks['augmentation']
        onet_tasks[["augmentation", "automation", "total"]] = onet_tasks[["augmentation", "automation", 'total']].apply(lambda x: x * onet_tasks['t_wgt'])

        temp = onet_tasks.groupby("O*NET-SOC Code").agg({
            'augmentation': 'sum',
            'automation':   'sum',
            'total' :    'sum',
            #'pct_of_convs': 'sum'
        }).reset_index().assign(
            augmentation = lambda x: x['augmentation'],
            automation   = lambda x: x['automation'],
            total = lambda x: x['total'],
            #pct_of_convs = lambda x: x['pct_of_convs']
        )
        
        # Sets tasks with 0s in augmentation, automation, and total to NaN so that they are ignored in future calculations.
        # This is only important for the method in which we do not fill tasks. In that configuration, 0s across the board (except for total) is a valid entry
        # In this configuration, it would only serve to confound the future aggregations
        if not fill:
            temp[['augmentation','automation','total']] = temp[['augmentation','automation','total']].mask(temp[['augmentation','automation','total']].eq(0).all(axis=1))
        
        temp["fm"] = fill
        frames.append(temp)
        
    usage = pd.concat(frames, ignore_index = True).assign(
        soc2018 = lambda x: x["O*NET-SOC Code"].str[:7]
    )

    df_fm = usage[usage['fm'] == True].copy()
    df_nonfm = usage[usage['fm'] == False].copy()

    df_fm_agg = df_fm.groupby('soc2018')[['total']].sum().rename(columns={'total': 'total_fm'})
    df_nonfm_agg = df_nonfm.groupby('soc2018')[['augmentation', 'automation', 'total']].sum() #, 'pct_of_convs'
    

    usage = df_nonfm_agg.join(df_fm_agg, how='outer').reset_index()

    return usage

def clean_master(rsc_path: str):
    master = pd.read_csv(os.path.join(rsc_path, "master_no_webb.csv"))
    master_no_null = master.dropna()
    measures = [col for col in master_no_null.columns if col not in ["soc2018"]]

    master = master.merge(
        do_pca(master_no_null, "soc2018", measures),
        on  = 'soc2018',
        how = 'left'
    )
    return master   

def do_pca(this: pd.DataFrame, dex: str, measures: list[str]):
    this = this.copy()  # avoid SettingWithCopyWarning since `this` is a .dropna() slice
    # Standardize all measures to z-scores and add as new columns with _z suffix
    scaler = StandardScaler()
    z_scores = scaler.fit_transform(this[measures])
    for i, col in enumerate(measures):
        this[f"{col}_z"] = z_scores[:, i]

    # Create list of z-score column names
    measures_z = [f"{col}_z" for col in measures]

    # PCA on ALL measures
    pca_all = PCA(n_components=1)
    pca_all.fit(this[measures_z])
    pca_weights_all = pca_all.components_[0]
    this["pca"] = this[measures_z] @ pca_weights_all

    # Print PCA weights explicitly
    # print("="*80)
    # print("PCA WEIGHTS (WITHOUT WEBB)")
    # print("="*80)
    # print("\nPCA Weights (All Measures):")
    # print(dex)
    # weights_df_all = pd.DataFrame({
    #     'Measure': measures,
    #     'PCA Weight': pca_weights_all
    # }).sort_values('PCA Weight', ascending=False)
    # print(weights_df_all.to_string(index=False))
    # print(f"\nVariance explained by PC1: {pca_all.explained_variance_ratio_[0]:.2%}")

    return this[[dex, "pca"]]

