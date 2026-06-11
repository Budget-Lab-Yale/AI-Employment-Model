import pandas
import numpy
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import os
import math
import subprocess

# Canonical industry codes (also used to locate the per-industry input files).
INDUSTRIES = [
    'natural_resources_and_mining', 'construction', 'manufacturing', 'trade_transportation_and_utilities',
    'information', 'financial_activities', 'professional_and_business_services', 'education_and_health_services',
    'leisure_and_hospitality', 'other_services',
]

INDUSTRY_MAP = dict(zip(
    INDUSTRIES,
    ['Natural Resources and Mining', 'Construction', 'Manufacturing', 'Trade, Transportation, and Utilities',
     'Information', 'Financial Activities', 'Professional and Business Services', 'Education and Health Services',
     'Leisure and Hospitality', 'Other Services']
))

SERIES_MAP = {
    "AI (Nov22)":     "Baseline Nov 2022 (AI)",
    "Jan21":          "Baseline Jan 2021",
    "Jan22":          "Baseline Jan 2022",
    "Jul22":          "Baseline Jul 2022",
    "AI":             "AI (baseline Nov 2022)",
    "Computers":      "Computers (baseline Jan 1984)",
    "Control":        "Control (baseline Jan 2016)",
    "Internet":       "Internet (baseline Jan 1996)",
    "dissimilarity":  "Dissimilarity",
    "_15_26":         "15-'26",
    "_22_26":         "22-'26",
    "1-5 weeks":      "<5 Weeks",
}

# Series labels for the top-industries chart (industry.csv column -> label).
TOP_INDUSTRY_MAP = {
    "all":                                "All Sectors",
    "information":                        "Information",
    "financial_activities":               "Financial Activities",
    "professional_and_business_services": "Professional and Business Services",
}

# Variant labels keyed by the sub-directory they live in.
DISSIM_VARIANTS = {"indexed": "indexed", "rolling": "rolling"}
EXPOSURE_VARIANTS = {"feb26": "observed", "feb26_fm": "missing-as-zero"}

# Family folders under tracker_path/ that each chart's subdir lives inside.
DISSIMILARITY_DIR = "dissimilarity"
AI_METRICS_DIR    = "ai-metrics"
EFFECTS_DIR       = "sdid"

# Columns that identify/structure an exposure sheet but are never a metric.
EXPOSURE_ID_COLS = {"time", "group", "group_val"}

DURATION_MAP = {"1-5 Weeks": "<5 Weeks"}
AUTO_AUG_PCT_MAP   = {"percent_automated": "Automated",   "percent_augmented": "Augmented"}
AUTO_AUG_LEVEL_MAP = {"automation": "Automation",         "augmentation": "Augmentation"}
EXPOSURE_GROUP_MAP = {
    "percent_lowest_exposed":  "Lowest Exposure Group",
    "percent_middle_exposed":  "Middle Exposure Group",
    "percent_highest_exposed": "Highest Exposure Group",
}


def read_csv(path):
    return pandas.read_csv(path)


def _round_sig(x, sig):
    if pandas.isna(x) or x == 0:
        return x
    return round(x, -int(math.floor(math.log10(abs(x)))) + (sig - 1))


# subdir -> filename stem under effects_path. Each stem has a "<stem>_data.csv"
# (Synthetic/Treated) and a "<stem>_impact_data.csv" (Difference + CIs).
EFFECTS = {
    "employment-shares":   "sdid_emp_share_sa",
    "real-hourly-wages":   "sdid_log_real_hrly_wage_sa",
    "unemployment-rate":   "sdid_logurate_sa",
}


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


def clean_output(out_path_full: str, tracker_path: str, effects_path: str):
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

    clean_dissimilarity(out_path_full, tracker_path)
    clean_ai_metrics(out_path_full, tracker_path)
    clean_effects(effects_path, tracker_path)
    return


def clean_effects(effects_path: str, tracker_path: str):
    # Each metric pairs a "_data" file (Synthetic/Treated lines) with an
    # "_impact_data" file (the Difference line plus its confidence band). They
    # are stacked: the Synthetic/Treated block (interleaved by quarter) first,
    # then the Difference block. Both files live in a per-metric subdirectory
    # named after the stem without its "sdid_" prefix (e.g. "emp_share_sa").
    for subdir, base in EFFECTS.items():
        input_subdir = base[len("sdid_"):] if base.startswith("sdid_") else base
        data_path = os.path.join(effects_path, input_subdir, base + "_data.csv")
        imp_path  = os.path.join(effects_path, input_subdir, base + "_impact_data.csv")

        frames = []
        if os.path.exists(data_path):
            frames.append(clean_sheet(read_csv(data_path),
                                      time_col="yq", sig_figs=6, sort_time=True, dropna=False))
        if os.path.exists(imp_path):
            frames.append(clean_sheet(read_csv(imp_path),
                                      time_col="yq", value_col="d", force_series="Difference",
                                      ci_cols=("lci", "uci"), sig_figs=6, sort_time=True, dropna=False))
        if not frames:
            continue

        out = pandas.concat(frames, ignore_index=True)
        for c in ["lower_ci", "upper_ci"]:
            if c not in out.columns:
                out[c] = numpy.nan
        out = out[["time", "series", "value", "lower_ci", "upper_ci"]]

        out_dir = os.path.join(tracker_path, EFFECTS_DIR, subdir)
        os.makedirs(out_dir, exist_ok=True)
        out.to_csv(os.path.join(out_dir, "data.csv"), index=False)
    return


def clean_dissimilarity(out_path_full: str, tracker_path: str):
    # Each chart is described declaratively, then built by the same loop. A chart
    # is a list of "frames"; every frame is one call to clean_sheet that says
    # which file to read, what to keep, and how to label it. Charts that come in
    # indexed/rolling flavours list one frame per variant; charts that don't
    # (graduates, technological changes in this version) list a single frame
    # with variant=None and therefore emit no variant column.

    def variant_frames(filename, **kw):
        # One frame per existing variant sub-directory (indexed / rolling).
        # Rolling files use a calendar date (YYYY-MM-01) as their time axis
        # rather than the integer months_gone offset used by indexed files.
        frames = []
        for subdir, label in DISSIM_VARIANTS.items():
            path = os.path.join(out_path_full, subdir, filename)
            if os.path.exists(path):
                extra = {"time_col": "period"} if subdir == "rolling" else {}
                frames.append(dict(path=path, variant=label, **extra, **kw))
        return frames

    def flat_frame(filename, **kw):
        # A single, variant-less frame (file sits directly under out_path_full).
        return [dict(path=os.path.join(out_path_full, filename), variant=None, **kw)]

    charts = {
        # Technological changes: cols AI/Computers/Control/Internet -> series. No variant.
        "total-labor-force-historical": flat_frame(
            os.path.join("indexed", "dissimilarity.csv"),
            label_map=SERIES_MAP, round_to=6,
        ),

        # Recent baselines: cols AI (Nov22)/Jan21/Jan22/Jul22 -> series.
        "total-labor-force-recent": variant_frames(
            "recent.csv", label_map=SERIES_MAP, round_to=6,
        ),

        # By industry: one file per industry; baseline cols -> series, industry tagged
        # with its hyphenated slug. Now covers every industry, not just a few.
        "by-industry-recent": [
            frame
            for industry in INDUSTRIES
            for frame in variant_frames(
                industry + ".csv",
                label_map=SERIES_MAP,
                industry=industry.replace("_", "-"),
                round_to=6,
            )
        ],

        # Major industries historical: every.csv columns are already the display
        # names, so each becomes a series as-is.
        "major-industries-historical": variant_frames(
            "every.csv", round_to=6,
        ),

        # Top industries: keep "all" + the three most-exposed industries from
        # industry.csv and relabel them as the series.
        "top-industries-recent": variant_frames(
            "industry.csv",
            keep_cols=list(TOP_INDUSTRY_MAP),
            label_map=TOP_INDUSTRY_MAP,
            round_to=6,
        ),

        # New vs older graduates: two single-column files, each forced to one
        # series label. No variant column.
        "new-vs-older-grads-historical": [
            dict(path=os.path.join(out_path_full, "recent_grads_dissimilarity" + key + ".csv"),
                 variant=None, force_series=key, label_map=SERIES_MAP, round_to=4)
            for key in ["_15_26", "_22_26"]
        ],

        # Recent graduates since 2021: the "dissimilarity" column over a date axis.
        # No variant column.
        "new-vs-older-grads-recent": [
            dict(path=os.path.join(out_path_full, "recent_grads_dissimilarity_lone.csv"),
                 variant=None, time_col="time", label_map=SERIES_MAP, round_to=4)
        ],
    }

    for subdir, frames in charts.items():
        built = []
        for f in frames:
            path = f.pop("path")
            if not os.path.exists(path):
                continue
            # Keep recent all-NA months (e.g. Oct 2025) as blank rows in the frame.
            f.setdefault("dropna", False)
            built.append(clean_sheet(read_csv(path), **f))

        if not built:
            continue

        out = pandas.concat(built, ignore_index=True)
        out_dir = os.path.join(tracker_path, DISSIMILARITY_DIR, subdir)
        os.makedirs(out_dir, exist_ok=True)
        out.to_csv(os.path.join(out_dir, "data.csv"), index=False)

    return


def clean_ai_metrics(out_path_full: str, tracker_path: str):
    # Read each variant's exposure file once (feb26 = observed,
    # feb26_fm = missing-as-zero); only those present on disk are loaded.
    sheets = {}
    for variant in EXPOSURE_VARIANTS:
        path = os.path.join(out_path_full, "gpt4_rubric1_beta", variant, "exposure.csv")
        if os.path.exists(path):
            sheets[variant] = read_csv(path)

    # subdir, group, metrics, label_map, variants
    specs = [
        ("augmented-occupations-by-unemployment-duration", "DURUNEMP_cat",  ["percent_augmented"],       DURATION_MAP,        ["feb26", "feb26_fm"]),
        ("augmented-tasks-by-unemployment-duration",       "DURUNEMP_cat",  ["augmentation"],            DURATION_MAP,        ["feb26", "feb26_fm"]),
        ("automated-occupations-by-unemployment-duration", "DURUNEMP_cat",  ["percent_automated"],       DURATION_MAP,        ["feb26", "feb26_fm"]),
        ("automated-tasks-by-unemployment-duration",       "DURUNEMP_cat",  ["automation"],              DURATION_MAP,        ["feb26", "feb26_fm"]),
        ("automated-vs-augmented-occupations",             "EMPSTAT",       list(AUTO_AUG_PCT_MAP),      AUTO_AUG_PCT_MAP,    ["feb26", "feb26_fm"]),
        ("automation-vs-augmentation-usage",               "EMPSTAT",       list(AUTO_AUG_LEVEL_MAP),    AUTO_AUG_LEVEL_MAP,  ["feb26", "feb26_fm"]),
        ("highest-exposure-share-among-unemployed",        "DURUNEMP_cat",  ["percent_highest_exposed"], DURATION_MAP,        ["feb26"]),
        ("task-exposure-among-unemployed",                 "DURUNEMP_cat",  ["tasks_exposed_pct"],       DURATION_MAP,        ["feb26"]),
        ("workers-by-exposure-level",                      "EMPSTAT",       list(EXPOSURE_GROUP_MAP),    EXPOSURE_GROUP_MAP,  ["feb26"]),
    ]

    for subdir, group, metrics, label_map, variants in specs:
        frames = []
        for variant in variants:
            if variant not in sheets:
                continue
            frames.append(
                clean_sheet(sheets[variant], variant=EXPOSURE_VARIANTS[variant],
                            group=group, metrics=metrics, label_map=label_map,
                            round_to=4, dropna=False)
            )

        out = pandas.concat(frames, ignore_index=True)
        if variants == ["feb26"]:
            out = out.drop(columns="variant")

        out_dir = os.path.join(tracker_path, AI_METRICS_DIR, subdir)
        os.makedirs(out_dir, exist_ok=True)
        out.to_csv(os.path.join(out_dir, "data.csv"), index=False)

    return


def clean_sheet(this: pandas.DataFrame, variant=None, *, group=None, metrics=None,
                series_from=None, label_map=None, time_col="months_gone",
                keep_cols=None, force_series=None, industry=None, industry_map=None,
                value_col=None, ci_cols=None, sig_figs=None, sort_time=False,
                dropna=True, round_to=None):
    # ------------------------------------------------------------------
    # Exposure (AI metrics) branch. Auto-selected by the `group_val`
    # column, which only exposure sheets carry.
    # ------------------------------------------------------------------
    if "group_val" in this.columns:
        if group is not None:
            this = this[this["group"] == group]

        if metrics is None:
            metrics = [c for c in this.columns if c not in EXPOSURE_ID_COLS]
        metrics = list(metrics)

        if series_from is None:
            series_from = "group_val" if len(metrics) == 1 else "metric"

        if series_from == "metric":
            out = pandas.melt(this, id_vars=["time"], value_vars=metrics,
                              var_name="series", value_name="value")
        else:
            metric = metrics[0]
            out = (this[["time", "group_val", metric]]
                   .rename(columns={"group_val": "series", metric: "value"}))

        if label_map is not None:
            out["series"] = out["series"].map(label_map).fillna(out["series"])
        if variant is not None:
            out = out.assign(variant=variant)

        # A stray non-numeric token (e.g. a non-standard missing marker) would
        # make the column object dtype; turn those into NaN so they read as blank.
        out["value"] = pandas.to_numeric(out["value"], errors="coerce")
        if dropna:
            out = out.dropna(subset=["value"])
        if round_to is not None:
            out["value"] = out["value"].round(round_to)

        ordered = ["time", "series"] + (["variant"] if variant is not None else []) + ["value"]
        return out[ordered].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Dissimilarity / effects branch.
    #   * default: every non-time column becomes a series (wide melt)
    #   * value_col set: that single column is the value (no melt), used by the
    #     effects "impact" files, optionally with a confidence band (ci_cols).
    # ------------------------------------------------------------------
    if time_col and time_col in this.columns and time_col != "time":
        this = this.rename(columns={time_col: "time"})
    elif time_col and time_col != "time" and "time" not in this.columns:
        # A specific, non-default time column was requested (e.g. "period" for
        # rolling files) but it isn't in the sheet. Silently falling back to the
        # first column here is how integer months_gone leaked into rolling output
        # before — so fail loudly instead. The fix lives upstream: the
        # corresponding run_dissimilarity pivot must use that column as its index.
        raise KeyError(
            f"clean_sheet: requested time_col {time_col!r} not found in columns "
            f"{list(this.columns)}. The source file was likely generated before "
            f"run_dissimilarity was switched to pivot on {time_col!r}; regenerate it.")
    elif time_col and "time" not in this.columns:
        # time_col was "time" (the default signal) but the column isn't named that;
        # use the first column as the time axis, keeping its values as-is.
        this = this.rename(columns={this.columns[0]: "time"})

    if value_col is not None:
        out = this[["time", value_col]].rename(columns={value_col: "value"})
        if ci_cols is not None:
            lo, hi = ci_cols
            out["lower_ci"] = this[lo].values
            out["upper_ci"] = this[hi].values
    else:
        value_cols = list(keep_cols) if keep_cols is not None else [c for c in this.columns if c != "time"]
        out = pandas.melt(this, id_vars="time", value_vars=value_cols,
                          var_name="series", value_name="value")

    # Force a constant series label (single-column / impact files).
    if force_series is not None:
        out["series"] = force_series
    if label_map is not None:
        out["series"] = out["series"].map(label_map).fillna(out["series"])

    # Attach a constant industry column when requested.
    if industry is not None:
        out["industry"] = industry
        if industry_map is not None:
            out["industry"] = out["industry"].map(industry_map).fillna(out["industry"])

    if variant is not None:
        out = out.assign(variant=variant)

    # A stray non-numeric token (e.g. a non-standard missing marker for a recent
    # month) makes the melted column object dtype; coerce to NaN so it reads as a
    # blank. Combined with dropna=False the row is kept rather than removed.
    num_cols = ["value"] + [c for c in ("lower_ci", "upper_ci") if c in out.columns]
    for c in num_cols:
        out[c] = pandas.to_numeric(out[c], errors="coerce")

    if dropna:
        out = out.dropna(subset=["value"])

    if sig_figs is not None:
        for c in num_cols:
            out[c] = out[c].apply(lambda v: _round_sig(v, sig_figs))
    elif round_to is not None:
        for c in num_cols:
            out[c] = out[c].round(round_to)

    if sort_time:
        out = out.sort_values("time", kind="stable")

    ordered = (["time"]
               + (["industry"] if industry is not None else [])
               + ["series"]
               + (["variant"] if variant is not None else [])
               + ["value"]
               + [c for c in ("lower_ci", "upper_ci") if c in out.columns])
    return out[ordered].reset_index(drop=True)