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


def _month_range(start_yr: int, start_mn: int, end_yr: int, end_mn: int) -> list:
    """
    Returns a list of zero-padded "YYYYMM" strings covering every calendar month
    from (start_yr, start_mn) through (end_yr, end_mn), inclusive.

    Iteration advances month-by-month; when the month counter exceeds 12 it
    rolls over to January of the next year.  Tuple comparison handles the
    year boundary without any modulo sentinel arithmetic.

    Example: _month_range(2023, 11, 2024, 2)
             → ["202311", "202312", "202401", "202402"]
    """
    stamps = []
    yr, mn = start_yr, start_mn
    while (yr, mn) <= (end_yr, end_mn):
        stamps.append(str(yr) + str(cur_mn := mn).zfill(2))   # zero-pad month
        mn += 1
        if mn > 12:          # roll over December → January of next year
            mn = 1
            yr += 1
    return stamps


def build_mismatch(ID: str, start: str, end: str, freq = 1, occ_subset = '', ind_subset = '',
                   pre_trend = 0, rolling = False, rolling_lag = 12):
    """
    Constructs a time series of occupational dissimilarity (mismatch) scores.

    The dissimilarity index measures how much the occupational distribution of
    employment has shifted relative to a baseline.  For each target month the
    index sums, across all occupation codes, half the absolute difference
    between the share of employment in that occupation in the current period
    and in the baseline period.  A value of 0 means no change; a value of 100
    means the current and baseline distributions are completely non-overlapping.

    Parameters
    ----------
    ID          : str  – Subdirectory name that identifies which model/data variant to use.
    start       : str  – First target month in "YYYY.MM" format (e.g. "2019.01").
    end         : str  – Last target month in "YYYY.MM" format, inclusive.
    freq        : int  – Number of months to average together to form a single
                         observation (1 = monthly, 3 = quarterly rolling avg, etc.).
    occ_subset  : str  – Optional "first.last" occupation-code filter, e.g. "1000.3000".
                         Empty string = keep all occupations.
    ind_subset  : str  – Optional "first.last" industry-code filter.
                         Empty string = keep all industries.
    pre_trend   : int  – Number of months *before* `start` to also compute and include
                         in the output, so the caller can inspect the pre-period trend.
    rolling     : bool – If True, use a rolling baseline: compare each month to the
                         corresponding month exactly `rolling_lag` months earlier,
                         rather than to a single fixed baseline.
    rolling_lag : int  – How many months back the rolling baseline lags (default 12,
                         i.e. year-over-year comparison).

    Returns
    -------
    pandas.DataFrame with columns:
        dissimilarity  – the index value (0–100) for each period
        months_gone    – integer offset from `start` (negative = pre-trend months)
    """

    # ── Parse "YYYY.MM" strings into separate integer year/month components ──
    start = start.split('.')
    end   = end.split('.')

    yr_dex = int(start[0])
    mn_dex = int(start[1])
    end_yr = int(end[0])
    end_mn = int(end[1])

    # Normalize in case a caller passes a month value outside [1,12]
    # (e.g. month 13 → January of the following year).
    yr_dex += (mn_dex - 1) // 12
    mn_dex  = (mn_dex - 1) % 12 + 1
    end_yr += (end_mn - 1) // 12
    end_mn  = (end_mn - 1) % 12 + 1

    # October 2025 is unavailable
    oct_25 = "202510"

    # ── Determine the earliest month that must be read from disk ──
    # We must load data starting (freq-1) months before yr_dex/mn_dex so that
    # the very first target-month window average has all its constituent months.
    # pre_trend additional months may push the read start even further back.
    # If rolling, we also need rolling_lag extra months so the lag window exists.
    read_yr = yr_dex
    read_mn = mn_dex - (freq - 1) - pre_trend
    if rolling:
        read_mn -= rolling_lag   # need an extra rolling_lag months of history

    # Re-normalize after the backward shift (result could be month 0 or negative)
    read_yr += (read_mn - 1) // 12
    read_mn  = (read_mn - 1) % 12 + 1

    # ── Bulk-load every required month from parquet into a single DataFrame ──
    # This avoids re-reading files repeatedly as we iterate over target months.
    all_months = load_all_months(ID, read_yr, read_mn, end_yr, end_mn, occ_subset, ind_subset, oct_25)

    # ── Iterate over every target month and compute its dissimilarity score ──
    months = None
    for stamp in _month_range(yr_dex, mn_dex, end_yr, end_mn):
        if months is None:
            # First iteration: calc_mis_month also establishes the baseline
            # distribution (stored in the "base" column for fixed-baseline mode).
            months = calc_mis_month(stamp, freq, oct_25, all_months,
                                    rolling=rolling, rolling_lag=rolling_lag)
            if rolling:
                # Rolling mode: no persistent "base" column; just keep the score column.
                months = months[["OCC", stamp]]
            else:
                # Fixed-baseline mode: retain "base" so subsequent months are
                # always compared against the same initial distribution.
                months = months[["OCC", "base", stamp]]
        else:
            # Subsequent iterations: pass the existing baseline so all months
            # share the same reference distribution (fixed mode only).
            this_month = calc_mis_month(stamp, freq, oct_25, all_months,
                                        base=None if rolling else months[["OCC", "base"]],
                                        rolling=rolling, rolling_lag=rolling_lag)
            # Merge the new score column onto the growing wide table by occupation code.
            months = months.merge(
                this_month[["OCC", stamp]],
                how = 'outer',   # keep all OCC codes even if one month has no data
                on  = 'OCC'
            )

    # ── Optionally prepend pre-trend months (computed but not part of the main window) ──
    if pre_trend > 0:
        # The pre-trend window runs from (start − pre_trend months) to (start − 1 month).
        pt_end_mn = mn_dex - 1
        pt_end_yr = yr_dex
        if pt_end_mn < 1:       # handle January boundary (month 0 → December prior year)
            pt_end_mn += 12
            pt_end_yr -= 1

        pt_start_mn = mn_dex - pre_trend
        pt_start_yr = yr_dex
        pt_start_yr += (pt_start_mn - 1) // 12
        pt_start_mn  = (pt_start_mn - 1) % 12 + 1

        for stamp in _month_range(pt_start_yr, pt_start_mn, pt_end_yr, pt_end_mn):
            this_month = calc_mis_month(stamp, freq, oct_25, all_months,
                                        base=None if rolling else months[["OCC", "base"]],
                                        rolling=rolling, rolling_lag=rolling_lag)
            months = months.merge(
                this_month[["OCC", stamp]],
                how = 'outer',
                on  = 'OCC'
            )

    # ── Aggregate occupation-level scores into a single aggregate index per month ──
    # Drop the "OCC" identifier column (and "base" in fixed mode) before summing,
    # so that each remaining column is a single month's per-occupation contributions.
    # Summing across occupations gives the total dissimilarity index for that month.
    # min_count=1 ensures that an all-NaN month returns NaN rather than 0.
    cols_to_drop = ["OCC"] if rolling else ["OCC", "base"]
    data = (
        months
        .drop(cols_to_drop, axis=1)
        .sum(axis=0, min_count=1)   # column-wise sum: one scalar per YYYYMM stamp
        .reset_index()
        .rename(columns={"index": "MONTH", 0: "dissimilarity"})
        .sort_values('MONTH')
    )

    # ── Attach a "months_gone" offset so callers can align series at time 0 ──
    # Pre-trend months get negative offsets; the first target month is 0.
    deltas = data.reset_index()
    deltas["months_gone"] = deltas.index - pre_trend

    if rolling:
        deltas["period"] = pandas.to_datetime(deltas["MONTH"], format="%Y%m").dt.strftime("%Y-%m-02")

    deltas = deltas.drop(['index', "MONTH"], axis=1)

    return deltas


def calc_mis_month(this_month: str, freq: int, oct_25: str, all_months: pandas.DataFrame,
                   base = None, rolling = False, rolling_lag = 12):
    """
    Computes the per-occupation dissimilarity contributions for a single target month.

    The dissimilarity index for one occupation in one month equals:
        | share_current(occ) − share_baseline(occ) | × 100 / 2

    Summing this across all occupations yields the aggregate index (0–100).

    Parameters
    ----------
    this_month  : str             – Target "YYYYMM" stamp.
    freq        : int             – Window size in months for the moving average.
    oct_25      : str             – The flagged month stamp ("202510") to skip.
    all_months  : DataFrame       – Pre-loaded panel with columns OCC, WTFINL, MONTH.
    base        : DataFrame|None  – In fixed-baseline mode, the caller passes the
                                    baseline OCC/WTFINL distribution computed at the
                                    first iteration so all months share the same reference.
                                    None on the first call (baseline is derived here)
                                    or when rolling=True.
    rolling     : bool            – If True, compare to the distribution rolling_lag
                                    months earlier instead of a fixed baseline.
    rolling_lag : int             – Lag in months for the rolling baseline.

    Returns
    -------
    DataFrame with columns: OCC, base, WTFINL, <this_month>
        where <this_month> holds the per-occupation dissimilarity contribution.
    """

    # ── Special handling for the flagged October 2025 month ──
    # The raw data for this month is unavailable, so we mark every
    # occupation's contribution as NaN, preserving the shape for merging.
    if this_month == oct_25:
        month = all_months[all_months["MONTH"] == this_month][["OCC", "WTFINL"]]
        
        if base is None:
            # First iteration: create a baseline and current-month frame, both NaN.
            base = month[['OCC']].copy()
            base['base'] = float('nan')
            month = base.copy()
            month[this_month] = float('nan')
        else:
            # Subsequent iteration: keep the existing baseline but mark this month NaN.
            month = base[['OCC', 'base']].copy()
            month[this_month] = float('nan')
        
        return month

    # ── Determine the start of the freq-month averaging window ──
    # If freq=3 and this_month="202303", the window covers 202301–202303.
    cur_yr = int(this_month[:4])
    cur_mn = int(this_month[4:6]) - (freq - 1)   # shift back (freq-1) months

    # Normalize: shifting back may produce month ≤ 0, crossing a year boundary.
    cur_yr += (cur_mn - 1) // 12
    cur_mn  = (cur_mn - 1) % 12 + 1
    cur_start = str(cur_yr) + str(cur_mn).zfill(2)

    # ── Compute the freq-month average employment distribution for this window ──
    month = _window_average(cur_start, freq, all_months, oct_25)

    # If _window_average returned empty (one or more required months missing),
    # return a frame of NaNs so the merge in build_mismatch still has an OCC key.
    if month.empty:
        result = pandas.DataFrame({"OCC": pandas.Series(dtype=str)})
        result[this_month] = float('nan')
        if not rolling and base is not None:
            result = base[['OCC', 'base']].copy()
            result[this_month] = float('nan')
        return result

    if rolling:
        # ── Rolling baseline: find the window that is rolling_lag months earlier ──
        lag_yr = int(this_month[:4])
        lag_mn = int(this_month[4:6]) - rolling_lag - (freq - 1)

        lag_yr += (lag_mn - 1) // 12
        lag_mn  = (lag_mn - 1) % 12 + 1
        lag_month_stamp = str(lag_yr) + str(lag_mn).zfill(2)

        lag_month = _window_average(lag_month_stamp, freq, all_months, oct_25)

        if lag_month.empty:
            # Lag period has missing data; mark as NaN and return.
            month['base'] = float('nan')
            month[this_month] = float('nan')
            return month

        # Rename the lag window's WTFINL to "base" and merge with current window.
        lag_month = lag_month.rename(columns={'WTFINL': 'base'})
        month = lag_month.merge(month, how='outer', on='OCC')

    else:
        # ── Fixed baseline: use the passed-in baseline or derive it on first call ──
        if base is None:
            # First call: this month's distribution becomes the baseline itself.
            base = month.rename(columns={'WTFINL': 'base'})
        month = base.merge(month, how='outer', on='OCC')

    # ── Compute aggregate totals for normalization ──
    # These are the total weighted employment counts across all occupations.
    # NaN entries from unmatched occupations contribute 0 via nansum semantics.
    base_all  = month['base'].sum()
    month_all = month['WTFINL'].sum()

    # Guard against division by zero if a period has no employment data.
    if base_all < 1e-6 or month_all < 1e-6:
        month[this_month] = float('nan')
        return month

    # ── Core dissimilarity formula (Duncan index) ──
    # For each occupation: |share_current − share_baseline| × 100 / 2
    month[this_month] = abs((month['WTFINL'] / month_all) - (month['base'] / base_all)) * 100 / 2

    return month


def _window_average(start_month: str, freq: int, all_months: pandas.DataFrame, oct_25: str) -> pandas.DataFrame:
    """
    Computes the mean WTFINL (final person-weight) per occupation across a
    contiguous `freq`-month window beginning at `start_month`.

    The flagged October 2025 month is excluded from the window.  If the
    resulting window has fewer unique months than expected (because some months
    are missing from all_months), an empty DataFrame is returned so the caller
    can propagate NaN rather than silently using an incomplete average.

    Parameters
    ----------
    start_month : str       – First month of the window in "YYYYMM" format.
    freq        : int       – Number of months in the window.
    all_months  : DataFrame – Pre-loaded panel (OCC, WTFINL, MONTH).
    oct_25      : str       – Stamp of the excluded month.

    Returns
    -------
    DataFrame with columns OCC and WTFINL (the within-window mean weight).
    Empty DataFrame if any required month is absent from all_months.
    """
    yr = int(start_month[:4])
    mn = int(start_month[4:6])
    # Compute the last month of the window: start + (freq-1) months.
    end_yr = yr + (mn + freq - 2) // 12
    end_mn = (mn + freq - 2) % 12 + 1
    # Build all stamps in the window, excluding the flagged October 2025 month.
    stamps = [s for s in _month_range(yr, mn, end_yr, end_mn) if s != oct_25]

    # Subset the pre-loaded panel to only rows belonging to this window.
    window = all_months[all_months["MONTH"].isin(stamps)]

    # Completeness check: if any window month is entirely absent, bail out.
    if window["MONTH"].nunique() < len(stamps):
        return pandas.DataFrame(columns=["OCC", "WTFINL"])

    # Average WTFINL across the window months for each occupation.
    avg = window.groupby("OCC")["WTFINL"].mean().reset_index()
    return avg


def load_all_months(ID: str, start_yr: int, start_mn: int, end_yr: int, end_mn: int,
                    occ_subset: str, ind_subset: str, oct_25: str) -> pandas.DataFrame:
    """
    Reads every monthly parquet file in the range [start_yr/start_mn, end_yr/end_mn]
    into a single long-format DataFrame, then synthesizes the flagged October 2025
    month as the mean of September and November 2025.

    Parameters
    ----------
    ID          : str  – Subdirectory identifying the data variant.
    start_yr/mn : int  – Inclusive start of the date range to load.
    end_yr/mn   : int  – Inclusive end of the date range to load.
    occ_subset  : str  – Optional occupation filter "first.last".
    ind_subset  : str  – Optional industry filter "first.last".
    oct_25      : str  – Stamp of the excluded raw month ("202510").

    Returns
    -------
    Long-format DataFrame with columns: OCC, WTFINL, MONTH.
    """
    frames = []

    # Read each month except the flagged one (its raw data is skipped entirely).
    for stamp in _month_range(start_yr, start_mn, end_yr, end_mn):
        if stamp != "202510":
            frames.append((stamp, _read_month(ID, stamp, occ_subset, ind_subset)))

    # If both neighboring months are present, synthesize October 2025 as their
    # per-occupation average — this imputes the missing month while preserving
    # the overall occupational structure.
    stamps_needed = [f[0] for f in frames]
    if "202509" in stamps_needed and "202511" in stamps_needed:
        sep = next(f[1] for f in frames if f[0] == "202509")
        nov = next(f[1] for f in frames if f[0] == "202511")
        oct = pandas.concat([sep, nov], ignore_index=True).groupby("OCC").agg({"WTFINL": "mean"}).reset_index()
        frames.append(("202510", oct))

    # Stack all monthly frames into a single long-format panel, tagging each row
    # with its "YYYYMM" stamp so downstream functions can filter by month.
    all_months = pandas.concat(
        [df.assign(MONTH=stamp) for stamp, df in frames],
        ignore_index=True
    )
    return all_months

def _read_month(ID: str, this_month: str, occ_subset: str, ind_subset: str) -> pandas.DataFrame:
    """
    Reads one monthly CPS parquet file and applies standard cleaning and
    optional subsetting filters.

    Cleaning steps applied:
      1. Drop records with AGE ≤ 15 (exclude minors from the labor market sample).
      2. Drop records with OCC == "0000" (non-employed / no occupation assigned).
      3. Optionally restrict to a contiguous range of industry codes (IND).
      4. Aggregate to occupation level by summing final person-weights (WTFINL).
      5. Optionally restrict to a contiguous range of occupation codes (OCC).

    Parameters
    ----------
    ID          : str  – Subdirectory identifying the data variant.
    this_month  : str  – "YYYYMM" stamp of the month to read.
    occ_subset  : str  – "first.last" occupation code range, or '' for all.
    ind_subset  : str  – "first.last" industry code range, or '' for all.

    Returns
    -------
    DataFrame with columns OCC and WTFINL (total weighted employment per occupation).
    """
    # Construct the full file path and read from parquet.
    full_month = pandas.read_parquet(
        os.path.join(
            "/nfs/roberts/project/pi_nrs36/shared/model_data/AI-Employment-Model/dissimilarity",
            ID, "cps_" + this_month + ".parquet"
        ),
        engine='pyarrow'
    )

    # ── Standard exclusions ──
    full_month = full_month[full_month["AGE"] > 15]      # working-age adults only
    full_month = full_month[full_month["OCC"] != "0000"] # drop non-occupational records
    count      = full_month["WTFINL"].sum()              # total weighted count (kept for reference)

    # ── Optional industry restriction ──
    # Keep only rows whose IND code falls within [first, last].
    if ind_subset:
        masks = []
        for rng in ind_subset.split('|'):
            first, last = int(rng.split('.')[0]), int(rng.split('.')[1])
            ind_numeric = pandas.to_numeric(full_month['IND'])
            masks.append((ind_numeric >= first) & (ind_numeric <= last))
        combined = masks[0]
        for m in masks[1:]:
            combined = combined | m
        full_month = full_month[combined].reset_index()

    # ── Aggregate to occupation level ──
    # Sum WTFINL within each occupation code to get total weighted employment.
    out = full_month.groupby("OCC").agg({"WTFINL": 'sum'}).reset_index()

    # ── Optional occupation restriction ──
    # Applied after aggregation to avoid filtering out individual workers
    # whose occupation code is on the boundary.
    if occ_subset:
        occ_parts = occ_subset.split('.')
        out = out[
            (pandas.to_numeric(out['OCC']) >= int(occ_parts[0])) &
            (pandas.to_numeric(out['OCC']) <= int(occ_parts[1]))
        ].reset_index()

    return out


def get_month_dd(ID: str, this_month: str, occ_subset: str, ind_subset: str):
    """
    Returns the occupation-level employment distribution for a single month,
    with special handling for the flagged October 2025 month.

    Unlike _read_month (an internal helper used during bulk loading), this
    function is called directly by external code that needs the distribution
    for a specific point in time — for example to inspect or plot a snapshot.

    For October 2025, the function recursively fetches September and November
    2025 and returns their per-occupation average, mirroring the imputation
    used in load_all_months.

    Parameters
    ----------
    ID          : str  – Subdirectory identifying the data variant.
    this_month  : str  – Target "YYYYMM" stamp.
    occ_subset  : str  – "first.last" occupation code range, or '' for all.
    ind_subset  : str  – "first.last" industry code range, or '' for all.

    Returns
    -------
    DataFrame with columns OCC and WTFINL.
    """
    if this_month == "202510":
        # Impute October 2025 as the mean of its two neighbors.
        sep = get_month_dd(ID, "202509", occ_subset, ind_subset)
        nov = get_month_dd(ID, "202511", occ_subset, ind_subset)
        out = (
            pandas.concat([sep, nov], ignore_index=True)
            .groupby("OCC").agg({"WTFINL": 'mean'})
            .reset_index()
        )
    else:
        # ── Standard path: read and clean the requested month ──
        full_month = pandas.read_parquet(
            os.path.join(
                "/nfs/roberts/project/pi_nrs36/shared/model_data/AI-Employment-Model/dissimilarity",
                ID, "cps_" + this_month + ".parquet"
            ),
            engine='pyarrow'
        )

        full_month = full_month[full_month["AGE"] > 15]      # working-age adults only
        full_month = full_month[full_month["OCC"] != "0000"] # drop non-occupational records
        count      = full_month["WTFINL"].sum()

        # ── Optional industry restriction ──
        if ind_subset:
            ind_subset = ind_subset.split('.')
            first      = int(ind_subset[0])
            last       = int(ind_subset[1])
            full_month = full_month[
                (pandas.to_numeric(full_month['IND']) >= first) &
                (pandas.to_numeric(full_month['IND']) <= last)
            ].reset_index()

        # ── Aggregate to occupation level ──
        out = full_month.groupby("OCC").agg({"WTFINL": 'sum'}).reset_index()

        # ── Optional occupation restriction ──
        if occ_subset:
            occ_subset = occ_subset.split('.')
            out = out[
                (pandas.to_numeric(out['OCC']) >= int(occ_subset[0])) &
                (pandas.to_numeric(out['OCC']) <= int(occ_subset[1]))
            ].reset_index()

    return out