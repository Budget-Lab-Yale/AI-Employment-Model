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


# Industry labels used throughout — defined once here so they can be referenced
# consistently in build_mis_ind, calc_mis_ind_month, and the output DataFrame.
INDUSTRIES = [
    'Natural Resources and Mining',
    'Construction',
    'Manufacturing',
    'Trade, Transportation, and Utilities',
    'Information',
    'Financial Activities',
    'Professional and Business Services',
    'Education and Health Services',
    'Leisure and Hospitality',
    'Other Services',
]


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY: Month-range generator
# ─────────────────────────────────────────────────────────────────────────────

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
        stamps.append(str(yr) + str(mn).zfill(2))   # zero-pad month to two digits
        mn += 1
        if mn > 12:          # roll over December → January of next year
            mn = 1
            yr += 1
    return stamps


# ─────────────────────────────────────────────────────────────────────────────
# TOP-LEVEL: Build the full industry-level dissimilarity time series
# ─────────────────────────────────────────────────────────────────────────────

def build_mis_ind(ID: str, start: str, end: str, freq = 1, rolling = False, rolling_lag = 12):
    """
    Constructs a time series of occupational dissimilarity (mismatch) scores
    broken out by broad industry sector.

    For each target month and each industry, the index sums — across all
    occupation codes within that industry — half the absolute difference between
    the share of employment in that occupation in the current period and in the
    baseline period.  A value of 0 means no change; a value of 100 means the
    distributions are completely non-overlapping.

    The moving average is a trailing average: a target month's window covers
    that month and the (freq-1) months immediately preceding it.

    Parameters
    ----------
    ID          : str  – Subdirectory name that identifies which model/data variant to use.
    start       : str  – First target month in "YYYY.MM" format (e.g. "2019.01").
    end         : str  – Last target month in "YYYY.MM" format, inclusive.
    freq        : int  – Number of months to average together to form a single
                         observation (1 = monthly, 3 = quarterly trailing avg, etc.).
    rolling     : bool – If True, use a rolling baseline: compare each month to the
                         corresponding month exactly `rolling_lag` months earlier,
                         rather than to a single fixed baseline.
    rolling_lag : int  – How many months back the rolling baseline lags (default 12,
                         i.e. year-over-year comparison).

    Returns
    -------
    pandas.DataFrame with columns:
        months_gone  – integer row index (0-based position in the output series)
        <industry>   – one column per industry in INDUSTRIES, each holding the
                       dissimilarity index value (0–100) for that period
    """

    # ── Parse "YYYY.MM" strings into separate integer year/month components ──
    start = start.split('.')
    end   = end.split('.')

    yr_dex = int(start[0])
    mn_dex = int(start[1])
    end_yr = int(end[0])
    end_mn = int(end[1])

    # Normalize in case a caller passes a month value outside [1,12].
    yr_dex += (mn_dex - 1) // 12
    mn_dex  = (mn_dex - 1) % 12 + 1
    end_yr += (end_mn - 1) // 12
    end_mn  = (end_mn - 1) % 12 + 1

    # October 2025 is flagged as unavailable and is excluded from raw reads;
    # any window that would have included it is computed over the remaining months.
    oct_25 = "202510"

    # ── Determine the earliest month that must be read from disk ──
    # We must load data starting (freq-1) months before yr_dex/mn_dex so that
    # the very first target-month trailing window has all its constituent months.
    # If rolling, we also need rolling_lag extra months so the lag window exists.
    read_yr = yr_dex
    read_mn = mn_dex - (freq - 1)
    if rolling:
        read_mn -= rolling_lag   # need an extra rolling_lag months of history

    # Normalize after the backward shift (result could be month 0 or negative).
    read_yr += (read_mn - 1) // 12
    read_mn  = (read_mn - 1) % 12 + 1

    # ── Bulk-load every required month from parquet into a single DataFrame ──
    # This avoids re-reading files repeatedly as we iterate over target months.
    all_months = load_all_months_ind(ID, read_yr, read_mn, end_yr, end_mn, oct_25)

    # ── Iterate over every target month and compute per-industry dissimilarity scores ──
    months = None
    for stamp in _month_range(yr_dex, mn_dex, end_yr, end_mn):
        if months is None:
            # First iteration: calc_mis_ind_month establishes the baseline distribution
            # (stored in the "base" column for fixed-baseline mode).
            base = calc_mis_ind_month(stamp, freq, oct_25, all_months,
                                      rolling=rolling, rolling_lag=rolling_lag)
            # Aggregate per-occupation contributions to a single score per industry.
            months = base.groupby('Industry').agg({stamp: 'sum'}).reset_index()
        else:
            # Subsequent iterations: pass the existing baseline (OCC, Industry, base)
            # so all months are compared against the same initial distribution.
            this_month = calc_mis_ind_month(stamp, freq, oct_25, all_months,
                                            base=None if rolling else base[["Industry", "OCC", "base"]],
                                            rolling=rolling, rolling_lag=rolling_lag)
            # Aggregate and merge the new industry scores onto the growing wide table.
            this_month = this_month.groupby('Industry').agg({stamp: 'sum'}).reset_index()
            months = months.merge(
                this_month,
                how = 'outer',   # keep all industries even if one month has no data
                on  = 'Industry'
            )

    # ── Reshape from wide (industry rows × month columns) to long (month rows × industry columns) ──
    # Transpose so each row is a month and each column is an industry's score.
    deltas = months.set_index('Industry').T.reset_index(names='MONTH').rename_axis(columns=None)

    # Explicitly NaN out all industry scores for the flagged October 2025 stamp.
    # (The per-occupation values were already set to NaN in calc_mis_ind_month, but
    # the groupby sum may produce 0 instead of NaN for fully-NaN groups.)
    deltas.loc[deltas["MONTH"] == oct_25, INDUSTRIES] = numpy.nan

    # ── Attach a positional index and drop the raw MONTH stamp ──
    # months_gone is a simple 0-based integer offset along the output series.
    deltas["months_gone"] = deltas.index

    if rolling:
        deltas["period"] = pandas.to_datetime(deltas["MONTH"], format="%Y%m").dt.strftime("%Y-%m-02")
        deltas = deltas.loc[:, ['period'] + INDUSTRIES]
    else:
        deltas = deltas.loc[:, ['months_gone'] + INDUSTRIES]

    return deltas

# ─────────────────────────────────────────────────────────────────────────────
# CORE CALCULATION: Per-occupation dissimilarity contributions for one month,
#                   stratified by industry
# ─────────────────────────────────────────────────────────────────────────────

def calc_mis_ind_month(this_month: str, freq: int, oct_25: str, all_months: pandas.DataFrame,
                       base = None, rolling = False, rolling_lag = 12):
    """
    Computes the per-occupation, per-industry dissimilarity contributions for a
    single target month.

    Within each industry the dissimilarity contribution for one occupation equals:
        | share_current(occ) − share_baseline(occ) | × 100 / 2

    where shares are computed *within* the industry (i.e. each occupation's
    WTFINL divided by the industry total, not the economy-wide total).
    Summing across occupations within an industry gives that industry's index.

    The moving average window is a trailing window: it covers this_month and the
    (freq-1) months immediately before it.  If October 2025 falls inside a window,
    it is skipped and the average is computed over the remaining months (e.g. 11
    months instead of 12), rather than propagating NaN for the whole window.

    Parameters
    ----------
    this_month  : str             – Target "YYYYMM" stamp.
    freq        : int             – Window size in months for the trailing average.
    oct_25      : str             – The flagged month stamp ("202510") to skip.
    all_months  : DataFrame       – Pre-loaded panel with columns OCC, Industry, WTFINL, MONTH.
    base        : DataFrame|None  – In fixed-baseline mode, the caller passes the
                                    baseline OCC/Industry/WTFINL distribution computed at
                                    the first iteration so all months share the same reference.
                                    None on the first call (baseline is derived here)
                                    or when rolling=True.
    rolling     : bool            – If True, compare to the distribution rolling_lag
                                    months earlier instead of a fixed baseline.
    rolling_lag : int             – Lag in months for the rolling baseline.

    Returns
    -------
    DataFrame with columns: OCC, Industry, base, <this_month>
        where <this_month> holds the per-occupation, per-industry dissimilarity contribution.
    """

    # ── Special handling for October 2025 itself as a target month ──
    # When this_month is the flagged stamp, there is no current-period data to
    # compare against, so every occupation/industry contribution is NaN.
    if this_month == oct_25:
        month = all_months[all_months["MONTH"] == this_month][["OCC", "Industry", "WTFINL"]]

        if base is None:
            # First iteration: build a skeleton baseline and current frame, both NaN.
            base = month[['OCC', 'Industry']].copy()
            base['base'] = float('nan')
            month = base.copy()
            month[this_month] = float('nan')
        else:
            # Subsequent iteration: keep the existing baseline, mark this month NaN.
            month = base[['OCC', 'Industry', 'base']].copy()
            month[this_month] = float('nan')

        return month[['OCC', 'Industry', 'base', this_month]]

    # ── Determine the start of the trailing freq-month window ──
    # If freq=12 and this_month="202511", the window covers 202412–202511,
    # but 202510 is skipped so the average is over 11 months instead of 12.
    cur_yr = int(this_month[:4])
    cur_mn = int(this_month[4:6]) - (freq - 1)   # shift back (freq-1) months

    # Normalize: shifting back may produce month ≤ 0, crossing a year boundary.
    cur_yr += (cur_mn - 1) // 12
    cur_mn  = (cur_mn - 1) % 12 + 1
    cur_start = str(cur_yr) + str(cur_mn).zfill(2)

    # ── Compute the trailing freq-month average employment distribution ──
    month = _window_average_ind(cur_start, freq, all_months, oct_25)

    if rolling:
        # ── Rolling baseline: find the trailing window rolling_lag months earlier ──
        lag_yr = int(this_month[:4])
        lag_mn = int(this_month[4:6]) - rolling_lag - (freq - 1)

        lag_yr += (lag_mn - 1) // 12
        lag_mn  = (lag_mn - 1) % 12 + 1
        lag_month_stamp = str(lag_yr) + str(lag_mn).zfill(2)

        lag_month = _window_average_ind(lag_month_stamp, freq, all_months, oct_25)
        # Rename lag window's WTFINL to "base" before merging with current window.
        lag_month = lag_month.rename(columns={'WTFINL': 'base'})
        month = lag_month.merge(month, how='outer', on=['OCC', 'Industry'])

    else:
        # ── Fixed baseline: use the passed-in baseline or derive it on the first call ──
        if base is None:
            # First call: this month's distribution becomes the fixed baseline.
            base = month.rename(columns={'WTFINL': 'base'})
        month = base.merge(month, how='outer', on=['OCC', 'Industry'])

    # ── Compute industry-level totals for normalization ──
    # Each occupation's share is computed within its industry, not economy-wide,
    # so the denominators are industry-specific sums of WTFINL.
    month['base_all']  = month['base'].groupby(month['Industry']).transform('sum')
    month['month_all'] = month['WTFINL'].groupby(month['Industry']).transform('sum')

    # ── Core dissimilarity formula (Duncan index), applied within each industry ──
    # For each occupation within an industry:
    #     |share_current − share_baseline| × 100 / 2
    # Division by 2 keeps the index in the [0, 100] range.
    month[this_month] = abs(
        (month['WTFINL'] / month['month_all']) - (month['base'] / month['base_all'])
    ) * 100 / 2

    return month[['OCC', 'Industry', 'base', this_month]]


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: Trailing average employment distribution over a multi-month window
# ─────────────────────────────────────────────────────────────────────────────

def _window_average_ind(start_month: str, freq: int, all_months: pandas.DataFrame, oct_25: str) -> pandas.DataFrame:
    """
    Computes the mean WTFINL (final person-weight) per occupation-industry pair
    across a contiguous `freq`-month trailing window beginning at `start_month`.

    If October 2025 falls inside the window it is skipped rather than causing
    the whole window to return NaN — the average is computed over however many
    valid months remain (e.g. 11 out of 12).

    Parameters
    ----------
    start_month : str       – First (earliest) month of the window in "YYYYMM" format.
                              The caller is responsible for shifting this back (freq-1)
                              months from the target month so the window trails correctly.
    freq        : int       – Nominal number of months in the window.
    all_months  : DataFrame – Pre-loaded panel (OCC, Industry, WTFINL, MONTH).
    oct_25      : str       – Stamp of the month to skip within the window.

    Returns
    -------
    DataFrame with columns OCC, Industry, and WTFINL (the within-window mean weight).
    """
    yr = int(start_month[:4])
    mn = int(start_month[4:6])
    # Compute the last month of the window: start + (freq-1) months forward.
    end_yr = yr + (mn + freq - 2) // 12
    end_mn = (mn + freq - 2) % 12 + 1
    # Build all stamps in the window, skipping the flagged month.
    # Unlike the occupational version, a missing oct_25 does NOT cause an empty
    # return — the average simply uses fewer months.
    stamps = [s for s in _month_range(yr, mn, end_yr, end_mn) if s != oct_25]

    window = all_months[all_months["MONTH"].isin(stamps)]
    avg = window.groupby(["OCC", "Industry"])["WTFINL"].mean().reset_index()
    return avg


# ─────────────────────────────────────────────────────────────────────────────
# I/O: Bulk-load all required months from parquet
# ─────────────────────────────────────────────────────────────────────────────

def load_all_months_ind(ID: str, start_yr: int, start_mn: int, end_yr: int, end_mn: int,
                        oct_25: str) -> pandas.DataFrame:
    """
    Reads every monthly parquet file in the range [start_yr/start_mn, end_yr/end_mn]
    into a single long-format DataFrame with industry labels attached, then
    synthesizes the flagged month by copying September 2025 with WTFINL set to NaN.

    The NaN imputation (rather than the Sep/Nov average used in the occupational
    version) preserves the original get_month_ind_dd behavior for October 2025.
    Windows that span October 2025 are handled in _window_average_ind by skipping
    the flagged stamp and averaging over the remaining months.

    Parameters
    ----------
    ID          : str  – Subdirectory identifying the data variant.
    start_yr/mn : int  – Inclusive start of the date range to load.
    end_yr/mn   : int  – Inclusive end of the date range to load.
    oct_25      : str  – Stamp of the flagged month to synthesize.

    Returns
    -------
    Long-format DataFrame with columns: OCC, Industry, WTFINL, MONTH.
    """
    frames = []

    # Read each month except the flagged one (its raw data is skipped entirely).
    for stamp in _month_range(start_yr, start_mn, end_yr, end_mn):
        if stamp != "202510":
            frames.append((stamp, _read_month_ind(ID, stamp)))

    # Synthesize the flagged month: copy September 2025's structure but set all
    # weights to NaN.  _window_average_ind will skip this stamp entirely when
    # building window averages, so the NaN values are never actually used in
    # calculations — the frame is included only to keep the MONTH column consistent.
    stamps_needed = [f[0] for f in frames]
    if "202509" in stamps_needed:
        sep = next(f[1] for f in frames if f[0] == "202509").copy()
        sep["WTFINL"] = float("nan")
        frames.append(("202510", sep))

    # Stack all monthly frames into a single long-format panel, tagging each row
    # with its "YYYYMM" stamp so downstream functions can filter by month.
    all_months = pandas.concat(
        [df.assign(MONTH=stamp) for stamp, df in frames],
        ignore_index=True
    )
    return all_months


# ─────────────────────────────────────────────────────────────────────────────
# I/O: Read and clean a single month's parquet file, with industry mapping
# ─────────────────────────────────────────────────────────────────────────────

def _read_month_ind(ID: str, this_month: str) -> pandas.DataFrame:
    """
    Reads one monthly CPS parquet file, applies standard cleaning, maps each
    record to one of the ten broad BLS industry sectors, and aggregates to the
    occupation-industry level.

    Cleaning steps applied:
      1. Drop records with AGE ≤ 15 (exclude minors from the labor market sample).
      2. Drop records with OCC == "0000" (non-employed / no occupation assigned).
      3. Map numeric IND codes to one of ten named industry sectors via numpy.select;
         records that fall outside all defined ranges are dropped.
      4. Aggregate to (OCC, Industry) level by summing final person-weights (WTFINL).

    Industry code ranges (CPS IND codes):
      Natural Resources and Mining       :  100– 560
      Construction                       :  770–1060
      Manufacturing                      : 1070–4060
      Trade, Transportation & Utilities  : 4070–6390 and 570–760
      Information                        : 6470–6860
      Financial Activities               : 6870–7260
      Professional and Business Services : 7270–7790
      Education and Health Services      : 7860–8470
      Leisure and Hospitality            : 8560–8690
      Other Services                     : 8770–9290

    Parameters
    ----------
    ID          : str  – Subdirectory identifying the data variant.
    this_month  : str  – "YYYYMM" stamp of the month to read.

    Returns
    -------
    DataFrame with columns OCC, Industry, and WTFINL
    (total weighted employment per occupation-industry pair).
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
    full_month = full_month[full_month["AGE"] > 15]       # working-age adults only
    full_month = full_month[full_month["OCC"] != "0000"]  # drop non-occupational records

    # ── Industry mapping ──
    # Convert IND to numeric then assign each record to one of the ten sectors.
    # Records outside all defined ranges receive the default value '' and are dropped.
    full_month["IND"] = pandas.to_numeric(full_month['IND'])
    full_month['Industry'] = numpy.select(
        [
            full_month['IND'].ge(100)  & full_month['IND'].le(560),
            full_month['IND'].ge(770)  & full_month['IND'].le(1060),
            full_month['IND'].ge(1070) & full_month['IND'].le(4060),
            (full_month['IND'].ge(4070) & full_month['IND'].le(6390)) | (full_month['IND'].ge(570) & full_month['IND'].le(760)),
            full_month['IND'].ge(6470) & full_month['IND'].le(6860),
            full_month['IND'].ge(6870) & full_month['IND'].le(7260),
            full_month['IND'].ge(7270) & full_month['IND'].le(7790),
            full_month['IND'].ge(7860) & full_month['IND'].le(8470),
            full_month['IND'].ge(8560) & full_month['IND'].le(8690),
            full_month['IND'].ge(8770) & full_month['IND'].le(9290),
        ],
        INDUSTRIES,
        default=''
    )
    # Drop records that didn't match any industry range.
    full_month = full_month[full_month['Industry'] != '']

    # ── Aggregate to occupation-industry level ──
    # Sum WTFINL within each (OCC, Industry) pair to get total weighted employment.
    out = full_month.groupby(["OCC", "Industry"]).agg({"WTFINL": 'sum'}).reset_index()
    return out


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC: Retrieve the employment distribution for a single month (point-in-time)
# Retained unchanged for external callers; no longer used internally.
# ─────────────────────────────────────────────────────────────────────────────

def get_month_ind_dd(ID: str, this_month: str):
    """
    Returns the occupation-by-industry employment distribution for a single month,
    with special handling for October 2025.

    This function is no longer called anywhere inside this module — all internal
    bulk loading goes through load_all_months_ind / _read_month_ind — but it is
    retained here for any external callers that depend on it.

    For October 2025, the function copies September 2025's distribution and sets
    all WTFINL values to NaN, signaling that no reliable data exists for that month.

    Parameters
    ----------
    ID          : str  – Subdirectory identifying the data variant.
    this_month  : str  – Target "YYYYMM" stamp.

    Returns
    -------
    DataFrame with columns OCC, Industry, and WTFINL.
    """
    if this_month == "202510":
        # Copy September's occupational structure but mark all weights as missing.
        out = get_month_ind_dd(ID, "202509")
        out["WTFINL"] = float("nan")

    else:
        # ── Standard path: read, clean, map industries, and aggregate ──
        full_month = pandas.read_parquet(
            os.path.join(
                "/nfs/roberts/project/pi_nrs36/shared/model_data/AI-Employment-Model/dissimilarity",
                ID, "cps_" + this_month + ".parquet"
            ),
            engine='pyarrow'
        )

        full_month = full_month[full_month["AGE"] > 15]       # working-age adults only
        full_month = full_month[full_month["OCC"] != "0000"]  # drop non-occupational records

        # ── Industry mapping (same logic as _read_month_ind) ──
        full_month["IND"] = pandas.to_numeric(full_month['IND'])
        full_month['Industry'] = numpy.select(
            [
                full_month['IND'].ge(100)  & full_month['IND'].le(560),
                full_month['IND'].ge(770)  & full_month['IND'].le(1060),
                full_month['IND'].ge(1070) & full_month['IND'].le(4060),
                (full_month['IND'].ge(4070) & full_month['IND'].le(6390)) | (full_month['IND'].ge(570) & full_month['IND'].le(760)),
                full_month['IND'].ge(6470) & full_month['IND'].le(6860),
                full_month['IND'].ge(6870) & full_month['IND'].le(7260),
                full_month['IND'].ge(7270) & full_month['IND'].le(7790),
                full_month['IND'].ge(7860) & full_month['IND'].le(8470),
                full_month['IND'].ge(8560) & full_month['IND'].le(8690),
                full_month['IND'].ge(8770) & full_month['IND'].le(9290),
            ],
            INDUSTRIES,
            default=''
        )
        full_month = full_month[full_month['Industry'] != '']

        # ── Aggregate to occupation-industry level ──
        out = full_month.groupby(["OCC", "Industry"]).agg({"WTFINL": 'sum'}).reset_index()

    return out