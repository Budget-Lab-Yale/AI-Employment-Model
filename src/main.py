import pandas
import numpy
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import os

from calc_other  import build_other_output
from calc_dd     import build_mismatch
from calc_edu_dd import build_mis_edu
from calc_ind_dd import build_mis_ind

stamp = datetime.datetime.now().strftime('%Y%m%d%H%M')
#stamp = "202604141017"
out_path = os.path.join("/nfs/roberts/project/pi_nrs36/shared/model_data/AI-Employment-Model")

# FLAG: Create both base and rolling subdirectories
for subdir in ["indexed", "rolling"]:
    if not os.path.exists(os.path.join(out_path, "output", stamp, subdir)):
        os.makedirs(os.path.join(out_path, "output", stamp, subdir))

# Loop over whether to calculate dissimilarity against start_period or a rolling comparison
for rolling in [False, True]:
    subdir = "rolling" if rolling else "indexed"

    # When rolling=True, pivot on the calendar date column instead of the
    # integer months_gone offset. The `period` column (YYYY-MM-01) is
    # produced by build_mismatch when rolling=True; indexed mode has no
    # such column so it continues to use months_gone.
    index_col = "period" if rolling else "months_gone"

    print(f"Dissimilarity by Period (rolling={rolling})")
    
    start_period = "2022.11"
    end_period   = "2026.4"
    
    # Build dissimilarity data by historical time periods
    params = pandas.DataFrame({
        'ID':    ["Computers", "Internet", "Control", "AI"],
        'start': ["1984.1","1996.1", "2016.1", start_period],
        'end':   ["1989.12", "2002.12", "2019.12", end_period]
        }
    )
    deltas = None
    for i in range(0, len(params)):
        # FLAG JOSH you changed pre-trend
        result = build_mismatch(params.loc[i, 'ID'], params.loc[i, 'start'], params.loc[i, 'end'], freq=12, pre_trend=0, rolling=rolling).assign(
            label = params.loc[i, 'ID']
        )
        deltas = result if deltas is None else pandas.concat([deltas, result])

    deltas.pivot(
        index   = index_col,
        columns = 'label',
        values  = 'dissimilarity'
    ).reset_index().to_csv(os.path.join(out_path, "output", stamp, subdir, "dissimilarity.csv"), index = False)
        
    print(f"AI Dissimilarity by start period (rolling={rolling})")
    # Build dissimilarity data by recent time periods
    if rolling:
        params = pandas.DataFrame({
            'ID':    ["AI"],
            'start': ["2022.01"],
            'end':   [end_period],
            'label': ["Jan22"]
            }
        )
    else:
        params = pandas.DataFrame({
            'ID':    ["AI", "AI", "AI", "AI"],
            'start': ["2022.01", "2022.11", "2022.07", "2021.01"],
            'end':   [end_period, end_period, end_period, end_period],
            'label': ["Jan22", "AI (Nov22)", "Jul22", "Jan21"]
            }
        )

    deltas = None
    for i in range(0, len(params)):
        result = build_mismatch(params.loc[i, 'ID'], params.loc[i, 'start'], params.loc[i, 'end'], 12, rolling=rolling).assign(
            label = params.loc[i, 'label']
        )
        deltas = result if deltas is None else pandas.concat([deltas, result])

    deltas.pivot(
        index   = index_col,
        columns = 'label',
        values  = 'dissimilarity'
    ).reset_index().to_csv(os.path.join(out_path, "output", stamp, subdir, "recent.csv"), index = False)

    print(f"Dissimilarity by Industry (rolling={rolling})")
    # By industry analyses
    recent = pandas.DataFrame({
    'name': [
        'recent',
        'natural_resources_and_mining',
        'construction',
        'manufacturing',
        'trade_transportation_and_utilities',
        'information',
        'financial_activities',
        'professional_and_business_services',
        'education_and_health_services',
        'leisure_and_hospitality',
        'other_services'
    ],
    'industry': [
        '',
        '0100.0560',
        '0770.1060',
        '1070.4060',
        '4070.6390|0570.0760',
        '6470.6860',
        '6870.7260',
        '7270.7790',
        '7860.8470',
        '8560.8690',
        '8770.9290'
    ]
    })

    for j in range(0, len(recent)):
        deltas = None
        for i in range(0, len(params)):
            result = build_mismatch(params.loc[i, 'ID'], params.loc[i, 'start'], params.loc[i, 'end'], 12, '', recent.loc[j,'industry'], rolling=rolling).assign(
                label = params.loc[i, 'label']
            )
            deltas = result if deltas is None else pandas.concat([deltas, result])
        name = recent.loc[j, 'name'] + ".csv"
        deltas.pivot(
            index   = index_col,
            columns = 'label',
            values  = 'dissimilarity'
        ).reset_index().to_csv(os.path.join(out_path, "output", stamp, subdir, name), index = False)

    params2 = pandas.DataFrame({
    'label': [
        'all',
        'natural_resources_and_mining',
        'construction',
        'manufacturing',
        'trade_transportation_and_utilities',
        'information',
        'financial_activities',
        'professional_and_business_services',
        'education_and_health_services',
        'leisure_and_hospitality',
        'other_services'
    ],
    'industry': [
        '',
        '0100.0560',
        '0770.1060',
        '1070.4060',
        '4070.6390|0570.0760',
        '6470.6860',
        '6870.7260',
        '7270.7790',
        '7860.8470',
        '8560.8690',
        '8770.9290'
    ]
    })
    deltas = None
    for i in range(0, len(params2)):
            result = build_mismatch("AI", start_period, end_period, 12, '', params2.loc[i,'industry'], rolling=rolling).assign(
                label = params2.loc[i, 'label']
            )
            deltas = result if deltas is None else pandas.concat([deltas, result])
    deltas.pivot(
        index   = index_col,
        columns = 'label',
        values  = 'dissimilarity'
    ).reset_index().to_csv(os.path.join(out_path, "output", stamp, subdir, 'industry.csv'), index = False)
    
    # all industries back to 2004
    if rolling:
        start_period = "2005.1"
    else:
        start_period = "2003.12"
    end_period   = "2026.4"
    print(f"Dissimilarity Long (rolling={rolling})")
    build_mis_ind("Industry", start_period, end_period, 12, rolling=rolling).to_csv(os.path.join(out_path, "output", stamp, subdir, 'every.csv'), index = False)

print(f"Dissimilarity Graduates")
# By recency of graduation
recent_grads = build_mis_edu("dissimilarity/Industry", "2022.01", end_period, 3).to_csv(os.path.join(out_path, "output", stamp, "recent_grads_dissimilarity_22_26.csv"), index = False)
recent_grads = build_mis_edu("dissimilarity/Industry", "2015.01", end_period, 3).to_csv(os.path.join(out_path, "output", stamp, "recent_grads_dissimilarity_15_26.csv"), index = False)
recent_grads = build_mis_edu("input/feb26", "2021.01", end_period, 3, time = True).to_csv(os.path.join(out_path, "output", stamp, "recent_grads_dissimilarity_lone.csv"), index = False)

# Builds exposure/usage metrics over time
start_period = '2022.9'

end_period   = "2026.4"

write_path = os.path.join(out_path, "output", stamp, 'gpt4_rubric1_beta')

print("Beta")
for m in ["feb26", "feb26_fm"]:
  if not os.path.exists(write_path):
    os.makedirs(write_path)
  print(f"Building output for: {m}")
  build_other_output(m, start_period, end_period, write_path, metric = 'gpt4_rubric1_beta', freq = 3)