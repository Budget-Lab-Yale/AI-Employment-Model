import pandas
import numpy
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

from parsing     import build_linked_dataset
from calc_other  import build_other_output
from calc_dd     import build_mismatch
from calc_edu_dd import build_mis_edu
from calc_ind_dd import build_mis_ind

stamp = datetime.datetime.now().strftime('%Y%m%d%H%M')
out_path = os.path.join("/gpfs/gibbs/project/sarin/shared/model_data/AI-Employment-Model", stamp)
#out_path = os.path.join("/gpfs/gibbs/project/sarin/jmk263/Repositories/AI-Employment-Model/output", stamp)


if not os.path.exists(out_path):
    os.makedirs(out_path)

# ID = '2025092202'
# start_period = "2003.01"
# end_period   = "2025.08"
# build_linked_dataset(ID, start_period, end_period, "OCC2010")

# ID = '2025092312'
# start_period = "1980.1"
# end_period = "1989.12"
# build_linked_dataset(ID, start_period, end_period, "OCC1990")

# ID = '2025092313'
# start_period = "1994.1"
# end_period = "2002.12"
# build_linked_dataset(ID, start_period, end_period)

# ID = '2025092202_14-19'
# start_period = "2014.1"
# end_period   = "2019.12"
# build_linked_dataset(ID, start_period, end_period)

# ID = '2025091811'
# start_period = "2021.1"
# end_period   = "2025.8"
# Different Usage data is selected inside of the function below
# build_linked_dataset(ID, start_period, end_period)

ID = '2025092202_20-25'
# start_period = "2020.01"
# end_period   = "2025.8"
# build_linked_dataset(ID, start_period, end_period)


start_period = "2021.12"
end_period   = "2025.8"

# Build dissimilarity data by historical time periods
params = pandas.DataFrame({
    'ID':    ['2025092312', '2025092313', '2025092202_14-19', '2025092202_20-25'],
    'start': ["1983.2","1995.2", "2015.2", start_period],
    'end':   ["1989.12", "2002.12", "2019.12", end_period],
    'label': ["Computers", "Internet", "Control", "AI"]
    }
)

deltas = pandas.DataFrame(columns=['MONTH', "dissimilarity"])
for i in range(0, len(params)):
    deltas = pandas.concat([deltas, build_mismatch(params.loc[i, 'ID'], params.loc[i, 'start'], params.loc[i, 'end'], freq=12, pre_trend=12).assign(
        label = params.loc[i, 'label']
    )])

deltas.pivot(
    index   = 'months_gone',
    columns = 'label',
    values  = 'dissimilarity'
).reset_index().to_csv(os.path.join(out_path, "dissimilarity.csv"), index = False)


# Build dissimilarity data by recent time periods
params = pandas.DataFrame({
    'ID':    [ID, ID, ID, ID],
    'start': ["2020.02","2021.02", "2021.12", "2021.08"],
    'end':   [end_period, end_period, end_period, end_period],
    'label': ["Jan21", "Jan22", "AI (Nov22)", "Jul22"]
    }
)


# By industry analyses
recent = pandas.DataFrame({
    'name':['recent', 'information', 'financial_activities', 'professional_and_business_services'],
    'industry': ['','6470.6860','6870.7260','7270.7790']
})

for j in range(0, len(recent)):
    print(recent.loc[j,'industry'])
    deltas = pandas.DataFrame(columns=['MONTH', "dissimilarity"])
    for i in range(0, len(params)):
        deltas = pandas.concat([deltas, build_mismatch(params.loc[i, 'ID'], params.loc[i, 'start'], params.loc[i, 'end'], 12, '', recent.loc[j,'industry']).assign(
            label = params.loc[i, 'label']
        )])
    name = recent.loc[j, 'name'] + ".csv"
    deltas.pivot(
        index   = 'months_gone',
        columns = 'label',
        values  = 'dissimilarity'
    ).reset_index().to_csv(os.path.join(out_path, name), index = False)

params2 = pandas.DataFrame({
    'label':['all', 'information', 'financial_activities', 'professional_and_business_services'],
    'industry': ['','6470.6860','6870.7260','7270.7790']
})
deltas = pandas.DataFrame(columns=['MONTH', "dissimilarity"])
for i in range(0, len(params2)):
        deltas = pandas.concat([deltas, build_mismatch(ID, start_period, end_period, 12, '', params2.loc[i,'industry']).assign(
            label = params2.loc[i, 'label']
        )])
deltas.pivot(
    index   = 'months_gone',
    columns = 'label',
    values  = 'dissimilarity'
).reset_index().to_csv(os.path.join(out_path, 'industry.csv'), index = False)


# By recency of graduation
# The following start date of November 2020 allows us to calculate the 3 month trailing average. A programmatic fix for this is forthcoming.
recent_grads = build_mis_edu(ID, "2020.11", end_period, 3).to_csv(os.path.join(out_path, "recent_grads_dissimilarity.csv"), index = False)
recent_grads = build_mis_edu("2025092202", "2014.11", end_period, 3).to_csv(os.path.join(out_path, "recent_grads_dissimilarity2.csv"), index = False)


# Builds exposure/usage metrics over time
# print("Alpha")
# build_other_output(ID, start_period, end_period, out_path, metric = 'gpt4_rubric1_alpha', freq = 1)
start_period = '2022.9'
print("Beta")
build_other_output('2025091811_august', start_period, end_period, out_path, metric = 'gpt4_rubric1_beta', freq = 3)
build_other_output('2025091811_march', start_period, end_period, out_path, metric = 'gpt4_rubric1_beta', freq = 3)
build_other_output('2025091811_august_claude', start_period, end_period, out_path, metric = 'gpt4_rubric1_beta', freq = 3)

# build_other_output(ID, start_period, end_period, out_path, metric = 'gpt4_automation', freq = 3)
# print("Gamma")
# build_other_output(ID, start_period, end_period, out_path, metric = 'gpt4_rubric1_gamma', freq = 3)


# all industries back to 2004
ID = '2025092202'
start_period = "2003.2"
end_period   = "2025.8"

build_mis_ind(ID, start_period, end_period, 12).to_csv(os.path.join(out_path, 'every.csv'), index = False)


ID = '2025090512'

start_period = "2024.1"
end_period   = "2024.12"

# # # Takes raw monthly CPS and combines it with AI exposure data
# build_linked_dataset(ID, start_period, end_period)

# build_other_output(ID, start_period, end_period, out_path, metric = 'gpt4_rubric1_beta', freq = 3)


# # build_foreign(ID, start_period, end_period, metric = 'gpt4_rubric1_beta').to_csv(os.path.join(out_path, 'foreign.csv'), index = False)