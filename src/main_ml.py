import pandas
import numpy
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

from parsing     import build_linked_dataset
from calc        import calc_block, do_benchmark, build_std_output
from calc_other  import build_other_output
from calc_dd     import build_mismatch
from calc_edu_dd import build_mis_edu

stamp = datetime.datetime.now().strftime('%Y%m%d%H%M')
#out_path = os.path.join("/gpfs/gibbs/project/sarin/shared/model_data/AI-Employment-Model", stamp)
out_path = os.path.join("/gpfs/gibbs/project/sarin/jmk263/Repositories/AI-Employment-Model/output", stamp)
build_data = False

if not os.path.exists(out_path):
    os.makedirs(out_path)

ID = '2025050812'
start_period = "1980.1"
end_period = "1987.12"

# build_linked_dataset(ID, start_period, end_period, "OCC1990")

ID = '2025050211'
start_period = "1995.1"
end_period = "2002.12"

# build_linked_dataset(ID, start_period, end_period)

start_period = "2015.1"
end_period = "2019.12"

# build_linked_dataset(ID, start_period, end_period)

ID = '2025071609'

start_period = "2022.11"
end_period   = "2025.5"

# Takes raw monthly CPS and combines it with AI exposure data
# build_linked_dataset(ID, start_period, end_period)

# build dissimilarity data by historical time periods
params = pandas.DataFrame({
    'ID':    ['2025050812', '2025050211', '2025050211', ID],
    'start': ["1983.1","1996.1", "2016.1", start_period],
    'end':   ["1987.12", "2002.12", "2019.12", end_period],
    'label': ["Computers", "Internet", "Control", "AI"]
    }
)

deltas = pandas.DataFrame(columns=['MONTH', "dissimilarity"])
for i in range(0, len(params)):
    deltas = pandas.concat([deltas, build_mismatch(params.loc[i, 'ID'], params.loc[i, 'start'], params.loc[i, 'end'], 3).assign(
        label = params.loc[i, 'label']
    )])

deltas.pivot(
    index   = 'months_gone',
    columns = 'label',
    values  = 'dissimilarity'
).reset_index().to_csv(os.path.join(out_path, "dissimilarity.csv"), index = False)


# build dissimilarity data by recent time periods
params = pandas.DataFrame({
    'ID':    [ID, ID, ID, ID],
    'start': ["2021.01","2022.01", "2022.11", "2022.07"],
    'end':   [end_period, end_period, end_period, end_period],
    'label': ["Jan21", "Jan22", "AI (Nov22)", "Jul22"]
    }
)


# by industry analyses
recent = pandas.DataFrame({
    'name':['recent', 'information', 'financial_activities', 'professional_and_business_services'],
    'industry': ['','6470.6781','6870.7190','7270.7790']
})

for j in range(0, len(recent)):
    print(recent.loc[j,'industry'])
    deltas = pandas.DataFrame(columns=['MONTH', "dissimilarity"])
    for i in range(0, len(params)):
        deltas = pandas.concat([deltas, build_mismatch(params.loc[i, 'ID'], params.loc[i, 'start'], params.loc[i, 'end'], 3, '', recent.loc[j,'industry']).assign(
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
    'industry': ['','6470.6781','6870.7190','7270.7790']
})
deltas = pandas.DataFrame(columns=['MONTH', "dissimilarity"])
for i in range(0, len(params)):
        deltas = pandas.concat([deltas, build_mismatch(ID, start_period, end_period, 3, '', params2.loc[i,'industry']).assign(
            label = params2.loc[i, 'label']
        )])
deltas.pivot(
    index   = 'months_gone',
    columns = 'label',
    values  = 'dissimilarity'
).reset_index().to_csv(os.path.join(out_path, 'industry.csv'), index = False)


# by recency of graduation
recent_grads = build_mis_edu(ID, start_period, end_period, 3).to_csv(os.path.join(out_path, "recent_grads_dissimilarity.csv"), index = False)

# Builds graphs depicting headline employment metrics
# print("Alpha")
# build_other_output(ID, start_period, end_period, out_path, metric = 'gpt4_rubric1_alpha', freq = 1)
print("Beta")
build_other_output(ID, start_period, end_period, out_path, metric = 'gpt4_rubric1_beta', freq = 1)
# print("Gamma")
# build_other_output(ID, start_period, end_period, out_path, metric = 'gpt4_rubric1_gamma', freq = 1)

