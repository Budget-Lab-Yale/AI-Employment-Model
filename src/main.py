import pandas
import numpy
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

from calc_other  import build_other_output
from calc_dd     import build_mismatch
from calc_edu_dd import build_mis_edu
from calc_ind_dd import build_mis_ind

stamp = datetime.datetime.now().strftime('%Y%m%d%H%M')
stamp = "202601201257"
out_path = os.path.join("/gpfs/gibbs/project/sarin/shared/model_data/AI-Employment-Model")
#out_path = os.path.join("/gpfs/gibbs/project/sarin/shared/model_data/AI-Employment-Model/output", stamp)

if not os.path.exists(os.path.join(out_path, "output", stamp)):
    os.makedirs(os.path.join(out_path, "output", stamp))

start_period = "2021.12"
end_period   = "2025.12"

print("Dissimilarity by Period")
# Build dissimilarity data by historical time periods
params = pandas.DataFrame({
    'ID':    ["Computers", "Internet", "Control", "AI"],
    'start': ["1983.2","1995.2", "2015.2", start_period],
    'end':   ["1989.12", "2002.12", "2019.12", end_period]
    }
)
deltas = pandas.DataFrame(columns=['MONTH', "dissimilarity"])
for i in range(0, len(params)):
    deltas = pandas.concat([deltas, build_mismatch(params.loc[i, 'ID'], params.loc[i, 'start'], params.loc[i, 'end'], freq=12, pre_trend=12).assign(
        label = params.loc[i, 'ID']
    )])

deltas.pivot(
    index   = 'months_gone',
    columns = 'label',
    values  = 'dissimilarity'
).reset_index().to_csv(os.path.join(out_path, "output", stamp, "dissimilarity.csv"), index = False)


print("AI Dissimilarity by start period")
# Build dissimilarity data by recent time periods
params = pandas.DataFrame({
    'ID':    ["AI", "AI", "AI", "AI"],
    'start': ["2020.02","2021.02", "2021.12", "2021.08"],
    'end':   [end_period, end_period, end_period, end_period],
    'label': ["Jan21", "Jan22", "AI (Nov22)", "Jul22"]
    }
)

print("Dissimilarity by Industry")
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
    ).reset_index().to_csv(os.path.join(out_path, "output", stamp, name), index = False)

params2 = pandas.DataFrame({
    'label':['all', 'information', 'financial_activities', 'professional_and_business_services'],
    'industry': ['','6470.6860','6870.7260','7270.7790']
})
deltas = pandas.DataFrame(columns=['MONTH', "dissimilarity"])
for i in range(0, len(params2)):
        deltas = pandas.concat([deltas, build_mismatch("AI", start_period, end_period, 12, '', params2.loc[i,'industry']).assign(
            label = params2.loc[i, 'label']
        )])
deltas.pivot(
    index   = 'months_gone',
    columns = 'label',
    values  = 'dissimilarity'
).reset_index().to_csv(os.path.join(out_path, "output", stamp, 'industry.csv'), index = False)

# all industries back to 2004
#ID = '2025092202'
start_period = "2003.1"

end_period   = "2025.12"

build_mis_ind("Industry", start_period, end_period, 12).to_csv(os.path.join(out_path, "output", stamp, 'every.csv'), index = False) 

# By recency of graduation
# The following start date of November 2020 allows us to calculate the 3 month trailing average. A programmatic fix for this is forthcoming.
recent_grads = build_mis_edu("dissimilarity/Industry", "2020.11", end_period, 3).to_csv(os.path.join(out_path, "output", stamp, "recent_grads_dissimilarity_22_25.csv"), index = False)
recent_grads = build_mis_edu("dissimilarity/Industry", "2014.11", end_period, 3).to_csv(os.path.join(out_path, "output", stamp, "recent_grads_dissimilarity_15_25.csv"), index = False)
recent_grads = build_mis_edu("input/august", "2020.11", end_period, 3).to_csv(os.path.join(out_path, "output", stamp, "recent_grads_dissimilarity_lone.csv"), index = False)

# Builds exposure/usage metrics over time
start_period = '2022.9'

write_path = os.path.join(out_path, "output", stamp, 'gpt4_rubric1_beta')
print("Beta")
for m in ["march", "march_fm", "august_claude", "august_claude_fm", "august", "august_fm", "november", "november_fm"]:
  if not os.path.exists(write_path):
    os.makedirs(write_path)
  print(f"Building output for: {m}")
  build_other_output(m, start_period, end_period, write_path, metric = 'gpt4_rubric1_beta', freq = 3)  

