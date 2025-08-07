import pandas
import numpy
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

from parsing    import build_linked_dataset
from calc       import calc_block, do_benchmark, build_std_output
from calc_other import build_other_output
from calc_dd    import build_mismatch

stamp = datetime.datetime.now().strftime('%Y%m%d%H%M')
#stamp = '202506231821'
out_path = os.path.join("/gpfs/gibbs/project/sarin/jmk263/Repositories/AI-Employment-Model/output", stamp)

if not os.path.exists(out_path):
    os.makedirs(out_path)

ID = '2025050812'
start_period = "1980.1"
end_period = "1987.12"

#build_linked_dataset(ID, start_period, end_period, "OCC1990")

ID = '2025050211'
start_period = "1995.1"
end_period = "2002.12"

#build_linked_dataset(ID, start_period, end_period

start_period = "2015.1"
end_period = "2019.12"

#build_linked_dataset(ID, start_period, end_period)

ID = '2025071609'

start_period = "2021.1"
end_period   = "2025.5"

# Takes raw monthly CPS and combines it with AI exposure data

ID = '2025080614'
start_period = "2000.1"
end_period   = "2025.6"

build_linked_dataset(ID, start_period, end_period, occ_code="OCC2010", simple = True)
bark
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

# Builds graphs depicting headline employment metrics
print("Alpha")
build_other_output(ID, start_period, end_period, out_path, metric = 'gpt4_rubric1_alpha', freq = 1)
print("Beta")
build_other_output(ID, start_period, end_period, out_path, metric = 'gpt4_rubric1_beta', freq = 1)
print("Gamma")
build_other_output(ID, start_period, end_period, out_path, metric = 'gpt4_rubric1_gamma', freq = 1)