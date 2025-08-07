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

data = pandas.read_csv("/gpfs/gibbs/project/sarin/jmk263/Repositories/AI-Employment-Model/output/202508011324/gpt4_rubric1_beta/openai_vs_anthropic.csv")
labels = pandas.read_csv("/gpfs/gibbs/project/sarin/jmk263/Repositories/AI-Employment-Model/resources/crosswalk.csv")[["Occupation", "Code", "cps_code"]].rename(
    columns = {
        "cps_code": "OCC"
    }
).assign(
    major_code = lambda x: x["Code"].str[:2].astype(int)
).merge(
    pandas.read_csv("/gpfs/gibbs/project/sarin/jmk263/Repositories/AI-Employment-Model/resources/soc_major.csv"),
    on = "major_code",
    how = "left"
)
data2=data[data["gpt4_rubric1_beta"].notna()]
data3=data[data["not_filtered"].notna()]

out = data3[data3["gpt4_rubric1_beta"].notna()].assign(
    quadrant = lambda x: numpy.where(
        x["gpt4_rubric1_beta"] > .5,
        numpy.where(x["not_filtered"] > .5, 1, 4),
        numpy.where(x["not_filtered"] > .5, 2, 3),
    )
).merge(
    labels,
    on = "OCC",
    how = "left"
).assign(count = 1)

distro = out.groupby(["quadrant", "major_occ"])["count"].sum().groupby("quadrant").apply(lambda x: x*100 / x.sum())
print(distro)
distro.to_csv("/gpfs/gibbs/project/sarin/jmk263/Repositories/AI-Employment-Model/output/202508011324/gpt4_rubric1_beta/distro.csv")

""" for q in range(1,5):
    print(f"Occupations in Quadrant: {q}")
    frog = out[out["quadrant"] == q]["Occupation"]
    print(frog)
 """
out.to_csv("/gpfs/gibbs/project/sarin/jmk263/Repositories/AI-Employment-Model/output/202508011324/gpt4_rubric1_beta/full.csv")
