import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

filepath1 = "../results/lda_tuning_results_7-10.csv"
filepath2 = "../results/lda_tuning_results_10-15.csv"
df1 = pd.read_csv(filepath1, sep=",", engine="python")
df2 = pd.read_csv(filepath2, sep=",", engine="python")


def round_float(s):
    """1. if s is float, round it to 0 decimals
       2. else return s as is
    """
    if isinstance(s, int):
        return s
    try:
        r = round(float(s), 2)
    except:
        r = s
    return r


df_data = pd.concat([df1, df2])
df_data = df_data.applymap(round_float).copy()
# df_data = df_data["Beta"].map(
#     lambda x: round(x, 2) if isinstance(float(x), (float)) else x
# )
print(df_data)
print(
    "Optimal Parameters:\n",
    df_data[df_data["Coherence"] == df_data["Coherence"].values.max()],
)


fig = plt.figure(figsize=(15, 9))
fig = sns.scatterplot(
    data=df_data,
    # x=df_data.index,
    x="Topics",
    y="Coherence",
    style="Alpha",
    hue="Beta",
    s=200,
)
sns.despine()
plt.xlabel("Number of Topics", fontsize=25)
plt.ylabel("Coherence Score", fontsize=25)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout()
plt.legend(loc="upper center", ncol=5, fontsize=16)
plt.savefig("../results/tm_gs_results.pdf")
