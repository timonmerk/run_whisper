import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests

import matplotlib as mpl
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'font.size': 10
})

l_video_features = [
    "AU_1", "AU_2", "AU_4", "AU_5", "AU_6", "AU_7", "AU_9", "AU_10",
    "AU_11", "AU_12", "AU_13", "AU_14", "AU_15", "AU_16", "AU_17",
    "AU_18", "AU_19", "AU_20", "AU_22", "AU_23", "AU_24", "AU_25",
    "AU_26", "AU_27", "AU_32", "AU_38", "AU_39",
    "AU_L1", "AU_R1", "AU_L2", "AU_R2", "AU_L4", "AU_R4",
    "AU_L6", "AU_R6", "AU_L10", "AU_R10",
    "AU_L12", "AU_R12", "AU_L14", "AU_R14"
]

READ_RS = True

PATH_SUDS = "/Users/Timon/Documents/Houston/OCD_RCS/OCD_RCS/neural_audio_fau_combined.csv"
PATH_RS = "/Users/Timon/Documents/Houston/whisper/audio_neural_features_combined_rs.csv"

if READ_RS:
    PATH_DATA = PATH_RS
else:
    PATH_DATA = PATH_SUDS
df_features = pd.read_csv(PATH_DATA)
if READ_RS is False:
    df_features["time"] = pd.to_datetime(df_features["time"])
else:   
    df_features["date"] = pd.to_datetime(df_features["date"])
    df_features = df_features.drop(columns=["subject"])
    df_features = df_features.rename(columns={"sub": "subject"})
    # if columns start with FAU_, rename the FAU_
    df_features = df_features.rename(columns={c: c[4:] for c in df_features.columns if c.startswith("FAU_")})
    # remove cols that contain "corr"
    df_features = df_features[[c for c in df_features.columns if "corr" not in c and "psd" not in c]]
    df_features = df_features.rename(columns={"YBOCS II Total Score": "score"})


corrs_p_val = []
for sub in df_features["subject"].unique():
    df_sub = df_features[df_features["subject"] == sub]
    for col in df_sub.columns:
        if col not in l_video_features:
            continue
        row_1 = df_sub[col].values
        row_2 = df_sub["score"].values
        # exclude nans
        mask = ~np.isnan(row_1) & ~np.isnan(row_2)
        if np.sum(mask) < 2:
            continue
        
        corr, p_val = pearsonr(row_1[mask], row_2[mask])
        corrs_p_val.append({
            "feature": col,
            "corr": corr,
            "p_val": p_val,
            "shuffled": False,
            "subject": sub
        })
        p_val_shuffled_group = []
        corr_shuffled_group = []
        #for _ in range(50):
        #    corr_shuffled, p_val_shuffled = pearsonr(np.random.permutation(row_1[mask]), row_2[mask])
        corr_shuffled, p_val_shuffled = pearsonr(np.random.permutation(row_1[mask]), row_2[mask])
        p_val_shuffled_group.append(p_val_shuffled)
        corr_shuffled_group.append(corr_shuffled)
        p_val_shuffled = np.mean(p_val_shuffled_group)
        corr_shuffled = np.mean(corr_shuffled_group)
        corrs_p_val.append({
            "feature": col,
            "corr": corr_shuffled,
            "p_val": p_val_shuffled,
            "shuffled": True,
            "subject": sub
        })

df_corrs_p_val = pd.DataFrame(corrs_p_val)
df_corrs_p_val["corr_abs"] = np.abs(df_corrs_p_val["corr"])

df_corrs_p_val.query("p_val < 0.01").groupby("shuffled").count()
# SUDS: num correlations p<0.01 shuffled: 4, non-shuffled 45
# RS: num correlations p<0.01 shuffled: 4, non-shuffled 5

plt.figure(figsize=(10, 3))
plt.subplot(1, 4, 1)
#plt.hist(df_corrs_p_val.query("shuffled == False")["p_val"], bins=50, alpha=0.7, label="True distr")
sns.histplot(data = df_corrs_p_val, x="p_val", bins=30, stat="density",  kde=False, palette="viridis",
             hue="shuffled")  # hue="subject",

plt.title("P-values True")

plt.subplot(1, 4, 2)
sns.histplot(data=df_corrs_p_val, x="corr_abs", bins=30, hue="shuffled",
             stat="density", kde=False,  palette="viridis")
plt.title("P-values Shuffled")
sns.despine()

plt.subplot(1, 4, 3)
# show p<0.05 threshold corrected number of significant correlations as bar plot
df_corrs_p_val["significant"] = df_corrs_p_val["p_val"] < 0.05
df_counts = df_corrs_p_val.query("significant == True").groupby(["subject", "shuffled"]).size().reset_index(name='counts')
sns.barplot(data=df_counts, x="subject", y="counts", hue="shuffled", palette="viridis")
plt.title("p<0.05")
plt.xlabel("Subject")
plt.ylabel("Count")

plt.subplot(1, 4, 4)
# show p<0.05 threshold corrected number of significant correlations as bar plot
df_corrs_p_val["significant"] = df_corrs_p_val["p_val"] < 0.05/len(l_video_features)
df_counts = df_corrs_p_val.query("significant == True").groupby(["subject", "shuffled"]).size().reset_index(name='counts')
sns.barplot(data=df_counts, x="subject", y="counts", hue="shuffled", palette="viridis")
plt.title("p < Bonf. corr.")
plt.xlabel("Subject")
plt.ylabel("Count")
plt.suptitle(f"RS={READ_RS}")
plt.tight_layout()

plt.savefig(f"video_correlations_RS_{READ_RS}_incl_r.pdf")
