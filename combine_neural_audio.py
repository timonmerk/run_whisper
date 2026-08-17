import pandas as pd
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests
from scipy.stats import zscore

l_audio_features = ["Loudness_sma3","alphaRatio_sma3","hammarbergIndex_sma3","slope0-500_sma3","slope500-1500_sma3","spectralFlux_sma3","mfcc1_sma3","mfcc2_sma3","mfcc3_sma3","mfcc4_sma3","F0semitoneFrom27.5Hz_sma3nz","jitterLocal_sma3nz","shimmerLocaldB_sma3nz","HNRdBACF_sma3nz","logRelF0-H1-H2_sma3nz","logRelF0-H1-A3_sma3nz","F1frequency_sma3nz","F1bandwidth_sma3nz","F1amplitudeLogRelF0_sma3nz","F2frequency_sma3nz","F2bandwidth_sma3nz","F2amplitudeLogRelF0_sma3nz","F3frequency_sma3nz","F3bandwidth_sma3nz","F3amplitudeLogRelF0_sma3nz","F0semitoneFrom27.5Hz_sma3nz_amean","F0semitoneFrom27.5Hz_sma3nz_stddevNorm","F0semitoneFrom27.5Hz_sma3nz_percentile20.0","F0semitoneFrom27.5Hz_sma3nz_percentile50.0","F0semitoneFrom27.5Hz_sma3nz_percentile80.0","F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2","F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope","F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope","F0semitoneFrom27.5Hz_sma3nz_meanFallingSlope","F0semitoneFrom27.5Hz_sma3nz_stddevFallingSlope","loudness_sma3_amean","loudness_sma3_stddevNorm","loudness_sma3_percentile20.0","loudness_sma3_percentile50.0","loudness_sma3_percentile80.0","loudness_sma3_pctlrange0-2","loudness_sma3_meanRisingSlope","loudness_sma3_stddevRisingSlope","loudness_sma3_meanFallingSlope","loudness_sma3_stddevFallingSlope","spectralFlux_sma3_amean","spectralFlux_sma3_stddevNorm","mfcc1_sma3_amean","mfcc1_sma3_stddevNorm","mfcc2_sma3_amean","mfcc2_sma3_stddevNorm","mfcc3_sma3_amean","mfcc3_sma3_stddevNorm","mfcc4_sma3_amean","mfcc4_sma3_stddevNorm","jitterLocal_sma3nz_amean","jitterLocal_sma3nz_stddevNorm","shimmerLocaldB_sma3nz_amean","shimmerLocaldB_sma3nz_stddevNorm","HNRdBACF_sma3nz_amean","HNRdBACF_sma3nz_stddevNorm","logRelF0-H1-H2_sma3nz_amean","logRelF0-H1-H2_sma3nz_stddevNorm","logRelF0-H1-A3_sma3nz_amean","logRelF0-H1-A3_sma3nz_stddevNorm","F1frequency_sma3nz_amean","F1frequency_sma3nz_stddevNorm","F1bandwidth_sma3nz_amean","F1bandwidth_sma3nz_stddevNorm","F1amplitudeLogRelF0_sma3nz_amean","F1amplitudeLogRelF0_sma3nz_stddevNorm","F2frequency_sma3nz_amean","F2frequency_sma3nz_stddevNorm","F2bandwidth_sma3nz_amean","F2bandwidth_sma3nz_stddevNorm","F2amplitudeLogRelF0_sma3nz_amean","F2amplitudeLogRelF0_sma3nz_stddevNorm","F3frequency_sma3nz_amean","F3frequency_sma3nz_stddevNorm","F3bandwidth_sma3nz_amean","F3bandwidth_sma3nz_stddevNorm","F3amplitudeLogRelF0_sma3nz_amean","F3amplitudeLogRelF0_sma3nz_stddevNorm","alphaRatioV_sma3nz_amean","alphaRatioV_sma3nz_stddevNorm","hammarbergIndexV_sma3nz_amean","hammarbergIndexV_sma3nz_stddevNorm","slopeV0-500_sma3nz_amean","slopeV0-500_sma3nz_stddevNorm","slopeV500-1500_sma3nz_amean","slopeV500-1500_sma3nz_stddevNorm","spectralFluxV_sma3nz_amean","spectralFluxV_sma3nz_stddevNorm","mfcc1V_sma3nz_amean","mfcc1V_sma3nz_stddevNorm","mfcc2V_sma3nz_amean","mfcc2V_sma3nz_stddevNorm","mfcc3V_sma3nz_amean","mfcc3V_sma3nz_stddevNorm","mfcc4V_sma3nz_amean","mfcc4V_sma3nz_stddevNorm","alphaRatioUV_sma3nz_amean","hammarbergIndexUV_sma3nz_amean","slopeUV0-500_sma3nz_amean","slopeUV500-1500_sma3nz_amean","spectralFluxUV_sma3nz_amean","loudnessPeaksPerSec","VoicedSegmentsPerSec","MeanVoicedSegmentLengthSec","StddevVoicedSegmentLengthSec","MeanUnvoicedSegmentLength","StddevUnvoicedSegmentLength","equivalentSoundLevel_dBp","arousal","dominance","valence"] + [f"Dim {i}" for i in range(1024)] + ["duration"]


l_audio_groups = {
    "LLDs":     # Intensity low level descriptors
        ["Loudness_sma3",
        # Spectral shape / tilt
        "alphaRatio_sma3",
        "hammarbergIndex_sma3",
        "slope0-500_sma3",
        "slope500-1500_sma3",
        "spectralFlux_sma3",
        # Cepstral features (MFCCs)
        "mfcc1_sma3",
        "mfcc2_sma3",
        "mfcc3_sma3",
        "mfcc4_sma3",
        # Pitch & prosody
        "F0semitoneFrom27.5Hz_sma3nz",
        # Voice quality
        "jitterLocal_sma3nz",
        "shimmerLocaldB_sma3nz",
        "HNRdBACF_sma3nz",
        # Harmonics
        "logRelF0-H1-H2_sma3nz",
        "logRelF0-H1-A3_sma3nz",
        # Formants
        "F1frequency_sma3nz",
        "F1bandwidth_sma3nz",
        "F1amplitudeLogRelF0_sma3nz",
        "F2frequency_sma3nz",
        "F2bandwidth_sma3nz",
        "F2amplitudeLogRelF0_sma3nz",
        "F3frequency_sma3nz",
        "F3bandwidth_sma3nz",
        "F3amplitudeLogRelF0_sma3nz"],
    "Functionals": l_audio_features[25:113],  # functionals: basic stats (mean, std, etc.), percentiles, slope relates /temporal, higher order moments (skewness)
    "VAD": l_audio_features[113:116],
    "Embeddings": l_audio_features[116:-1],
}

READ_RS = False

PATH_BASE = "/Users/Timon/Library/CloudStorage/Box-Box/APL_BCM_Share_SUDS/Audio_Analysis"
if READ_RS is False:
    PATH_AUDIO_FEATURES = os.path.join(PATH_BASE, "output_audio_features")
else:
    PATH_AUDIO_FEATURES = os.path.join(PATH_BASE, "output_audio_features_resting-state")
files_ = os.listdir(PATH_AUDIO_FEATURES)

df_ = []
for f in files_:
    df = pd.read_csv(os.path.join(PATH_AUDIO_FEATURES, f))
    df_ = df_ + [df]
df_audio = pd.concat(df_, axis=0, ignore_index=True)
if READ_RS:
    df_audio["date"] = pd.to_datetime(df_audio["date"])
    df_audio["sub"] = df_audio["file"].apply(lambda x: int(os.path.basename(x).split('_')[0]))
else:
    df_audio['time'] = pd.to_datetime(df_audio['time'])

if READ_RS:
    PATH_FEATURES = "/Users/Timon/Documents/Houston/resting_state_OCD/FAUS_rs/fau_neural_combined.csv"
else:
    PATH_FEATURES = "/Users/Timon/Documents/Houston/OCD_RCS/OCD_RCS/all_subjects_features.csv"

df_features = pd.read_csv(PATH_FEATURES)
if READ_RS:
    df_features["date"] = pd.to_datetime(df_features["date"])
else:
    df_features["time"] = pd.to_datetime(df_features["time"])
# iterate through df_features, and extract for each row the plus minus 1 minute from df_audio, which also has a time column

READ_AUDIO_FEATURES = True
if READ_AUDIO_FEATURES:
    dfs = []
    dfs_speech = []
    dfs_non_speech = []
    for i, row in tqdm(df_features.iterrows()):
        if READ_RS is False:
            start_time = row['time'] - pd.Timedelta(minutes=1)
            end_time = row['time'] + pd.Timedelta(minutes=1)
            sub = row["subject"]
            df_audio_sub = df_audio[df_audio['sub'] == sub]
            mask = (df_audio_sub['time'] >= start_time) & (df_audio_sub['time'] <= end_time)
            df_audio_sub_sud = df_audio_sub[mask]
            # drop column file, score, text, speaker
            df_audio_sub_sud_ = df_audio_sub_sud.drop(columns=["file", "score", "text", "speaker", "date", "time", "sub", "start", "end", "time_min_sec"], errors="ignore")
            mean_features = df_audio_sub_sud_.mean(axis=0)
            mean_features["duration_sum"] = df_audio_sub_sud_["duration"].sum()
            mean_features["duration_ratio"] = mean_features["duration_sum"] / (2 * 60)  # divide by 2 minutes
            
        else:
            sub = int(row["subject"][4:])
            date = row["date"]
            df_audio_sub = df_audio.query("sub == @sub and date == @date")
            df_audio_sub_date_ = df_audio_sub.drop(columns=["file", "score", "text", "speaker", "date", "time", "subject", "start", "end", "duration"], errors="ignore")
            mean_features = df_audio_sub_date_.mean(axis=0)
        # combine row and mean_features
        combined = pd.concat([row, mean_features])

        dfs = dfs + [combined]

    df_audio_features_comb = pd.DataFrame(dfs)
    if READ_RS:
        df_audio_features_comb.to_csv("audio_neural_features_combined_rs.csv", index=False)
    else:
        df_audio_features_comb.to_csv("audio_neural_features_combined.csv", index=False)
        #df_audio_features_comb.to_csv("audio_neural_features_combined_suds_incl_speech_ratio.csv", index=False)
else:
    if READ_RS:
        df_audio_features_comb = pd.read_csv("audio_neural_features_combined_rs.csv")
    else:   
        df_audio_features_comb = pd.read_csv("audio_neural_features_combined.csv")


corrs_p_val = []
for sub in df_audio_features_comb["subject"].unique():
    df_sub = df_audio_features_comb[df_audio_features_comb["subject"] == sub]
    for col in df_sub.columns:
        if col in ["score", "subject"]:
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

plt.figure(figsize=(7, 3))
plt.subplot(1, 3, 1)
#plt.hist(df_corrs_p_val.query("shuffled == False")["p_val"], bins=50, alpha=0.7, label="True distr")
sns.histplot(data = df_corrs_p_val.query("shuffled == False"), x="p_val", bins=30, stat="density",  kde=False, multiple="stack", palette="viridis")  # hue="subject",
plt.ylim(0, 3.8)
plt.title("P-values True")
plt.subplot(1, 3, 2)
sns.histplot(data = df_corrs_p_val.query("shuffled == True"), x="p_val", bins=30, stat="density", kde=False, multiple="stack", palette="viridis")
plt.title("P-values Shuffled")
plt.ylim(0, 3.8)
sns.despine()

plt.subplot(1, 3, 3)
df_no_shuffle = df_corrs_p_val.query("shuffled == False")
rejected, pvals_corrected, _, _ = multipletests(
    df_no_shuffle["p_val"], alpha=0.05, method="fdr_bh"
)
df_no_shuffle["p_fdr"] = pvals_corrected
df_no_shuffle["significant"] = rejected
df_shuffle = df_corrs_p_val.query("shuffled == True")
rejected, pvals_corrected, _, _ = multipletests(
    df_shuffle["p_val"], alpha=0.05, method="fdr_bh"
)
df_shuffle["p_fdr"] = pvals_corrected
df_shuffle["significant"] = rejected
df_sig_test = pd.concat([df_no_shuffle, df_shuffle], axis=0, ignore_index=True)
df_significant = df_sig_test[df_sig_test["significant"] == True]
df_significant["positive"] = df_significant["corr"] > 0
df_significant = df_significant.groupby(["subject", "shuffled", "positive"]).size().reset_index(name='counts')

# there are no shuffled significant correlations ---> CHECK for future runs
sns.barplot(data=df_significant, x="subject", y="counts", hue="positive", palette="viridis")
plt.title("Number of Significant Correlations per Subject")
plt.xlabel("Subject")
plt.ylabel("Number of Significant Correlations")
plt.savefig("audio_correlations.pdf")
plt.show()


# z-score normalization using npy
df_audio_zs = df_audio_features_comb.copy().iloc[:, 1020:2160]
df_audio_zs_mean = df_audio_zs.mean(axis=0)
df_audio_zs_std = df_audio_zs.std(axis=0)
df_audio_zs = (df_audio_zs - df_audio_zs_mean) / df_audio_zs_std
# delete empty rows
df_audio_zs = df_audio_zs.dropna(how='all')
# get all columns in a colum "feature", "value"
df_audio_zs_melt = df_audio_zs.melt(var_name="feature", value_name="value")
# apply to the feature column the group names
def get_feature_group(feature):
    for group, features in l_audio_groups.items():
        if feature in features:
            return group
    return "Other"
df_audio_zs_melt["group"] = df_audio_zs_melt["feature"].apply(get_feature_group)

subjects = df_features["subject"]
nan_idx = df_features["SC_L_RawHjorth_Complexity"].isnull()
subjects = subjects[~nan_idx]

# make a heatmap plot of df_audio_zs for each group
num_groups = len(l_audio_groups)
fig, axes = plt.subplots(1, num_groups, figsize=(10, 4), sharex=True)
for ax, (group, features) in zip(axes, l_audio_groups.items()):
    df_group = df_audio_zs[features]
    im = ax.imshow(df_group.T, aspect="auto",
                   cmap="viridis", vmin=-3, vmax=3, interpolation="none")
    ax.set_title(f"{group}")
    # set x ticks to subject changes
    subject_changes = np.where(subjects.diff().fillna(0) != 0)[0]
    ax.set_xticks(subject_changes)
    ax.set_xticklabels(subjects.iloc[subject_changes], rotation=90)
# save figure
plt.savefig("audio_features_groups_heatmap.pdf", bbox_inches='tight')


plt.figure(figsize=(10, 6))
plt.imshow(df_audio_zs, aspect="auto")

# get the df part where df_audio_features_comb['valence'] is null
df_null = df_audio_features_comb[df_audio_features_comb['valence'].isnull()]

audio_corrs_start_idx = 1020
audio_corrs_end_idx = 2159
score_col = df_audio_features_comb["score"]
patient = df_audio_features_comb["subject"]
df_audio_features_comb = df_audio_features_comb.iloc[:, audio_corrs_start_idx:audio_corrs_end_idx]
df_audio_features_comb["score"] = score_col
df_audio_features_comb["subject"] = patient




#plt.hist(x=df_corrs_p_val.query("shuffled == False")["corr"], bins=50, alpha=0.7, label="True")
sns.histplot(data = df_corrs_p_val, x="corr",
             bins=50, stat="density",kde=False, hue="shuffled",
             label="KDE True", palette="viridis", multiple="stack", )  #
plt.title("Pearson correlations True")
sns.despine()
plt.subplot(2, 2, 2)
#plt.hist(df_corrs_p_val.query("shuffled == True")["corr"], bins=50, alpha=0.7, label="Shuffled", color='orange')
sns.histplot(data = df_corrs_p_val.query("shuffled == True"), x="corr", bins=50, stat="density", hue="subject", kde=False,  multiple="stack", palette="viridis"
             )
plt.legend()
sns.despine()
plt.suptitle("Pearson correlations Shuffled")

plt.legend()


# make a barplot with number of significant correltaions per subject
df_significant = df_corrs_p_val[df_corrs_p_val["p_val"] < 0.05].groupby(["subject", "shuffled"]).size().reset_index(name='counts')

plt.figure()
sns.barplot(data=df_significant, x="subject", y="counts", hue="shuffled", palette="viridis")
plt.title("Number of Significant Correlations per Subject")
plt.xlabel("Subject")
plt.ylabel("Number of Significant Correlations")
plt.legend(title="Shuffled")
plt.show()

df_sign_audio_features = df_corrs_p_val[df_corrs_p_val["p_val"] < 0.05].query("shuffled == False")
# count how many times each feature is significant across subjects
df_sign_audio_features_count = df_sign_audio_features.groupby("feature").size().reset_index(name='counts')
# sort by counts
df_sign_audio_features_count = df_sign_audio_features_count.sort_values(by='counts', ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(data=df_sign_audio_features_count.head(50), x="counts", y="feature", palette="viridis")
plt.title("Top 20 Significant Audio Features Across Subjects")
plt.tight_layout()


# compute the all to all correlation matrix
df_audio_features_comb_corr = df_audio_features_comb.corr()
# sort it by mean correlation with score
df_audio_features_comb_corr = df_audio_features_comb_corr.reindex(df_audio_features_comb_corr.mean().sort_values(ascending=False).index, axis=0)
plt.figure(figsize=(12, 10))
sns.heatmap(df_audio_features_comb_corr, cmap='coolwarm', center=0, vmin=-0.2, vmax=0.2)
plt.title("Correlation Matrix of Audio Features and Score")



# run the correlation within each subject
subs = df_audio_features_comb["subject"].unique()
corrs = []
corrs_shuffle = []
for sub in subs:
    df_sub = df_audio_features_comb[df_audio_features_comb["subject"] == sub]
    df_sub_shuffles = df_sub.copy()
    df_sub_shuffles["score"] = np.random.permutation(df_sub_shuffles["score"].values)
    corr = df_sub.corr()["score"]
    corr_shuffle = df_sub_shuffles.corr()["score"]
    corr["subject"] = sub
    corr_shuffle["subject"] = sub
    corrs = corrs + [corr]
    corrs_shuffle = corrs_shuffle + [corr_shuffle]
df_corrs = pd.DataFrame(corrs)
df_corrs_shuffle = pd.DataFrame(corrs_shuffle)

# drop score column
df_corrs = df_corrs.drop(columns=["score"], errors="ignore")
df_corrs_shuffle = df_corrs_shuffle.drop(columns=["score"], errors="ignore")

res_shuffled = []
res_true = []
for sub in subs:
    corrs_sub = df_corrs.query("subject == @sub").T.sort_values(by="score").iloc[:-1]
    plt.plot(corrs_sub.values, label=f"Subject {sub}", alpha=0.5)
    corrs_sub_shuffle = df_corrs_shuffle.query("subject == @sub").T.sort_values(by="score").iloc[:-1]
    plt.plot(corrs_sub_shuffle.values, label=f"Subject {sub} (shuffled)", alpha=0.2, linestyle='--')
    res_true = res_true + [corrs_sub.values]
    res_shuffled = res_shuffled + [corrs_sub_shuffle.values]
res_true = np.array(res_true)
res_shuffled = np.array(res_shuffled)
# plot mean
plt.plot(res_true.mean(axis=0), label="Mean true", color='black', linewidth=2)
plt.plot(res_shuffled.mean(axis=0), label="Mean shuffled", color='grey', linewidth=2, linestyle='--')
plt.legend()
plt.axhline(0, color='black', linestyle='--')
plt.xlabel("Features sorted by correlation with score")
plt.show()

plt.plot((res_true - res_shuffled).mean(axis=0)[:, 0])

df_corrs_shuffle = pd.DataFrame(corrs_shuffle)

mean_res = df_corrs.mean(axis=0)
std_res = df_corrs.std(axis=0)

mean_res_shuffle = df_corrs_shuffle.mean(axis=0)
std_res_shuffle = df_corrs_shuffle.std(axis=0)

trace_shuffled = mean_res_shuffle.iloc[:-2].sort_values(ascending=False)
trace_true = mean_res.iloc[:-2].sort_values(ascending=False)

plt.figure()
plt.plot(trace_true.values, label="True", marker='o')
plt.plot(trace_shuffled.values, label="Shuffled", marker='o')
plt.legend()

# plot the top 5 and bottom 5 correlations, but show patient individual dots
top5 = mean_res.sort_values(ascending=False).index[1:6]
bottom5 = mean_res.sort_values(ascending=True).index[1:6]
top_bottom = top5.tolist() + bottom5.tolist()

corrs_best_df = df_corrs.reset_index()[["subject"] + top_bottom]

corrs_melt = corrs_best_df.melt(id_vars=["subject"], var_name="feature", value_name="correlation")

plt.figure()
sns.boxplot(data=corrs_melt, x="feature", y="correlation")
sns.swarmplot(data=corrs_melt, x="feature", y="correlation", color=".25")
plt.xticks(rotation=45)



# run tsne on df_audio_features_comb
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
df_tsne = df_audio_features_comb.copy()
# drop nan rows
df_tsne = df_tsne.dropna()
tsne_results = tsne.fit_transform(df_tsne.drop(columns=["score", "subject"], errors="ignore"))
df_tsne["tsne-2d-one"] = tsne_results[:, 0]
df_tsne["tsne-2d-two"] = tsne_results[:, 1]

df_tsne["score_normed_min_max_per_sub"] = df_tsne.groupby("subject")["score"].transform(lambda x: (x - x.min()) / (x.max() - x.min()))

plt.figure(figsize=(6, 4))
plt.subplot(1, 2, 1)
plt.scatter(df_tsne["tsne-2d-one"], df_tsne["tsne-2d-two"], c=df_tsne["score_normed_min_max_per_sub"], cmap='viridis', alpha=0.7)
plt.colorbar(label='score_normed_min_max_per_sub')
plt.title("t-SNE SUDS Score")

plt.subplot(1, 2, 2)
plt.scatter(df_tsne["tsne-2d-one"], df_tsne["tsne-2d-two"], c=pd.factorize(df_tsne["subject"])[0], cmap='Accent', alpha=0.7)
plt.colorbar(label='Subject')
# add the subject names to the colorbar
cbar = plt.colorbar(ticks=range(len(df_tsne["subject"].unique())))
cbar.ax.set_yticklabels(df_tsne["subject"].unique())
plt.tight_layout()
plt.title("t-SNE Subject")