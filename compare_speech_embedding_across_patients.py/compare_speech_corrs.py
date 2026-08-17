import pandas as pd
from scipy.stats import pearsonr
from scipy.stats import permutation_test
import seaborn as sns
import numpy as np
from statsmodels.stats.multitest import multipletests
import matplotlib as mpl
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial'],
    'font.size': 10
})
import matplotlib.pyplot as plt

r = np.array([0.18, 0.13, -0.33, -0.18, 0.26, -0.01, 0.13]) # speech
r= np.array([0.14, 0.02, -0.33, -0.08, -0.06, -0.22, 0.00, ]) # FAU

# test in a permutation test if the r values are significantly different from 0
# use scipy's permutation_test function
res = permutation_test(
    (r,),
    statistic=lambda x: np.mean(x),
    permutation_type='samples',
    alternative='two-sided',
    n_resamples=5000,
    random_state=42
)
print(f"p-value: {res.pvalue}")
print(f"mean: {np.mean(r)}")
print(f"std: {np.std(r)}")

#l_audio_features = ["Loudness_sma3","alphaRatio_sma3","hammarbergIndex_sma3","slope0-500_sma3","slope500-1500_sma3","spectralFlux_sma3","mfcc1_sma3","mfcc2_sma3","mfcc3_sma3","mfcc4_sma3","F0semitoneFrom27.5Hz_sma3nz","jitterLocal_sma3nz","shimmerLocaldB_sma3nz","HNRdBACF_sma3nz","logRelF0-H1-H2_sma3nz","logRelF0-H1-A3_sma3nz","F1frequency_sma3nz","F1bandwidth_sma3nz","F1amplitudeLogRelF0_sma3nz","F2frequency_sma3nz","F2bandwidth_sma3nz","F2amplitudeLogRelF0_sma3nz","F3frequency_sma3nz","F3bandwidth_sma3nz","F3amplitudeLogRelF0_sma3nz","F0semitoneFrom27.5Hz_sma3nz_amean","F0semitoneFrom27.5Hz_sma3nz_stddevNorm","F0semitoneFrom27.5Hz_sma3nz_percentile20.0","F0semitoneFrom27.5Hz_sma3nz_percentile50.0","F0semitoneFrom27.5Hz_sma3nz_percentile80.0","F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2","F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope","F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope","F0semitoneFrom27.5Hz_sma3nz_meanFallingSlope","F0semitoneFrom27.5Hz_sma3nz_stddevFallingSlope","loudness_sma3_amean","loudness_sma3_stddevNorm","loudness_sma3_percentile20.0","loudness_sma3_percentile50.0","loudness_sma3_percentile80.0","loudness_sma3_pctlrange0-2","loudness_sma3_meanRisingSlope","loudness_sma3_stddevRisingSlope","loudness_sma3_meanFallingSlope","loudness_sma3_stddevFallingSlope","spectralFlux_sma3_amean","spectralFlux_sma3_stddevNorm","mfcc1_sma3_amean","mfcc1_sma3_stddevNorm","mfcc2_sma3_amean","mfcc2_sma3_stddevNorm","mfcc3_sma3_amean","mfcc3_sma3_stddevNorm","mfcc4_sma3_amean","mfcc4_sma3_stddevNorm","jitterLocal_sma3nz_amean","jitterLocal_sma3nz_stddevNorm","shimmerLocaldB_sma3nz_amean","shimmerLocaldB_sma3nz_stddevNorm","HNRdBACF_sma3nz_amean","HNRdBACF_sma3nz_stddevNorm","logRelF0-H1-H2_sma3nz_amean","logRelF0-H1-H2_sma3nz_stddevNorm","logRelF0-H1-A3_sma3nz_amean","logRelF0-H1-A3_sma3nz_stddevNorm","F1frequency_sma3nz_amean","F1frequency_sma3nz_stddevNorm","F1bandwidth_sma3nz_amean","F1bandwidth_sma3nz_stddevNorm","F1amplitudeLogRelF0_sma3nz_amean","F1amplitudeLogRelF0_sma3nz_stddevNorm","F2frequency_sma3nz_amean","F2frequency_sma3nz_stddevNorm","F2bandwidth_sma3nz_amean","F2bandwidth_sma3nz_stddevNorm","F2amplitudeLogRelF0_sma3nz_amean","F2amplitudeLogRelF0_sma3nz_stddevNorm","F3frequency_sma3nz_amean","F3frequency_sma3nz_stddevNorm","F3bandwidth_sma3nz_amean","F3bandwidth_sma3nz_stddevNorm","F3amplitudeLogRelF0_sma3nz_amean","F3amplitudeLogRelF0_sma3nz_stddevNorm","alphaRatioV_sma3nz_amean","alphaRatioV_sma3nz_stddevNorm","hammarbergIndexV_sma3nz_amean","hammarbergIndexV_sma3nz_stddevNorm","slopeV0-500_sma3nz_amean","slopeV0-500_sma3nz_stddevNorm","slopeV500-1500_sma3nz_amean","slopeV500-1500_sma3nz_stddevNorm","spectralFluxV_sma3nz_amean","spectralFluxV_sma3nz_stddevNorm","mfcc1V_sma3nz_amean","mfcc1V_sma3nz_stddevNorm","mfcc2V_sma3nz_amean","mfcc2V_sma3nz_stddevNorm","mfcc3V_sma3nz_amean","mfcc3V_sma3nz_stddevNorm","mfcc4V_sma3nz_amean","mfcc4V_sma3nz_stddevNorm","alphaRatioUV_sma3nz_amean","hammarbergIndexUV_sma3nz_amean","slopeUV0-500_sma3nz_amean","slopeUV500-1500_sma3nz_amean","spectralFluxUV_sma3nz_amean","loudnessPeaksPerSec","VoicedSegmentsPerSec","MeanVoicedSegmentLengthSec","StddevVoicedSegmentLengthSec","MeanUnvoicedSegmentLength","StddevUnvoicedSegmentLength","equivalentSoundLevel_dBp","arousal","dominance","valence"] + [f"Dim {i}" for i in range(1024)]# + ["duration"]
l_audio_features = ["Loudness_sma3","alphaRatio_sma3","hammarbergIndex_sma3","slope0-500_sma3","slope500-1500_sma3","spectralFlux_sma3","mfcc1_sma3","mfcc2_sma3","mfcc3_sma3","mfcc4_sma3","F0semitoneFrom27.5Hz_sma3nz","jitterLocal_sma3nz","shimmerLocaldB_sma3nz","HNRdBACF_sma3nz","logRelF0-H1-H2_sma3nz","logRelF0-H1-A3_sma3nz","F1frequency_sma3nz","F1bandwidth_sma3nz","F1amplitudeLogRelF0_sma3nz","F2frequency_sma3nz","F2bandwidth_sma3nz","F2amplitudeLogRelF0_sma3nz","F3frequency_sma3nz","F3bandwidth_sma3nz","F3amplitudeLogRelF0_sma3nz","F0semitoneFrom27.5Hz_sma3nz_amean","F0semitoneFrom27.5Hz_sma3nz_stddevNorm","F0semitoneFrom27.5Hz_sma3nz_percentile20.0","F0semitoneFrom27.5Hz_sma3nz_percentile50.0","F0semitoneFrom27.5Hz_sma3nz_percentile80.0","F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2","F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope","F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope","F0semitoneFrom27.5Hz_sma3nz_meanFallingSlope","F0semitoneFrom27.5Hz_sma3nz_stddevFallingSlope","loudness_sma3_amean","loudness_sma3_stddevNorm","loudness_sma3_percentile20.0","loudness_sma3_percentile50.0","loudness_sma3_percentile80.0","loudness_sma3_pctlrange0-2","loudness_sma3_meanRisingSlope","loudness_sma3_stddevRisingSlope","loudness_sma3_meanFallingSlope","loudness_sma3_stddevFallingSlope","spectralFlux_sma3_amean","spectralFlux_sma3_stddevNorm","mfcc1_sma3_amean","mfcc1_sma3_stddevNorm","mfcc2_sma3_amean","mfcc2_sma3_stddevNorm","mfcc3_sma3_amean","mfcc3_sma3_stddevNorm","mfcc4_sma3_amean","mfcc4_sma3_stddevNorm","jitterLocal_sma3nz_amean","jitterLocal_sma3nz_stddevNorm","shimmerLocaldB_sma3nz_amean","shimmerLocaldB_sma3nz_stddevNorm","HNRdBACF_sma3nz_amean","HNRdBACF_sma3nz_stddevNorm","logRelF0-H1-H2_sma3nz_amean","logRelF0-H1-H2_sma3nz_stddevNorm","logRelF0-H1-A3_sma3nz_amean","logRelF0-H1-A3_sma3nz_stddevNorm","F1frequency_sma3nz_amean","F1frequency_sma3nz_stddevNorm","F1bandwidth_sma3nz_amean","F1bandwidth_sma3nz_stddevNorm","F1amplitudeLogRelF0_sma3nz_amean","F1amplitudeLogRelF0_sma3nz_stddevNorm","F2frequency_sma3nz_amean","F2frequency_sma3nz_stddevNorm","F2bandwidth_sma3nz_amean","F2bandwidth_sma3nz_stddevNorm","F2amplitudeLogRelF0_sma3nz_amean","F2amplitudeLogRelF0_sma3nz_stddevNorm","F3frequency_sma3nz_amean","F3frequency_sma3nz_stddevNorm","F3bandwidth_sma3nz_amean","F3bandwidth_sma3nz_stddevNorm","F3amplitudeLogRelF0_sma3nz_amean","F3amplitudeLogRelF0_sma3nz_stddevNorm","alphaRatioV_sma3nz_amean","alphaRatioV_sma3nz_stddevNorm","hammarbergIndexV_sma3nz_amean","hammarbergIndexV_sma3nz_stddevNorm","slopeV0-500_sma3nz_amean","slopeV0-500_sma3nz_stddevNorm","slopeV500-1500_sma3nz_amean","slopeV500-1500_sma3nz_stddevNorm","spectralFluxV_sma3nz_amean","spectralFluxV_sma3nz_stddevNorm","mfcc1V_sma3nz_amean","mfcc1V_sma3nz_stddevNorm","mfcc2V_sma3nz_amean","mfcc2V_sma3nz_stddevNorm","mfcc3V_sma3nz_amean","mfcc3V_sma3nz_stddevNorm","mfcc4V_sma3nz_amean","mfcc4V_sma3nz_stddevNorm","alphaRatioUV_sma3nz_amean","hammarbergIndexUV_sma3nz_amean","slopeUV0-500_sma3nz_amean","slopeUV500-1500_sma3nz_amean","spectralFluxUV_sma3nz_amean","loudnessPeaksPerSec","VoicedSegmentsPerSec","MeanVoicedSegmentLengthSec","StddevVoicedSegmentLengthSec","MeanUnvoicedSegmentLength","StddevUnvoicedSegmentLength","equivalentSoundLevel_dBp","arousal","dominance","valence"] 

df_rs = pd.read_csv("audio_neural_features_combined_rs.csv")
df_rs["subject"] = df_rs["subject"].apply(lambda x: int(str(x[4:])))
# for each column that starts with FAU_, remove the prefix FAU_
for col in df_rs.columns:
    if col.startswith("FAU_"):
        df_rs.rename(columns={col: col[4:]}, inplace=True)
# replce column 'YBOCS II Total Score' with 'score'
df_rs.rename(columns={"YBOCS II Total Score": "score"}, inplace=True)
df_suds = pd.read_csv("/Users/Timon/Documents/Houston/OCD_RCS/OCD_RCS/neural_audio_fau_combined.csv")
# replace columsn 'score_fau' with 'score'
df_suds.rename(columns={"score_fau": "score"}, inplace=True)

# for both dataframes, correlate for each subject each l_audio_feature with score, and store the results in a new dataframe
def correlate_features_with_score(df, features):
    results = []
    subjects = df["subject"].unique()
    for subject in subjects:
        df_subject = df[df["subject"] == subject]
        for feature in features:
            if feature in df_subject.columns:
                corr = df_subject[feature].corr(df_subject["score"])
                nan_idxs_both = df_subject[[feature, "score"]].isna().any(axis=1)

                p = pearsonr(df_subject[feature][~nan_idxs_both], df_subject["score"][~nan_idxs_both])[1]
                results.append({"subject": subject, "feature": feature, "correlation": corr, "p_value": p})
    return pd.DataFrame(results)

df_rs_corrs = correlate_features_with_score(df_rs, l_audio_features)
df_suds_corrs = correlate_features_with_score(df_suds, l_audio_features)

# for each subject correlate the correlations from both dataframes
df_merged = pd.merge(df_rs_corrs, df_suds_corrs, on=["subject", "feature"], suffixes=("_rs", "_suds"))
len(l_audio_features)

# # significance after FWE
alpha = 0.05
df_merged["sig_rs_fwe"] = df_merged["p_value_rs"] < alpha / len(l_audio_features)

results = []
subjects = df_merged["subject"].unique()
for subject in subjects:
    df_subject = df_merged[df_merged["subject"] == subject]
    corr = df_subject["correlation_rs"].corr(df_subject["correlation_suds"])
    corr, p_value = pearsonr(df_subject["correlation_rs"], df_subject["correlation_suds"])
    results.append({"subject": subject, "correlation_between_corrs": corr, "p_value": p_value})
df_final = pd.DataFrame(results)

# plot df_final as a barplot, mark significant correlations
plt.figure(figsize=(2, 3))
sns.barplot(x="subject", y="correlation_between_corrs", data=df_final)
for i, row in df_final.iterrows():
    if row["p_value"] < 0.05:
        plt.text(i, row["correlation_between_corrs"] + 0.01, "*", ha='center', va='bottom', color='black', fontsize=20)
plt.title("Correlation between Feature-Score Correlations across Subjects")
plt.xlabel("Subject")
plt.ylabel("Corrs suds vs y-bocs")
plt.savefig("correlation_between_feature_score_corrs_across_subjects.pdf")
#plt.ylim(-1, 1)


num_subjects = len(df_final)
fig, axes = plt.subplots(nrows=1, ncols=7+1, figsize=(15, 5))
axes = axes.flatten()
for i, subject in enumerate(df_final["subject"]):
    df_subject = df_merged[df_merged["subject"] == subject]
    sns.regplot(x="correlation_rs", y="correlation_suds", data=df_subject, ax=axes[i], scatter_kws={"alpha":0.2})
    axes[i].set_title(f"{subject}\nr={df_final[df_final['subject'] == subject]['correlation_between_corrs'].values[0]:.2f} (p={df_final[df_final['subject'] == subject]['p_value'].values[0]:.3f})")
    if i == 0:
    #axes[i].set_xlabel("Correlation with score (Resting State)")
        axes[i].set_ylabel("Correlation with score (SUDS Task)")
    axes[i].set_xlabel("YBOCS corr")
    sns.despine()

# last one: average across subjects
df_avg = df_merged.groupby("feature").mean().reset_index()
sns.regplot(x="correlation_rs", y="correlation_suds", data=df_avg, ax=axes[-1], scatter_kws={"alpha":0.1}, color="red")
axes[-1].set_title(f"Average {df_final['correlation_between_corrs'].mean():.2f} (p={df_final['p_value'].mean():.3f})")
#axes[-1].set_xlabel("Correlation with score (Resting State)")
#axes[-1].set_ylabel("Correlation with score (SUDS Task)")
axes[-1].set_xlabel("YBOCS corr")
plt.tight_layout()
plt.savefig("correlation_between_feature_score_corrs_across_subjects_scatterplots.pdf")
plt.show()

plt.figure()
sns.histplot(df_merged["correlation_rs"], bins=50, kde=True)
plt.title("Distribution of Correlation between Feature-Score Correlations across Subjects")
plt.xlabel("Correlation between Feature-Score Correlations")
plt.ylabel("Count")
plt.show()

plt.figure()
sns.histplot(data=df_merged, x="correlation_rs", hue = "subject", bins=50, kde=True)
plt.title("Distribution of Correlation between Feature-Score Correlations across Subjects")
plt.xlabel("Correlation between Feature-Score Correlations")
plt.ylabel("Count")
plt.show()

df_merged["sig_suds_fwe"] = df_merged["p_value_suds"] < alpha / len(l_audio_features)

# make a barplot with significant correlations per subject
plt.figure()
plt.subplot(1, 2, 1)
sig_counts_rs = df_merged[df_merged["sig_rs_fwe"]].groupby("subject").size()
sig_counts_rs = sig_counts_rs.reindex(df_merged["subject"].unique(), fill_value=0)
sig_counts_rs.plot(kind="bar")
plt.title("Number of Significant Feature-Score Correlations (Resting State)")
plt.xlabel("Subject")
plt.ylabel("Count")
plt.subplot(1, 2, 2)
sig_counts_suds = l_audio_features[l_audio_features["sig_suds_fwe"]].groupby("subject").size()
sig_counts_suds = sig_counts_suds.reindex(df_merged["subject"].unique(), fill_value=0)
sig_counts_suds.plot(kind="bar", color="orange")
plt.title("Number of Significant Feature-Score Correlations (SUDS Task)")
plt.xlabel("Subject")
plt.ylabel("Count")
plt.tight_layout()
plt.show()
