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

READ_RS = False
l_audio_features = ["Loudness_sma3","alphaRatio_sma3","hammarbergIndex_sma3","slope0-500_sma3","slope500-1500_sma3","spectralFlux_sma3","mfcc1_sma3","mfcc2_sma3","mfcc3_sma3","mfcc4_sma3","F0semitoneFrom27.5Hz_sma3nz","jitterLocal_sma3nz","shimmerLocaldB_sma3nz","HNRdBACF_sma3nz","logRelF0-H1-H2_sma3nz","logRelF0-H1-A3_sma3nz","F1frequency_sma3nz","F1bandwidth_sma3nz","F1amplitudeLogRelF0_sma3nz","F2frequency_sma3nz","F2bandwidth_sma3nz","F2amplitudeLogRelF0_sma3nz","F3frequency_sma3nz","F3bandwidth_sma3nz","F3amplitudeLogRelF0_sma3nz","F0semitoneFrom27.5Hz_sma3nz_amean","F0semitoneFrom27.5Hz_sma3nz_stddevNorm","F0semitoneFrom27.5Hz_sma3nz_percentile20.0","F0semitoneFrom27.5Hz_sma3nz_percentile50.0","F0semitoneFrom27.5Hz_sma3nz_percentile80.0","F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2","F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope","F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope","F0semitoneFrom27.5Hz_sma3nz_meanFallingSlope","F0semitoneFrom27.5Hz_sma3nz_stddevFallingSlope","loudness_sma3_amean","loudness_sma3_stddevNorm","loudness_sma3_percentile20.0","loudness_sma3_percentile50.0","loudness_sma3_percentile80.0","loudness_sma3_pctlrange0-2","loudness_sma3_meanRisingSlope","loudness_sma3_stddevRisingSlope","loudness_sma3_meanFallingSlope","loudness_sma3_stddevFallingSlope","spectralFlux_sma3_amean","spectralFlux_sma3_stddevNorm","mfcc1_sma3_amean","mfcc1_sma3_stddevNorm","mfcc2_sma3_amean","mfcc2_sma3_stddevNorm","mfcc3_sma3_amean","mfcc3_sma3_stddevNorm","mfcc4_sma3_amean","mfcc4_sma3_stddevNorm","jitterLocal_sma3nz_amean","jitterLocal_sma3nz_stddevNorm","shimmerLocaldB_sma3nz_amean","shimmerLocaldB_sma3nz_stddevNorm","HNRdBACF_sma3nz_amean","HNRdBACF_sma3nz_stddevNorm","logRelF0-H1-H2_sma3nz_amean","logRelF0-H1-H2_sma3nz_stddevNorm","logRelF0-H1-A3_sma3nz_amean","logRelF0-H1-A3_sma3nz_stddevNorm","F1frequency_sma3nz_amean","F1frequency_sma3nz_stddevNorm","F1bandwidth_sma3nz_amean","F1bandwidth_sma3nz_stddevNorm","F1amplitudeLogRelF0_sma3nz_amean","F1amplitudeLogRelF0_sma3nz_stddevNorm","F2frequency_sma3nz_amean","F2frequency_sma3nz_stddevNorm","F2bandwidth_sma3nz_amean","F2bandwidth_sma3nz_stddevNorm","F2amplitudeLogRelF0_sma3nz_amean","F2amplitudeLogRelF0_sma3nz_stddevNorm","F3frequency_sma3nz_amean","F3frequency_sma3nz_stddevNorm","F3bandwidth_sma3nz_amean","F3bandwidth_sma3nz_stddevNorm","F3amplitudeLogRelF0_sma3nz_amean","F3amplitudeLogRelF0_sma3nz_stddevNorm","alphaRatioV_sma3nz_amean","alphaRatioV_sma3nz_stddevNorm","hammarbergIndexV_sma3nz_amean","hammarbergIndexV_sma3nz_stddevNorm","slopeV0-500_sma3nz_amean","slopeV0-500_sma3nz_stddevNorm","slopeV500-1500_sma3nz_amean","slopeV500-1500_sma3nz_stddevNorm","spectralFluxV_sma3nz_amean","spectralFluxV_sma3nz_stddevNorm","mfcc1V_sma3nz_amean","mfcc1V_sma3nz_stddevNorm","mfcc2V_sma3nz_amean","mfcc2V_sma3nz_stddevNorm","mfcc3V_sma3nz_amean","mfcc3V_sma3nz_stddevNorm","mfcc4V_sma3nz_amean","mfcc4V_sma3nz_stddevNorm","alphaRatioUV_sma3nz_amean","hammarbergIndexUV_sma3nz_amean","slopeUV0-500_sma3nz_amean","slopeUV500-1500_sma3nz_amean","spectralFluxUV_sma3nz_amean","loudnessPeaksPerSec","VoicedSegmentsPerSec","MeanVoicedSegmentLengthSec","StddevVoicedSegmentLengthSec","MeanUnvoicedSegmentLength","StddevUnvoicedSegmentLength","equivalentSoundLevel_dBp","arousal","dominance","valence"] + [f"Dim {i}" for i in range(1024)] + ["duration"]


if READ_RS:
    df_audio_features_comb = pd.read_csv("audio_neural_features_combined_rs.csv")
    # rename ''YBOCS II Total Score' to 'score'
    df_audio_features_comb = df_audio_features_comb.rename(columns={"YBOCS II Total Score": "score"})
    df_audio_features_comb["subject"] = df_audio_features_comb["subject"].apply(lambda x: int(x[4:]))
else:   
    df_audio_features_comb = pd.read_csv("audio_neural_features_combined.csv")


corrs_p_val = []
for sub in df_audio_features_comb["subject"].unique():
    df_sub = df_audio_features_comb[df_audio_features_comb["subject"] == sub]
    for col in df_sub.columns:
        if col not in l_audio_features:
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
# SUDS: num correlations p<0.01 shuffled: 131, non-shuffled 402
# RS: num correlations p<0.01 shuffled: 90, non-shuffled 544

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
df_corrs_p_val["significant"] = df_corrs_p_val["p_val"] < 0.05/len(l_audio_features)
df_counts = df_corrs_p_val.query("significant == True").groupby(["subject", "shuffled"]).size().reset_index(name='counts')
sns.barplot(data=df_counts, x="subject", y="counts", hue="shuffled", palette="viridis")
plt.title("p < Bonf. corr.")
plt.xlabel("Subject")
plt.ylabel("Count")
plt.suptitle(f"RS={READ_RS}")
plt.tight_layout()

plt.savefig(f"audio_correlations_RS_{READ_RS}_incl_r.pdf")


# 