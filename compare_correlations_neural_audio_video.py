import pandas as pd
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
l_video_features = [
    "AU_1", "AU_2", "AU_4", "AU_5", "AU_6", "AU_7", "AU_9", "AU_10",
    "AU_11", "AU_12", "AU_13", "AU_14", "AU_15", "AU_16", "AU_17",
    "AU_18", "AU_19", "AU_20", "AU_22", "AU_23", "AU_24", "AU_25",
    "AU_26", "AU_27", "AU_32", "AU_38", "AU_39",
    "AU_L1", "AU_R1", "AU_L2", "AU_R2", "AU_L4", "AU_R4",
    "AU_L6", "AU_R6", "AU_L10", "AU_R10",
    "AU_L12", "AU_R12", "AU_L14", "AU_R14"
]

df_ybocs = pd.read_csv("/Users/Timon/Documents/Houston/OCD_RCS/OCD_RCS/feature_region_subject_correlations_ybocs.csv")
df_suds = pd.read_csv("/Users/Timon/Documents/Houston/OCD_RCS/OCD_RCS/feature_region_subject_correlations_suds.csv")

df_ybocs["corr_abs"] = np.abs(df_ybocs["correlation"])
df_suds["corr_abs"] = np.abs(df_suds["correlation"])

PATH_SUDS = "/Users/Timon/Documents/Houston/OCD_RCS/OCD_RCS/neural_audio_fau_combined.csv"
PATH_RS = "/Users/Timon/Documents/Houston/whisper/audio_neural_features_combined_rs.csv"


if READ_RS:
    # rename ''YBOCS II Total Score' to 'score'
    df_audio_features_comb = pd.read_csv(PATH_RS)
    df_audio_features_comb = df_audio_features_comb.rename(columns={"YBOCS II Total Score": "score"})
    df_audio_features_comb["subject"] = df_audio_features_comb["subject"].apply(lambda x: int(x[4:]))
    df_audio_features_comb = df_audio_features_comb.rename(columns={c: c[4:] for c in df_audio_features_comb.columns if c.startswith("FAU_")})
else:
    # rename score_
    df_audio_features_comb = pd.read_csv(PATH_SUDS)
corrs_p_val = []
for sub in df_audio_features_comb["subject"].unique():
    df_sub = df_audio_features_comb[df_audio_features_comb["subject"] == sub]
    for col in df_sub.columns:
        if col not in l_audio_features and col not in l_video_features:
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
            "subject": sub,
            "type" : "AUDIO" if col in l_audio_features else "VIDEO"
        })

df_corrs_p_val = pd.DataFrame(corrs_p_val)
df_corrs_p_val["corr_abs"] = np.abs(df_corrs_p_val["corr"])

mean_corrs_ = []
plt.figure(figsize=(15, 5))
for sub_idx, sub in enumerate(df_audio_features_comb["subject"].unique()):
    plt.subplot(1, len(df_audio_features_comb["subject"].unique()), sub_idx + 1)
    df_behav = df_corrs_p_val[df_corrs_p_val["subject"] == sub]
    df_behav["type"] = "behav"
    if READ_RS:
        df_neural = df_ybocs[df_ybocs["subject"] == sub]
    else:
        df_neural = df_suds[df_suds["subject"] == sub]
    df_neural["type"] = "neural"

    df_hist = pd.concat([df_behav[["corr_abs", "type"]], df_neural[["corr_abs", "type"]]], axis=0).reset_index()

    sns.histplot(data = df_hist, x="corr_abs", hue="type", bins=30, stat="density", common_norm=False,
                kde=False, palette="viridis")
    sns.despine()
    plt.title(f"Subject {sub}")
    means_ = df_hist.groupby("type")["corr_abs"].mean()
    quantile_75 = df_hist.groupby("type")["corr_abs"].quantile(0.75)
    quantile_50 = df_hist.groupby("type")["corr_abs"].quantile(0.50)
    mean_ = df_hist.groupby("type")["corr_abs"].mean()
    median_ = df_hist.groupby("type")["corr_abs"].median()
    std_ = df_hist.groupby("type")["corr_abs"].std()
    quantile_25 = df_hist.groupby("type")["corr_abs"].quantile(0.25)
    quantile_90 = df_hist.groupby("type")["corr_abs"].quantile(0.90)
    quantile_max = df_hist.groupby("type")["corr_abs"].max()


    mean_corrs_.append({
        "subject": sub,
        "mean_corr_behav": means_["behav"],
        "mean_corr_neural": means_["neural"],
        "quantile_75_behav": quantile_75["behav"],
        "quantile_75_neural": quantile_75["neural"],
        "quantile_50_behav": quantile_50["behav"],
        "quantile_50_neural": quantile_50["neural"],
        "quantile_25_behav": quantile_25["behav"],
        "quantile_25_neural": quantile_25["neural"],
        "quantile_90_behav": quantile_90["behav"],
        "quantile_90_neural": quantile_90["neural"],
        "quantile_max_behav": quantile_max["behav"],
        "quantile_max_neural": quantile_max["neural"],
        "mean_behav": mean_["behav"],
        "mean_neural": mean_["neural"],
        "median_behav": median_["behav"],
        "median_neural": median_["neural"],
        "std_behav": std_["behav"],
        "std_neural": std_["neural"],
    })

df_mean_corrs = pd.DataFrame(mean_corrs_)
df_mean_corrs.to_csv(f"mean_correlations_neural_audio_video_RS_{READ_RS}.csv", index=False)

plt.tight_layout()
plt.savefig(f"compare_neural_audio_video_corrs_hist_RS_{READ_RS}.pdf")

corrs_ = []
for sub in df_audio_features_comb["subject"].unique():
    if READ_RS:
        best_neural_corr = df_ybocs[df_ybocs["subject"] == sub]["corr_abs"].max()
    else:
        best_neural_corr = df_suds[df_suds["subject"] == sub]["corr_abs"].max()
    best_audio_corr = df_corrs_p_val[(df_corrs_p_val["subject"] == sub) & (df_corrs_p_val["type"] == "AUDIO")]["corr_abs"].max()
    best_video_corr = df_corrs_p_val[(df_corrs_p_val["subject"] == sub) & (df_corrs_p_val["type"] == "VIDEO")]["corr_abs"].max()
    for feat_type, best_corr in zip(["NEURAL", "AUDIO", "VIDEO"], [best_neural_corr, best_audio_corr, best_video_corr]):
        corrs_.append({
            "subject": sub,
            "feature_type": feat_type,
            "best_corr_abs": best_corr
        })
df_best_corrs = pd.DataFrame(corrs_)

plt.figure()
sns.boxplot(data=df_best_corrs, x="feature_type", y="best_corr_abs", showmeans=True, showfliers=False)
sns.swarmplot(data=df_best_corrs, x="feature_type", y="best_corr_abs", color=".25")
plt.title("correlations RS: " + str(READ_RS))
plt.savefig(f"compare_neural_audio_video_best_corrs_RS_{READ_RS}.pdf")
plt.show()