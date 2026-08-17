
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
import seaborn as sns

l_audio_features = ["Loudness_sma3","alphaRatio_sma3","hammarbergIndex_sma3","slope0-500_sma3","slope500-1500_sma3","spectralFlux_sma3","mfcc1_sma3","mfcc2_sma3","mfcc3_sma3","mfcc4_sma3","F0semitoneFrom27.5Hz_sma3nz","jitterLocal_sma3nz","shimmerLocaldB_sma3nz","HNRdBACF_sma3nz","logRelF0-H1-H2_sma3nz","logRelF0-H1-A3_sma3nz","F1frequency_sma3nz","F1bandwidth_sma3nz","F1amplitudeLogRelF0_sma3nz","F2frequency_sma3nz","F2bandwidth_sma3nz","F2amplitudeLogRelF0_sma3nz","F3frequency_sma3nz","F3bandwidth_sma3nz","F3amplitudeLogRelF0_sma3nz","F0semitoneFrom27.5Hz_sma3nz_amean","F0semitoneFrom27.5Hz_sma3nz_stddevNorm","F0semitoneFrom27.5Hz_sma3nz_percentile20.0","F0semitoneFrom27.5Hz_sma3nz_percentile50.0","F0semitoneFrom27.5Hz_sma3nz_percentile80.0","F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2","F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope","F0semitoneFrom27.5Hz_sma3nz_stddevRisingSlope","F0semitoneFrom27.5Hz_sma3nz_meanFallingSlope","F0semitoneFrom27.5Hz_sma3nz_stddevFallingSlope","loudness_sma3_amean","loudness_sma3_stddevNorm","loudness_sma3_percentile20.0","loudness_sma3_percentile50.0","loudness_sma3_percentile80.0","loudness_sma3_pctlrange0-2","loudness_sma3_meanRisingSlope","loudness_sma3_stddevRisingSlope","loudness_sma3_meanFallingSlope","loudness_sma3_stddevFallingSlope","spectralFlux_sma3_amean","spectralFlux_sma3_stddevNorm","mfcc1_sma3_amean","mfcc1_sma3_stddevNorm","mfcc2_sma3_amean","mfcc2_sma3_stddevNorm","mfcc3_sma3_amean","mfcc3_sma3_stddevNorm","mfcc4_sma3_amean","mfcc4_sma3_stddevNorm","jitterLocal_sma3nz_amean","jitterLocal_sma3nz_stddevNorm","shimmerLocaldB_sma3nz_amean","shimmerLocaldB_sma3nz_stddevNorm","HNRdBACF_sma3nz_amean","HNRdBACF_sma3nz_stddevNorm","logRelF0-H1-H2_sma3nz_amean","logRelF0-H1-H2_sma3nz_stddevNorm","logRelF0-H1-A3_sma3nz_amean","logRelF0-H1-A3_sma3nz_stddevNorm","F1frequency_sma3nz_amean","F1frequency_sma3nz_stddevNorm","F1bandwidth_sma3nz_amean","F1bandwidth_sma3nz_stddevNorm","F1amplitudeLogRelF0_sma3nz_amean","F1amplitudeLogRelF0_sma3nz_stddevNorm","F2frequency_sma3nz_amean","F2frequency_sma3nz_stddevNorm","F2bandwidth_sma3nz_amean","F2bandwidth_sma3nz_stddevNorm","F2amplitudeLogRelF0_sma3nz_amean","F2amplitudeLogRelF0_sma3nz_stddevNorm","F3frequency_sma3nz_amean","F3frequency_sma3nz_stddevNorm","F3bandwidth_sma3nz_amean","F3bandwidth_sma3nz_stddevNorm","F3amplitudeLogRelF0_sma3nz_amean","F3amplitudeLogRelF0_sma3nz_stddevNorm","alphaRatioV_sma3nz_amean","alphaRatioV_sma3nz_stddevNorm","hammarbergIndexV_sma3nz_amean","hammarbergIndexV_sma3nz_stddevNorm","slopeV0-500_sma3nz_amean","slopeV0-500_sma3nz_stddevNorm","slopeV500-1500_sma3nz_amean","slopeV500-1500_sma3nz_stddevNorm","spectralFluxV_sma3nz_amean","spectralFluxV_sma3nz_stddevNorm","mfcc1V_sma3nz_amean","mfcc1V_sma3nz_stddevNorm","mfcc2V_sma3nz_amean","mfcc2V_sma3nz_stddevNorm","mfcc3V_sma3nz_amean","mfcc3V_sma3nz_stddevNorm","mfcc4V_sma3nz_amean","mfcc4V_sma3nz_stddevNorm","alphaRatioUV_sma3nz_amean","hammarbergIndexUV_sma3nz_amean","slopeUV0-500_sma3nz_amean","slopeUV500-1500_sma3nz_amean","spectralFluxUV_sma3nz_amean","loudnessPeaksPerSec","VoicedSegmentsPerSec","MeanVoicedSegmentLengthSec","StddevVoicedSegmentLengthSec","MeanUnvoicedSegmentLength","StddevUnvoicedSegmentLength","equivalentSoundLevel_dBp","arousal","dominance","valence"] 
l_video_features = ['AU_1', 'AU_2', 'AU_4', 'AU_5', 'AU_6', 'AU_7', 'AU_9', 'AU_10', 'AU_11', 'AU_12', 'AU_13', 'AU_14', 'AU_15', 'AU_16', 'AU_17', 'AU_18', 'AU_19', 'AU_20', 'AU_22', 'AU_23', 'AU_24', 'AU_25', 'AU_26', 'AU_27', 'AU_32', 'AU_38', 'AU_39', 'AU_L1', 'AU_R1', 'AU_L2', 'AU_R2', 'AU_L4', 'AU_R4', 'AU_L6', 'AU_R6', 'AU_L10', 'AU_R10', 'AU_L12', 'AU_R12', 'AU_L14', 'AU_R14']

# plot neural t-sne

plt.figure(figsize=(5, 6))
for color_score in [False, True]:
    for idx_j, RUN_RS in enumerate([False, True]):
        if RUN_RS:
            s = 15
        else:
            s = 5
        plt.subplot(2, 2, (0 if color_score is False else 1) * 2 + idx_j + 1) 
        if RUN_RS:
            df_audio_features_comb = pd.read_csv("/Users/Timon/Documents/Houston/whisper/audio_neural_features_combined_rs.csv")
            df_audio_features_comb = df_audio_features_comb.rename(columns={"YBOCS II Total Score": "score_orig"})
            df_audio_features_comb["subject"] = df_audio_features_comb["subject"].apply(lambda x: int(x[4:]))   
            df_audio_features_comb = df_audio_features_comb.rename(columns={c: c[4:] for c in df_audio_features_comb.columns if c.startswith("FAU_")})
        else:
            df_audio_features_comb = pd.read_csv("/Users/Timon/Documents/Houston/OCD_RCS/OCD_RCS/neural_audio_fau_combined.csv")
            # renmame 'score' column to score_orig
            df_audio_features_comb = df_audio_features_comb.rename(columns={"score": "score_orig"})
        df_audio_features_comb_neural = df_audio_features_comb[[c for c in df_audio_features_comb.columns if c.startswith("SC") and "psd" not in c and "coherence" not in c and "corr" not in c]]
        df_audio_features_comb_neural_mean = df_audio_features_comb_neural.mean(axis=0)
        df_audio_features_comb_neural_std = df_audio_features_comb_neural.std(axis=0)
        df_audio_features_comb_neural = (df_audio_features_comb_neural - df_audio_features_comb_neural_mean) / df_audio_features_comb_neural_std
        df_audio_features_comb_neural["subject"] = df_audio_features_comb["subject"]
        # z-score the score per subject
        df_audio_features_comb_neural["score"] = df_audio_features_comb["score_orig"]
        df_audio_features_comb_neural["score"] = df_audio_features_comb_neural.groupby("subject")["score"].transform(lambda x: (x - x.mean()) / x.std())

        # delete empty rows
        df_audio_features_comb_neural = df_audio_features_comb_neural.dropna(how='all')
        # run tsne on df_audio_features_comb
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
        df_tsne = df_audio_features_comb_neural.copy()
        # drop nan rows
        df_tsne = df_tsne.dropna()
        tsne_results = tsne.fit_transform(df_tsne.drop(columns=["score", "subject"], errors="ignore"))
        df_tsne["tsne-2d-one"] = tsne_results[:, 0]
        df_tsne["tsne-2d-two"] = tsne_results[:, 1]
        if color_score is False:
            c_ = pd.factorize(df_tsne["subject"])[0]
        else:
            c_ = df_tsne["score"]

        plt.scatter(df_tsne["tsne-2d-one"],
                    df_tsne["tsne-2d-two"], c=c_,
                    cmap='Accent' if color_score is False else 'viridis', alpha=0.7, s=s)

        if color_score is False:
            subject_labels = df_tsne["subject"].unique()
            cbar = plt.colorbar(ticks=range(len(subject_labels)))
            cbar.ax.set_yticklabels(subject_labels)
        else:
            plt.colorbar(label='score')

        plt.title("" + ("Resting State" if RUN_RS else "Suds"))
        sns.despine()
        plt.xlabel("TSNE Dim. 1")
        plt.ylabel("TSNE Dim. 2")

plt.tight_layout()
plt.savefig(f"tsne_neural.pdf")

    



plt.figure(figsize=(5, 6))
for idx_i, AUDIO in enumerate([False, True]):
    for idx_j, RUN_RS in enumerate([False, True]):
        if RUN_RS:
            df_audio_features_comb = pd.read_csv("/Users/Timon/Documents/Houston/whisper/audio_neural_features_combined_rs.csv")
            df_audio_features_comb = df_audio_features_comb.rename(columns={"YBOCS II Total Score": "score"})
            df_audio_features_comb["subject"] = df_audio_features_comb["subject"].apply(lambda x: int(x[4:]))   
            df_audio_features_comb = df_audio_features_comb.rename(columns={c: c[4:] for c in df_audio_features_comb.columns if c.startswith("FAU_")})

        else:
            df_audio_features_comb = pd.read_csv("/Users/Timon/Documents/Houston/OCD_RCS/OCD_RCS/neural_audio_fau_combined.csv")

        if AUDIO:
            df_audio_zs = df_audio_features_comb.copy()[l_audio_features]
        else:
            df_audio_zs = df_audio_features_comb.copy()[l_video_features]

        df_audio_zs_mean = df_audio_zs.mean(axis=0)
        df_audio_zs_std = df_audio_zs.std(axis=0)
        df_audio_zs = (df_audio_zs - df_audio_zs_mean) / df_audio_zs_std
        df_audio_zs["subject"] = df_audio_features_comb["subject"]
        # delete empty rows
        df_audio_zs = df_audio_zs.dropna(how='all')

        # run tsne on df_audio_features_comb

        tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
        df_tsne = df_audio_zs.copy()

        # drop nan rows
        df_tsne = df_tsne.dropna()
        tsne_results = tsne.fit_transform(df_tsne.drop(columns=["score", "subject"], errors="ignore"))
        df_tsne["tsne-2d-one"] = tsne_results[:, 0]
        df_tsne["tsne-2d-two"] = tsne_results[:, 1]
        plt.subplot(2, 2, idx_i * 2 + idx_j + 1)
        if RUN_RS:
            s = 15
        else:
            s = 5
        plt.scatter(df_tsne["tsne-2d-one"],
                    df_tsne["tsne-2d-two"], c=pd.factorize(df_tsne["subject"])[0],
                    cmap='Accent', alpha=0.7, s=s)
        #plt.colorbar(label='Subject')
        # add the subject names to the colorbar
        cbar = plt.colorbar(ticks=range(len(df_tsne["subject"].unique())))
        cbar.ax.set_yticklabels(df_tsne["subject"].unique())
        plt.title("" + ("Audio" if AUDIO else "Video") + " - " + ("Resting State" if RUN_RS else "Suds"))
        sns.despine()
        plt.xlabel("TSNE Dim. 1")
        plt.ylabel("TSNE Dim. 2")
plt.tight_layout()
plt.savefig(f"tsne_all.pdf", bbox_inches='tight')