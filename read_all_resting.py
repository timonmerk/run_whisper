PATH_READ = "/Users/Timon/Library/CloudStorage/Box-Box/APL_BCM_Share_SUDS/Audio_Analysis/WhisperX_restingstate/manual_annot"

import os
import pandas as pd

files = os.listdir(PATH_READ)

files_all = [f for f in files if f.endswith('.csv')]
df = []

for f in files_all:
    df_read = pd.read_csv(os.path.join(PATH_READ, f))
    sub = f.split('_')[0]
    date = f.split('_')[1]
    df_read["subject"] = sub
    df_read["date"] = date
    df.append(df_read)
df_audio_features_comb = pd.concat(df, ignore_index=True)
df_audio_features_comb["speaker"] = 0
df_audio_features_comb = df_audio_features_comb[["subject", "date", "start", "end", "speaker", "text"]]
# sort by subject state and start
df_audio_features_comb = df_audio_features_comb.sort_values(by=["subject", "date", "start"]).reset_index(drop=True)
df_audio_features_comb[["subject", "date", "start", "end", "speaker", "text"]].to_csv("restingstate_all_subjects.csv", index=False)