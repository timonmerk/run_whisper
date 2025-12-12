import pandas as pd
import os

PATH_ = "/Users/Timon/Library/CloudStorage/Box-Box/APL_BCM_Share_SUDS/Resting-State-Audios"

files_wav = [os.path.join(PATH_, f) for f in os.listdir(PATH_) if f.endswith('.wav')]
dates_wav = [os.path.basename(f).split('_')[1] for f in files_wav]
subs_wav = [int(os.path.basename(f).split('_')[0]) for f in files_wav]
vid_name_wav = [os.path.basename(f).split('_')[2][:-4] for f in files_wav]
df_wav = pd.DataFrame({
    'file': files_wav,
    'subject': subs_wav,
    'date': dates_wav,
    'video_name': vid_name_wav
})

df = pd.read_csv("restingstate_all_subjects.csv", delimiter=";")
df_subs = df['subject'].unique()
df_dates = df['date'].unique()
df["file"] = ""

for i, sub in enumerate(df_subs):
    df_sub = df[df['subject'] == sub]
    dates_sub = df_sub['date'].unique()
    for date in dates_sub:
        df_wav_ = df_wav[(df_wav['subject'] == sub) & (df_wav['date'] == date)]
        # add filepath column to df_sub
        if len(df_wav_) > 0:
            df.loc[(df['subject'] == sub) & (df['date'] == date), 'file'] = df_wav_['file'].iloc[0]
        else:
            print(f"No wav file found for subject {sub} on date {date}")
df.to_csv("restingstate_all_subjects_with_filepaths.csv", index=False)