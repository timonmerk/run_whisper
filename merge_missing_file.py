import pandas as pd

# read the csv file 
# PATH_READ = "/Users/Timon/Documents/Houston/OCD_RCS/OCD_RCS/speaker_diarization/annotations_patients_incl_box_files_merged_final.csv"
# df = pd.read_csv(PATH_READ, sep=';')
# df.query("speaker == 1").to_csv("/Users/Timon/Documents/Houston/OCD_RCS/OCD_RCS/speaker_diarization/annotations_patients_incl_box_files_merged_final_speaker1.csv", index=False)

df_pre = pd.read_csv("/Users/Timon/Documents/Houston/OCD_RCS/OCD_RCS/speaker_diarization/annotations_patients_incl_box_files.csv", sep=';')
df_new = pd.read_csv("/Users/Timon/Documents/Houston/OCD_RCS/OCD_RCS/speaker_diarization/missing_annots_11.csv", sep=';')

# remove Unnamed: 0 column from df_new
df_new = df_new.drop(columns=['Unnamed: 0'], errors='ignore')

# start end
# convert start end from min:sec to float seconds in df_new
#df_new[['start', 'end']] = df_new[['start', 'end']].apply(lambda x: x.str.split(':').apply(lambda y: float(y[0]) * 60 + float(y[1])))
# rename start_dt to start an 
df_new["file"] = None

# drop feature_time
df_new = df_new.drop(columns=['feature_time'], errors='ignore')
# rename columns 'start_dt' to 'time'
df_new = df_new.rename(columns={'start_dt': 'time'})

# create a colum time_min_sec in df_new, which is time in min:sec format
df_new['time_min_sec'] = df_new['start'].apply(lambda x: f"{int(x//60)}:{int(x%60):02d}")
# drop end_dt from df_new
df_new = df_new.drop(columns=['end_dt'], errors='ignore')

# concat the two dataframes
df_merged = pd.concat([df_pre, df_new], axis=0, ignore_index=True)
# drop Unnamed: 0 column from df_merged
df_merged = df_merged.drop(columns=['Unnamed: 0'], errors='ignore')

# save to csv
df_merged.to_csv("/Users/Timon/Documents/Houston/OCD_RCS/OCD_RCS/speaker_diarization/annotations_patients_incl_box_files_merged.csv", index=False)