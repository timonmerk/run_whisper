import pandas as pd

annotations_df = pd.read_csv('/Users/Timon/Documents/Houston/OCD_RCS/OCD_RCS/speaker_diarization/annotations_patients.csv', sep=';')

# wo sind die wav files?
PATH_AUDIO = '/Users/Timon/Library/CloudStorage/Box-Box/aDBS ERP Master Folder'
# read all m4a or wav files in the folder and its subfolders
import glob
import os
files = sorted(glob.glob(PATH_AUDIO + '/**/**/**/*.m4a', recursive=True)) + sorted(glob.glob(PATH_AUDIO + '/**/**/**/*.wav', recursive=True))
file_names = [os.path.basename(f) for f in files]
# replace spaces with _
file_names = [f.replace(' ', '_') for f in file_names]
fclean = [file[7:-4] for file in file_names]
# delete underscores
fclean = [file.replace('_', '') for file in fclean]
# try to infer the datetime from each filename
df_files = pd.DataFrame({'file': files, 'file_name': file_names, 'fclean': fclean})
df_files['date'] = pd.to_datetime(df_files['fclean'], errors='coerce', format="mixed")
sub_names = ["DBS004", "DBS005", "DBS006", "DBS007", "DBS008", "DBS009", "DBS010", "DBS011", "DBS012"]
subs = []
for f in df_files['file']:
    sub = None
    for s in sub_names:
        if s in f:
            sub = s
            break
    if sub is None:
        subs.append(None)
    else:
        subs.append(int(sub[-3:]))
df_files['sub'] = subs

# iterate through annotations_df and match the sub and date column
files_match = []
for i, row in annotations_df.iterrows():
    sub = row['sub']
    date = pd.to_datetime(row['date'], errors='coerce', format="mixed")
    # find the file in df_files that matches the sub and date
    file_match = df_files[(df_files['sub'] == sub) & (df_files['date'] == date)]
    if len(file_match) == 0:
        files_match.append(None)
    else:
        files_match.append(file_match.iloc[0]['file'])
    
annotations_df['file'] = files_match
annotations_df.to_csv('/Users/Timon/Documents/Houston/OCD_RCS/OCD_RCS/speaker_diarization/annotations_patients_incl_box_files.csv', index=False)
# had then to add two manually by hand

