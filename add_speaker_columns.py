import pandas as pd
import os

path_ = "/Users/Timon/Downloads/audio_write/"

# read each csv file, add a column "SPEAKER_ID", fill with 0, and save
for filename in os.listdir(path_):
    if filename.endswith(".csv"):
        df = pd.read_csv(os.path.join(path_, filename))
        df["SPEAKER_ID"] = 0
        df.to_csv(os.path.join(path_, filename), index=False)

