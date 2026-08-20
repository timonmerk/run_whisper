import pandas as pd
import os
import math
import librosa
import opensmile
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor
from wav2vec_model import EmotionModel
from tqdm import tqdm
import re

def run_audio_analysis(
    filepath,
    df_diarization: pd.DataFrame,
    save_folder,
    max_chunk_duration=5.0,
):

    # Split target speaker utterances into fixed-length chunks
    #speaker_chunks = split_speaker_segments(diarization_df, target_speaker, min_chunk_duration=min_chunk_duration, max_chunk_duration=max_chunk_duration)
    #print(f"Found {len(speaker_chunks)} chunks for {target_speaker}")

    # Load audio
    sig, sr = librosa.load(filepath, sr=None, mono=True)
    # make sure to resample to 16kHz
    if sr != 16000:
        sig = librosa.resample(sig, orig_sr=sr, target_sr=16000)
        sr = 16000

    # Initialize extractors once
    smile_func = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals
    )
    smile_lld = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors
    )
    device = 'mps' # 'cpu'
    model_name = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = EmotionModel.from_pretrained(model_name)
    model.to(device)

    def process_func(x, sr, embeddings=False):
        y = processor(x, sampling_rate=sr)['input_values'][0]
        y = torch.from_numpy(y).to(device)
        with torch.no_grad():
            y = model(y)[0 if embeddings else 1]
        return y.detach().cpu().numpy()

    # Extract features chunk-wise
    results = []
    for i, (start, end) in enumerate(zip(df_diarization['start'], df_diarization['end'])):
        #print(f"Processing chunk {i} of {len(speaker_chunks)} from {start} to {end}")
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        chunk_sig = sig[start_sample:end_sample]

        # play chunk_sig
        #import sounddevice as sd
        #sd.play(chunk_sig, sr)

        # from matplotlib import pyplot as plt
        # plt.figure(figsize=(10, 2))
        # plt.plot(chunk_sig)
        # plt.title(df_diarization.iloc[i]['text'])
        # plt.savefig("example_timetrace.pdf")


        # OpenSMILE LLDs
        lld_df = smile_lld.process_signal(chunk_sig, sampling_rate=sr).reset_index()
        lld_df.rename(columns={"start": "start_time", "end": "end_time"}, inplace=True)
        lld_df["start_time"] = lld_df["start_time"].dt.total_seconds() + start
        lld_df["end_time"] = lld_df["end_time"].dt.total_seconds() + start
        lld_df["center_time"] = (lld_df["start_time"] + lld_df["end_time"]) / 2

        # OpenSMILE functionals
        func = smile_func.process_signal(chunk_sig, sampling_rate=sr)
        func["center_time"] = (start + end) / 2
        func_df = func.reset_index(drop=True)

        # Merge features
        combined = pd.merge_asof(
            lld_df.sort_values("center_time"),
            func_df.sort_values("center_time"),
            on="center_time",
            direction="nearest",
            tolerance=max_chunk_duration / 2
        )

        # Wav2Vec2 features
        chunk_tensor = torch.from_numpy(np.expand_dims(chunk_sig, axis=0))
        vad_preds = process_func(chunk_tensor, sr, embeddings=False)
        vad_embed = process_func(chunk_tensor, sr, embeddings=True)
        combined[['arousal', 'dominance', 'valence']] = pd.Series(vad_preds[0])
        dim_df = pd.DataFrame([vad_embed[0]], columns=[f"Dim {i}" for i in range(vad_embed.shape[1])])
        dim_df = pd.concat([dim_df] * len(combined), ignore_index=True)
        combined = pd.concat([combined.reset_index(drop=True), dim_df], axis=1)

        combined_m = combined.mean(axis=0).to_frame().T
        # add columns from df_diarization.iloc[i] to combined_m
        for col in df_diarization.columns:
            combined_m[col] = df_diarization.iloc[i][col]
            # don't really know where I got the score in the diarzation df..
        results.append(combined_m)

    all_features = pd.concat(results, axis=0, ignore_index=True)
    all_features = all_features.drop(columns=["center_time"], errors="ignore")

    # Save file
    all_features.to_csv(out_path, index=False)



PATH_FILES = "/Users/Timon/Library/CloudStorage/Box-Box/missing-ocd-videos"
folders = os.listdir(PATH_FILES)
# in each folder is one mp3 file, get the list of mp3 files
list_mp3_files = []
list_df_diarizations = []
for folder in folders:
    folder_path = os.path.join(PATH_FILES, folder)
    if os.path.isdir(folder_path):
        for file in os.listdir(folder_path):
            if file.endswith('.mp3'):
                list_mp3_files.append(os.path.join(folder_path, file))
            if file.endswith('.csv'):
                df_read = pd.read_csv(os.path.join(folder_path, file), delimiter=";")
                df_read["file"] = file
                subject = folder.split('_')[0]
                date = folder.split('_')[1]
                df_read["subject"] = subject
                df_read["date"] = date
                df_read["folder"] = folder
                list_df_diarizations.append(df_read)

df = pd.concat(list_df_diarizations, ignore_index=True)

for file in tqdm(df['file'].unique()):
    df_dia_f = df[df['file'] == file]
    sub = df_dia_f['subject'].iloc[0]
    print(f"Processing file: {file}")

    out_path = os.path.join(PATH_FILES, df_dia_f['folder'].iloc[0])
    out_path = os.path.join(out_path, file[:-4] + "_audio_features.csv")
    df_dia_f["duration"] = df_dia_f["end"] - df_dia_f["start"]
    df_dia_f = df_dia_f[df_dia_f["duration"] >= 2]
    df_dia_f = df_dia_f.query("SPEAKER_ID == 1")
    filepath = os.path.join(PATH_FILES, df_dia_f['folder'].iloc[0], file[:-4] + ".mp3")
    run_audio_analysis(filepath, df_dia_f, out_path, max_chunk_duration=10.0)
