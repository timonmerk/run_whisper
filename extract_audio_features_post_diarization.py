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

def split_speaker_segments(diarization_df, target_speaker, min_chunk_duration=1.0, max_chunk_duration=5.0):
    """Split speaker segments into chunks between min and max duration."""
    chunks = []
    for _, row in diarization_df.iterrows():
        if row['speaker'] != target_speaker:
            continue
        start, end = row['start'], row['end']
        duration = end - start

        # Case 1: segment is shorter than max duration
        if duration <= max_chunk_duration:
            if duration >= min_chunk_duration:
                chunks.append((start, end))
            continue

        # Case 2: segment is longer — split into multiple max-length chunks
        current = start
        while current + max_chunk_duration < end:
            chunks.append((current, current + max_chunk_duration))
            current += max_chunk_duration

        # Handle remainder chunk (only if ≥ min_chunk_duration)
        if end - current >= min_chunk_duration:
            chunks.append((current, end))

    return chunks

def run_audio_analysis(
    filepath,
    diarization_path,
    target_speaker,
    save_folder,
    min_chunk_duration=1.0,
    max_chunk_duration=5.0,
):
    # Load diarization CSV
    diarization_df = pd.read_csv(diarization_path)
    diarization_df['speaker'] = diarization_df['speaker'].str.strip()

    # Split target speaker utterances into fixed-length chunks
    speaker_chunks = split_speaker_segments(diarization_df, target_speaker, min_chunk_duration=min_chunk_duration, max_chunk_duration=max_chunk_duration)
    print(f"Found {len(speaker_chunks)} chunks for {target_speaker}")

    # Load audio
    sig, sr = librosa.load(filepath, sr=None, mono=True)

    # Initialize extractors once
    smile_func = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals
    )
    smile_lld = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors
    )
    device = 'cpu'
    model_name = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = EmotionModel.from_pretrained(model_name)

    def process_func(x, sr, embeddings=False):
        y = processor(x, sampling_rate=sr)['input_values'][0]
        y = torch.from_numpy(y).to(device)
        with torch.no_grad():
            y = model(y)[0 if embeddings else 1]
        return y.detach().cpu().numpy()

    # Extract features chunk-wise
    results = []
    for i, (start, end) in enumerate(speaker_chunks):
        print(f"Processing chunk {i} of {len(speaker_chunks)} from {start} to {end}")
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        chunk_sig = sig[start_sample:end_sample]

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

        results.append(combined)

    all_features = pd.concat(results, axis=0, ignore_index=True)
    all_features = all_features.drop(columns=["center_time"], errors="ignore")

    # Save file
    filename = os.path.basename(filepath)
    features_save_parquet = os.path.join(save_folder, filename.replace(".wav", "opensmile_vad_patient_only.parquet"))
    all_features.to_parquet(features_save_parquet, compression="snappy")
    print(f"Saved speaker-specific features to: {features_save_parquet}")

if __name__ == "__main__":
    filepath = "/projects/AIFMH/GENIUS/data/OCD_SUDS/DBS_004/DBS004_9.14.2020.wav"
    diarization_path = "/projects/AIFMH/GENIUS/data/OCD_SUDS/audio_outputs/whisperx/DBS004_9.14.2020_whisperx_utterance.csv"
    target_speaker = "SPEAKER_00"
    save_folder = "/projects/AIFMH/GENIUS/data/OCD_SUDS/audio_outputs/"
    min_chunk_duration = 1.0
    max_chunk_duration = 5.0
    run_audio_analysis(filepath, diarization_path, target_speaker, save_folder, min_chunk_duration, max_chunk_duration)
