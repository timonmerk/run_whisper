import os
import math
from posixpath import basename
from tracemalloc import start
import pandas as pd
import torch
import pyannote.audio.core.task
from torch.torch_version import TorchVersion
from omegaconf.listconfig import ListConfig
from omegaconf.dictconfig import DictConfig
import torchaudio
from moviepy import VideoFileClip
torch.serialization.add_safe_globals([TorchVersion, ListConfig, DictConfig])
torch.serialization.add_safe_globals([pyannote.audio.core.task.Specifications])
torch.serialization.add_safe_globals([pyannote.audio.core.task.Problem])
torch.serialization.safe_globals([pyannote.audio.core.task.Problem])
from tqdm import tqdm

import whisperx
from whisperx.diarize import DiarizationPipeline

PATH_FILES = "/Users/Timon/Downloads/vids_process"
outpath = "/Users/Timon/Downloads/audio_write"
WRITE_MP3 = False

if WRITE_MP3:
    
    list_videos = [f for f in os.listdir(PATH_FILES) if f.endswith('.MP4')]
    for video_file in list_videos:
        path_video = os.path.join(PATH_FILES, video_file)
        basename = os.path.splitext(os.path.basename(video_file))[0]
        video = VideoFileClip(path_video)
        audio_file = os.path.join(outpath, f"{basename}.mp3")
        video.audio.write_audiofile(audio_file, fps=16000, nbytes=2, codec="mp3", bitrate="192k", logger=None)

list_mp3_files = [f for f in os.listdir(outpath) if f.endswith('.mp3')]

for audio_file in tqdm(list_mp3_files):
        
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    #device = "mps" if torch.backends.mps.is_available() else device
    compute_type = "int8"
    SR = 16000


    def sec_to_hms(seconds: float) -> str:
        if pd.isna(seconds):
            return None
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = seconds % 60
        return f"{h:02d}:{m:02d}:{s:06.3f}"

    model = whisperx.load_model(
        "large-v2",
        device=device,
        compute_type=compute_type,
        vad_method="silero",
    )

    # Load audio once
    audio = whisperx.load_audio(os.path.join(outpath, audio_file)).astype("float32")
    #waveform, sr = torchaudio.load(os.path.join(PATH_FILES, audio_file))
    n_samples = len(audio)

    base = os.path.splitext(os.path.basename(audio_file))[0]


    result = model.transcribe(audio, batch_size=8, language="en")

    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    df_ = pd.DataFrame(result.get("segments", []))
    # delete 'words' column to save space
    if "words" in df_.columns:
        del df_["words"]

    out_csv = os.path.join(
        outpath,
        f"{base}.csv"
    )

    df_.to_csv(out_csv, index=False)
