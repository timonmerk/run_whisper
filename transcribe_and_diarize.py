import os
import sys
import glob
import librosa
import whisperx
import pandas as pd

def run_whisperx(audio_file, save_folder):

    device = "cpu"
    batch_size = 8 # reduce if low on GPU mem
    compute_type = "int8" # change to "int8" if low on GPU mem (may reduce accuracy)

    # 1. Transcribe with original whisper (batched)
    model = whisperx.load_model("large-v2", device, compute_type=compute_type)

    audio, sr = librosa.load(audio_file, sr=16000, mono=True)  # Force mono + 16kHz
    audio = audio.astype("float32")
    result = model.transcribe(audio, batch_size=batch_size)
    print(result["segments"]) # before alignment

    # 2. Align whisper output
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    print(result["segments"]) # after alignment

    # delete model if low on GPU resources
    # import gc; import torch; gc.collect(); torch.cuda.empty_cache(); del model_a

    # 3. Assign speaker labels
    diarize_model = whisperx.diarize.DiarizationPipeline(use_auth_token="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", device=device)

    # add min/max number of speakers if known
    diarize_segments = diarize_model(audio, min_speakers=2, max_speakers=2)

    result = whisperx.assign_word_speakers(diarize_segments, result)
    print(diarize_segments)
    print(result["segments"]) # segments are now assigned speaker IDs

    # Store utterance level outputs
    utterances = pd.DataFrame([
        {
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"],
            "speaker": seg.get("speaker", None),
            "score": seg.get("score", None)
        }
        for seg in result["segments"]
    ])

    # Store word level outputs
    word_rows = []
    for seg in result["segments"]:
        speaker = seg.get("speaker", None)
        for word in seg.get("words", []):
            word_rows.append({
                "start": word["start"],
                "end": word["end"],
                "text": word["word"],
                "score": word.get("score", None),
                "speaker": speaker
            })

    words = pd.DataFrame(word_rows)

    # Save outputs to CSV
    filename = os.path.basename(audio_file)
    utterance_save_csv = os.path.join(save_folder,filename.replace(".wav", "_whisperx_utterance.csv"))
    print(f"\nSaving utterance level dataframe to: {utterance_save_csv}")
    utterances.to_csv(utterance_save_csv, index=False)
    print(utterances)

    word_save_csv = os.path.join(save_folder,filename.replace(".wav", "_whisperx_word.csv"))
    print(f"\nSaving word level dataframe to: {word_save_csv}")
    words.to_csv(word_save_csv, index=False)
    print(words)


if __name__ == "__main__":

    # Get job id
    job_id = int(sys.argv[1])

    # Get all .wav files
    root = "/projects/AIFMH/PIIPHI_Restricted/OCD_SUDS"
    files = sorted(glob.glob(root + '/**/*.wav', recursive=True))

    # Run whisperx on specific file corresponding to job ID
    filepath = files[job_id]
    save_folder = "/projects/AIFMH/PIIPHI_Restricted/OCD_SUDS/audio_outputs/whisperx"
    run_whisperx(filepath, save_folder)