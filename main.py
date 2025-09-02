import sys
import glob
from transcribe_and_diarize import run_whisperx


if __name__ == "__main__":

    # Get job id
    #job_id = int(sys.argv[1])
    job_id = 0
    # Get all .wav files
    root = "/projects/AIFMH/PIIPHI_Restricted/OCD_SUDS"
    root = '/Users/Timon/Library/CloudStorage/Box-Box/aDBS ERP Master Folder'
    files = sorted(glob.glob(root + '/**/**/**/*.m4a', recursive=True))

    # Run whisperx on specific file corresponding to job ID
    filepath = files[job_id]
    save_folder = "/projects/AIFMH/PIIPHI_Restricted/OCD_SUDS/audio_outputs/whisperx"
    save_folder = "/Users/Timon/Documents/Houston/whisper/save_folder"
    run_whisperx(filepath, save_folder)