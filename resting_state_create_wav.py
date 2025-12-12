import os
import moviepy

file_path =  "/Users/Timon/Library/CloudStorage/Box-Box/APL_BCM_Share_SUDS/Resting-State-Videos"

subjects = os.listdir(file_path)
for subject in subjects:
    subject_path = os.path.join(file_path, subject)
    date_folders = os.listdir(subject_path)
    for date in date_folders:
        video_folder_path = os.path.join(subject_path, date)
        video_path = [f for f in os.listdir(video_folder_path) if f.endswith('.mp4')][0]
        video_full_path = os.path.join(video_folder_path, video_path)
        