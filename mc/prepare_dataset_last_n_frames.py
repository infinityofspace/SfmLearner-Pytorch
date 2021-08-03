"""
Get the last n frames of each video.
"""

import glob
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2


def convert_video_to_frames(video_path, frames_path, metadata_file_path, last_n_frames):
    with open(str(metadata_file_path)) as f:
        meta = json.load(f)

    max_frame_num = meta["true_video_frame_count"]

    frames_path.mkdir(parents=True)

    video = cv2.VideoCapture(str(video_path))
    i = 0

    # skip all non valid frames
    while video.isOpened() and i <= max_frame_num - last_n_frames:
        ret, frame = video.read()
        if not ret:
            print("the frames start value is higher than the number of frames")
            return

        i += 1

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        cv2.imwrite(str(frames_path.joinpath(f"{i}.png")), frame)

        i += 1


if len(sys.argv) != 3:
    print("usage: python3 prepare_dataset_first_time_stone.py <dataset-root-dir> <n-frames>")
    exit(1)

dataset_dir_path = Path(sys.argv[1])
dataset_dir_path = dataset_dir_path.joinpath("*")

scenes = glob.glob(str(dataset_dir_path))
scenes = [Path(p) for p in scenes]

last_n_frames = int(sys.argv[2])

func_args = [(path.joinpath("recording.mp4"),
              path.joinpath("last_n_frames"),
              path.joinpath("metadata.json"),
              last_n_frames) for path in scenes]

with Pool() as pool:
    pool.starmap(convert_video_to_frames, func_args)
