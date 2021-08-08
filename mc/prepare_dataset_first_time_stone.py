"""
Remove all frames at the beginning until the agent mines the first time stone or cobblestone.
Any frames with no movement will be removed.
"""

import glob
import json
import random
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np


def convert_video_to_frames(video_path, frames_path, frames_start):
    frames_path.mkdir(parents=True, exist_ok=True)

    video = cv2.VideoCapture(str(video_path))
    i = 0

    # skip all non valid frames
    while video.isOpened() and i < frames_start:
        ret, frame = video.read()
        if not ret:
            print("the frames start value is higher than the number of frames")
            return

        i += 1

    last_frame = None

    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        # filter static frames eg. all frames with no big changes
        if last_frame is not None and cv2.absdiff(last_frame, frame).mean() < 1:
            continue

        last_frame = frame

        cv2.imwrite(str(frames_path.joinpath(f"{i}.png")), frame)

        i += 1


def get_frames_start(metadata_file_path, actions_file_path):
    with open(str(metadata_file_path)) as f:
        meta = json.load(f)

    max_frame_num = meta["true_video_frame_count"]
    actions = np.load(str(actions_file_path))
    num_states = actions["reward"].shape[0]

    start_frame = max_frame_num - num_states

    # check when the agent mines the first time stone
    stones = np.where(actions["observation$inventory$stone"] > 0)[0]
    cobblestones = np.where(actions["observation$inventory$cobblestone"] > 0)[0]

    if len(stones) > 0:
        first_time_stone = stones[0]
    else:
        first_time_stone = 0

    if len(cobblestones) > 0:
        first_time_cobblestone = cobblestones[0]
    else:
        first_time_cobblestone = 0

    return start_frame + min(first_time_stone, first_time_cobblestone)


def save_dataset(dataset_desc_path, scene_paths, intrinsics):
    with open(str(dataset_desc_path), "w") as f:
        for path in scene_paths:
            frames_path = path.joinpath("frames")

            f.write(str(frames_path))
            f.write("\n")

            frames_path.mkdir(parents=True, exist_ok=True)

            np.savetxt(str(frames_path.joinpath("cam.txt")), intrinsics)

    func_args = [(path.joinpath("recording.mp4"), path.joinpath("frames"),
                  get_frames_start(path.joinpath("metadata.json"), path.joinpath("rendered.npz"))) for path in
                 scene_paths]
    with Pool() as pool:
        pool.starmap(convert_video_to_frames, func_args)


if len(sys.argv) != 2 and len(sys.argv) != 3:
    print("usage: python3 prepare_dataset_first_time_stone.py <dataset-root-dir> <optional: train-validate-ratio>")
    exit(1)

dataset_dir_path = Path(sys.argv[1])
dataset_desc_dir_path = dataset_dir_path.joinpath("dataset")

train_ratio = float(sys.argv[2]) if len(sys.argv) == 3 else 0.7

dataset_dir_path = dataset_dir_path.joinpath("*")

scenes = glob.glob(str(dataset_dir_path))
scenes = [Path(p) for p in scenes]

n_train = int(len(scenes) * train_ratio)

train_scenes = random.sample(scenes, n_train)
val_scenes = list(filter(lambda p: p not in train_scenes, scenes))

FOV = 20
IMG_SIZE = 64

focal_length = (FOV / 2) / np.tan((IMG_SIZE / 2))
intrinsics = np.array([[focal_length, 0, IMG_SIZE / 2],
                       [0, focal_length, IMG_SIZE / 2],
                       [0, 0, 1]])

dataset_desc_dir_path.mkdir(parents=True, exist_ok=True)

save_dataset(str(dataset_desc_dir_path.joinpath("train.txt")), train_scenes, intrinsics)
save_dataset(str(dataset_desc_dir_path.joinpath("val.txt")), val_scenes, intrinsics)

"""
dataset_root_dir/scene1/cam.txt
dataset_root_dir/scene1/1.png
dataset_root_dir/scene1/2.png
...
dataset_root_dir/scene2/cam.txt
dataset_root_dir/scene2/1.png
dataset_root_dir/scene2/2.png
dataset_root_dir/scene2/3.png

dataset_root_dir/dataset/train.txt
dataset_root_dir/dataset/val.txt
"""
