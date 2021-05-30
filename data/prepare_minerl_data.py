import glob
import random
import sys
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np


def convert_video_to_frames(video_path, frames_path):
    frames_path.mkdir(parents=True, exist_ok=True)

    video = cv2.VideoCapture(str(video_path))
    i = 0
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        cv2.imwrite(str(frames_path.joinpath(f"{i}.jpg")), frame)
        i += 1


def save_dataset(scene_desc_path, scene_paths, intrinsics):
    with open(str(scene_desc_path), "w") as f:
        for path in scene_paths:
            f.write(str(path.joinpath("frames")))
            f.write("\n")

            frames_path = path.joinpath("frames")

            frames_path.mkdir(parents=True, exist_ok=True)

            np.savetxt(str(frames_path.joinpath("cam.txt")), intrinsics)

    func_args = [(path.joinpath("recording.mp4"), path.joinpath("frames")) for path in scene_paths]
    with Pool() as pool:
        pool.starmap(convert_video_to_frames, func_args)


if len(sys.argv) != 2 and len(sys.argv) != 3:
    print("usage: python3 prepare_minerl_data.py <dataset-root-dir> <optional: train-validate-ratio>")
    exit(1)

dataset_dir_path = Path(sys.argv[1])
dataset_desc_dir_path = dataset_dir_path.joinpath("dataset")

train_ratio = float(sys.argv[2]) if len(sys.argv) == 3 else 0.7

dataset_dir_path = dataset_dir_path.joinpath("*")

scenes = glob.glob(str(dataset_dir_path))
scenes = [Path(p).absolute() for p in scenes]

n_train = int(len(scenes) * train_ratio)

train_scenes = random.sample(scenes, n_train)
val_scenes = list(filter(lambda p: p not in train_scenes, scenes))

focal_length = 2
intrinsics = np.array([[focal_length, 0, 32],
                       [0, focal_length, 32],
                       [0, 0, 1]])

dataset_desc_dir_path.mkdir(parents=True, exist_ok=True)

save_dataset(str(dataset_desc_dir_path.joinpath("train.txt")), train_scenes, intrinsics)
save_dataset(str(dataset_desc_dir_path.joinpath("val.txt")), val_scenes, intrinsics)

"""
dataset_root_dir/scene1/cam.txt
dataset_root_dir/scene1/1.jpg
dataset_root_dir/scene1/2.jpg
...
dataset_root_dir/scene2/cam.txt
dataset_root_dir/scene2/1.jpg
dataset_root_dir/scene2/2.jpg
dataset_root_dir/scene2/3.jpg

dataset_root_dir/dataset/train.txt
dataset_root_dir/dataset/val.txt
"""
