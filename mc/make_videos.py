import argparse
import os
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

import tqdm

from data import MODELS, DATASETS
import visualisation


def make_video(params):
    frames_root_path, pose_net_path, disp_net_path, seq_len, step, output, compress, grid_search = params

    if not grid_search:
        grid_search = {}
    visualisation.main(frames_root_path, pose_net_path, disp_net_path, seq_len, step, output, **grid_search)

    output_path = Path(output)
    print(output_path, compress, output_path.exists())
    if compress and output_path.exists():
        subprocess.call(
            ["ffmpeg", "-i", output, "-vcodec", "libx265", "-crf", "30", f"videos/compressed/{output_path.name}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT)


def setup_log():
    sys.stdout = open("make_videos.log", "w")


def main(processes, post_compress=True):
    step = 1

    root_output_path = Path("videos")
    root_output_path.mkdir(parents=True, exist_ok=True)
    if post_compress:
        root_output_path.joinpath("compressed").mkdir(parents=True, exist_ok=True)
        root_output_path.joinpath("orig").mkdir(parents=True, exist_ok=True)
        root_output_path = root_output_path.joinpath("orig")

    tasks = []
    for model in MODELS:
        for dataset in DATASETS:
            output_path = root_output_path.joinpath(
                "{trained_on}-seq_len_{seq_len}-{dataset_name}.mp4".format(**model, dataset_name=dataset["name"]))

            tasks.append((dataset["path"],
                          model["exp_pose_model_path"],
                          model["dispnet_model_path"],
                          model["seq_len"],
                          step,
                          output_path,
                          post_compress,
                          dataset.get("grid_search", None)))

    pool = Pool(processes=processes, initializer=setup_log)
    for _ in tqdm.tqdm(pool.imap_unordered(make_video, tasks), total=len(tasks)):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-p", "--processes", type=int, help="Number of processes to use", default=os.cpu_count())

    args = parser.parse_args()

    main(args.processes)
