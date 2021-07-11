import argparse
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

import tqdm

from mc.create_visualisation import generate_results, create_video
from mc.data import MODELS, DATASETS


def make_video(params):
    frames_root_path, pose_net_path, disp_net_path, seq_len, step, output, compress = params

    result_imgs, valid_points_list, poses = generate_results(frames_root_path,
                                                             pose_net_path,
                                                             disp_net_path,
                                                             seq_len,
                                                             step)

    create_video(output, result_imgs, valid_points_list, poses)

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
    for model in MODELS[:1]:
        for dataset in DATASETS[:2]:
            output_path = root_output_path.joinpath(
                "{trained_on}-seq_len_{seq_len}-{dataset_name}.mp4".format(**model, dataset_name=dataset["name"]))

            tasks.append((dataset["path"],
                          model["exp_pose_model_path"],
                          model["dispnet_model_path"],
                          model["seq_len"],
                          step,
                          output_path,
                          post_compress))

    pool = Pool(processes=processes, initializer=setup_log)
    for _ in tqdm.tqdm(pool.imap_unordered(make_video, tasks), total=len(tasks)):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-p", "--processes", type=str, help="Number of processes to use", default=8)

    args = parser.parse_args()

    main(args.processes)
