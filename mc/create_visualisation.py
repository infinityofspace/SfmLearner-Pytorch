import argparse
import glob
import sys
import typing
from itertools import repeat
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch
from imageio import imread
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation as R

sys.path.append("..")
from inverse_warp import pose_vec2mat, inverse_warp

from models import PoseExpNet, DispNetS
from utils import tensor2array

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

FOV = 20
IMG_SIZE = 64
focal_length = (FOV / 2) / np.tan((IMG_SIZE / 2))

INTRINSICS = np.array([[focal_length, 0, IMG_SIZE / 2],
                       [0, focal_length, IMG_SIZE / 2],
                       [0, 0, 1]])
INTRINSICS = torch.from_numpy(INTRINSICS.astype(np.float32)).unsqueeze(0)

RUNNING_PLOT_RANGE = list(reversed(range(20)))


def load_pose_exp_net(posenet_path: str) -> PoseExpNet:
    weights = torch.load(posenet_path, map_location=device)
    print(f"pose net epoch {weights['epoch']}")
    seq_length = int(weights["state_dict"]["conv1.0.weight"].size(1) / 3)
    pose_net = PoseExpNet(nb_ref_imgs=seq_length - 1, output_exp=False).to(device)
    pose_net.load_state_dict(weights["state_dict"], strict=False)

    return pose_net


def load_disp_net(dispnet_path: str) -> DispNetS:
    weights = torch.load(dispnet_path, map_location=device)
    print(f"disp net epoch {weights['epoch']}")
    disp_net = DispNetS().to(device)
    disp_net.load_state_dict(weights["state_dict"])
    disp_net.eval()

    return disp_net


def read_img(path):
    return imread(str(path)).astype(np.float32)[:, :, :3]


def pose_mat2vec(mat):
    translation = mat[:, :, 3]
    rotation_mat = mat[:, :, :3]
    rotation = R.from_matrix(rotation_mat).as_euler("xyz")
    return np.concatenate((translation, rotation), axis=1)


class SeqFrames(object):
    def __init__(self, root_path: Path, file_ending=".png", seq_length=3, step=1):
        paths = [Path(p) for p in glob.glob(str(root_path.joinpath(f"*{file_ending}")))]
        paths.sort(key=lambda p: int(p.name[:p.name.rfind(".")]))
        self.image_paths = np.array(paths)

        demi_length = (seq_length - 1) // 2
        shift_range = np.array([step * i for i in range(-demi_length, demi_length + 1)]).reshape(1, -1)
        indices = shift_range + np.arange(demi_length, len(self.image_paths) - demi_length).reshape(-1, 1)

        self.image_seq_paths = [self.image_paths[i] for i in indices]

    def generator(self):
        for seq_paths in self.image_seq_paths:
            imgs = [read_img(path) for path in seq_paths]

            yield imgs

    def __iter__(self) -> typing.Generator:
        return self.generator()

    def __len__(self) -> int:
        return len(self.image_paths)


def generate_running_plot(plot_range, y_values, y_label, x_label, y_ticks=None, legend_labels=None):
    fig = plt.figure()

    if not legend_labels:
        legend_labels = repeat("None")

    for y_vals, label in zip(y_values, legend_labels):
        plt.plot(plot_range[-len(y_vals):], y_vals, label=label)

    plt.xticks(plot_range)
    plt.xlabel(x_label)
    plt.yticks(y_ticks)
    plt.ylabel(y_label)

    if legend_labels:
        plt.legend(loc="upper right")

    fig.canvas.draw()
    plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    plot = plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close("all")

    return plot


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("frames_root_path", type=str, help="Path to root dir of frames")
    parser.add_argument("-p", "--pose-net-path", type=str, help="Path to trained PoseNet")
    parser.add_argument("-d", "--disp-net-path", type=str, help="Path to trained DispNet")
    parser.add_argument("-o", "--output", type=str, help="Path to output video file", default="video.mp4")

    args = parser.parse_args()

    if not args.pose_net_path and not args.disp_net_path:
        parser.error("No path for PoseNet or DispNet provided. Please provide at least one.")

    pose_net = None
    if args.pose_net_path:
        pose_net = load_pose_exp_net(args.pose_net_path)

    disp_net = None
    if args.disp_net_path:
        disp_net = load_disp_net(args.disp_net_path)

    output_video = imageio.get_writer(args.output, mode="I")

    losses = []
    poses = []

    for i, seq_frames in enumerate(SeqFrames(Path(args.frames_root_path))):
        imgs = [np.transpose(img, (2, 0, 1)) for img in seq_frames]

        ref_imgs = []
        for i, img in enumerate(imgs):
            temp_img = torch.from_numpy(img).unsqueeze(0)
            temp_img = ((temp_img / 255 - 0.5) / 0.5).to(device)
            if i == len(imgs) // 2:
                tgt_img = temp_img
                target_frame = np.transpose(img, (1, 2, 0)).astype(np.uint8)
                next_frame = np.transpose(imgs[i + 1], (1, 2, 0)).astype(np.uint8)
            else:
                ref_imgs.append(temp_img)

        pred_disp = None
        depth_img = np.zeros((64, 64, 3), dtype=np.uint8)

        if args.disp_net_path:
            pred_disp = disp_net(tgt_img)[0]

            depth_img = (255 * tensor2array(pred_disp, max_value=None, colormap='bone')).astype(np.uint8)

            depth_img = np.transpose(depth_img, (1, 2, 0))

        warped_frame = np.zeros((64, 64, 3), dtype=np.uint8)
        diff_img = np.zeros((64, 64, 3), dtype=np.uint8)
        photo_loss = None

        if args.pose_net_path:
            _, img_poses = pose_net(tgt_img, ref_imgs)

            img_poses = img_poses.cpu()[0]
            img_poses = torch.cat([img_poses[:len(imgs) // 2], torch.zeros(1, 6).float(), img_poses[len(imgs) // 2:]])

            inv_transform_matrices = pose_vec2mat(img_poses, rotation_mode="euler").detach().numpy().astype(np.float64)

            rot_matrices = np.linalg.inv(inv_transform_matrices[:, :, :3])
            tr_vectors = -rot_matrices @ inv_transform_matrices[:, :, -1:]

            transform_matrices = np.concatenate([rot_matrices, tr_vectors], axis=-1)

            first_inv_transform = inv_transform_matrices[0]
            final_poses = first_inv_transform[:, :3] @ transform_matrices
            final_poses[:, :, -1:] += first_inv_transform[:, -1:]

            img_poses = pose_mat2vec(final_poses)

            tgt_pose = img_poses[len(imgs) // 2 + 1]
            poses.append(tgt_pose)
            tgt_pose_tensor = torch.from_numpy(tgt_pose.astype(np.float32)).unsqueeze(0)

            if pred_disp is not None and False:
                depth_tensor = pred_disp
            else:
                depth_plane = np.full((64, 64, 1), 150, dtype=np.uint8)

                depth_plane = np.transpose(depth_plane, (2, 0, 1))
                depth_tensor = torch.from_numpy(depth_plane.astype(np.float32))

            warped_frame_tensor, valid_points = inverse_warp(tgt_img, depth_tensor, tgt_pose_tensor, INTRINSICS)

            warped_frame = np.transpose((warped_frame_tensor.squeeze(0) * 0.5 + 0.5) * 255,
                                        (1, 2, 0)).detach().numpy().astype(np.uint8)

            diff = (tgt_img - warped_frame_tensor) * valid_points.unsqueeze(1).float()

            diff_img = diff.squeeze(0).abs()
            diff_img = diff_img[0] + diff_img[1] + diff_img[2]

            diff_img = (255 * tensor2array(diff_img, max_value=1, colormap="bone")).astype(np.uint8)
            diff_img = np.transpose(diff_img, (1, 2, 0))

            photo_loss = diff.abs().mean()

            losses.append(photo_loss)

        loss_plot = generate_running_plot(plot_range=RUNNING_PLOT_RANGE,
                                          y_values=[losses[-len(RUNNING_PLOT_RANGE):]],
                                          y_label="photo loss",
                                          x_label="n-th last frame")

        # generate tx, ty, tz running plot
        txs = [p[0] for p in poses[-len(RUNNING_PLOT_RANGE):]]
        tys = [p[1] for p in poses[-len(RUNNING_PLOT_RANGE):]]
        tzs = [p[2] for p in poses[-len(RUNNING_PLOT_RANGE):]]
        trans_plot = generate_running_plot(plot_range=RUNNING_PLOT_RANGE,
                                           y_values=[txs, tys, tzs],
                                           y_label="translation",
                                           x_label="n-th last frame",
                                           legend_labels=["tx", "ty", "tz"])

        # generate yaw, pitch, roll running plot
        yaws = [p[3] for p in poses[-len(RUNNING_PLOT_RANGE):]]
        pitchs = [p[4] for p in poses[-len(RUNNING_PLOT_RANGE):]]
        rolls = [p[5] for p in poses[-len(RUNNING_PLOT_RANGE):]]
        rot_plot = generate_running_plot(plot_range=RUNNING_PLOT_RANGE,
                                         y_values=[yaws, pitchs, rolls],
                                         y_label="rotation",
                                         x_label="n-th last frame",
                                         legend_labels=["yaw", "pitch", "roll"])

        loss_plot = cv2.resize(loss_plot, dsize=(450, 250), interpolation=cv2.INTER_LINEAR)
        trans_plot = cv2.resize(trans_plot, dsize=(450, 250), interpolation=cv2.INTER_LINEAR)
        rot_plot = cv2.resize(rot_plot, dsize=(450, 250), interpolation=cv2.INTER_LINEAR)
        second_col = np.vstack((loss_plot, trans_plot, rot_plot))

        first_row = np.hstack((target_frame, next_frame, warped_frame))
        second_row = np.hstack((depth_img, diff_img, np.zeros_like(diff_img)))
        first_col = np.vstack((first_row, second_row))
        first_col = cv2.resize(first_col, dsize=(int(second_col.shape[0] / 3) * 4, second_col.shape[0]),
                               interpolation=cv2.INTER_LINEAR)

        output_video.append_data(np.concatenate((first_col, second_col), axis=1))

    output_video.close()
