import argparse
import glob
import sys
import typing
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch
from imageio import imread
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy.spatial.transform import Rotation as R

sys.path.append("..")
from inverse_warp import pose_vec2mat, inverse_warp

from models import PoseExpNet, DispNetS

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

FOV = 20
IMG_SIZE = 64
focal_length = (FOV / 2) / np.tan((IMG_SIZE / 2))

INTRINSICS = np.array([[focal_length, 0, IMG_SIZE / 2],
                       [0, focal_length, IMG_SIZE / 2],
                       [0, 0, 1]])
INTRINSICS = torch.from_numpy(INTRINSICS.astype(np.float32)).unsqueeze(0)

RUNNING_PLOT_RANGE = 20


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


def generate_running_plot(x_values, x_label, x_ticks, y_values, y_label, y_ticks=None, legend_labels=None):
    fig, ax = plt.subplots()
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    max_y_val = -np.inf
    min_y_val = np.inf

    for y_vals in y_values:
        max_y_val = max(max(y_vals), max_y_val)
        min_y_val = min(min(y_vals), min_y_val)
        plt.plot(x_values, y_vals)

    plt.xticks(x_ticks)
    plt.xlabel(x_label)
    ax.invert_xaxis()

    if min_y_val > min(y_ticks) and max_y_val < max(y_ticks):
        plt.yticks(y_ticks)
    plt.ylabel(y_label)

    if legend_labels:
        plt.legend(loc="upper right", labels=legend_labels)

    plt.grid()

    fig.canvas.draw()
    plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    plot = plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close("all")

    return plot


def generate_results(frames_root_path, pose_net_path, disp_net_path, seq_length=3, step=1):
    pose_net = load_pose_exp_net(pose_net_path)

    disp_net = None
    if args.disp_net_path:
        disp_net = load_disp_net(disp_net_path)

    # each result have the following structure:
    # (orig_frame, sequential_frame, warped_frame, depth_img, sequential_diff_img, warped_diff_img)
    result_imgs = []
    losses = []
    poses = []

    for i, seq_frames in enumerate(SeqFrames(Path(frames_root_path), seq_length=seq_length, step=step)):
        imgs = [np.transpose(img, (2, 0, 1)) for img in seq_frames]

        ref_imgs = []
        for i, img in enumerate(imgs):
            temp_img = torch.from_numpy(img).unsqueeze(0)
            temp_img = ((temp_img / 255 - 0.5) / 0.5).to(device)
            if i == len(imgs) // 2:
                tgt_img = temp_img
                sequential_img = ((torch.from_numpy(imgs[i + 1]).unsqueeze(0) / 255 - 0.5) / 0.5).to(device)
                tgt_frame = np.transpose(img, (1, 2, 0)).astype(np.uint8)
                next_frame = np.transpose(imgs[i + 1], (1, 2, 0)).astype(np.uint8)
            else:
                ref_imgs.append(temp_img)

        pred_disp = None
        depth_img = np.zeros((64, 64, 3), dtype=np.uint8)

        if args.disp_net_path:
            pred_disp = disp_net(tgt_img)[0]

            depth_img = pred_disp.squeeze(0).detach().numpy()
            depth_img = cv2.cvtColor((depth_img / 10) * 255, cv2.COLOR_GRAY2BGR).astype(np.uint8)

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

        # warped diff loss
        warped_frame_tensor, valid_points = inverse_warp(tgt_img, depth_tensor, tgt_pose_tensor, INTRINSICS)

        warped_frame = np.transpose((warped_frame_tensor.squeeze(0) * 0.5 + 0.5) * 255,
                                    (1, 2, 0)).detach().numpy().astype(np.uint8)

        warped_diff = (sequential_img - warped_frame_tensor) * valid_points.unsqueeze(1).float()

        warped_diff_img = warped_diff.squeeze(0).abs().numpy()
        warped_diff_img = np.transpose(warped_diff_img, (1, 2, 0))
        warped_diff_img = np.sum(warped_diff_img, axis=2) / (2 * 3)
        warped_diff_img = cv2.cvtColor(warped_diff_img * 255, cv2.COLOR_GRAY2BGR).astype(np.uint8)

        photo_loss = warped_diff.abs().mean()
        losses.append(photo_loss)

        # sequential diff
        sequential_diff = tgt_img - sequential_img

        sequential_diff_img = sequential_diff.squeeze(0).abs().numpy()
        sequential_diff_img = np.transpose(sequential_diff_img, (1, 2, 0))
        sequential_diff_img = np.sum(sequential_diff_img, axis=2) / (2 * 3)
        sequential_diff_img = cv2.cvtColor(sequential_diff_img * 255, cv2.COLOR_GRAY2BGR).astype(np.uint8)

        result_imgs.append((tgt_frame,
                            next_frame,
                            warped_frame,
                            depth_img,
                            sequential_diff_img,
                            warped_diff_img))

    return np.array(result_imgs), np.array(losses), np.array(poses)


def create_video(output_path, result_imgs, losses, poses, repeated_frames=1):
    loss_std = losses.std()
    loss_mean = losses.mean()

    loss_lower_bound = loss_mean + loss_std
    if len(np.where(losses < loss_lower_bound)) > len(losses) / 8:
        loss_lower_bound -= loss_std * 3
    loss_upper_bound = loss_mean + loss_std
    if len(np.where(losses > loss_upper_bound)) > len(losses) / 8:
        loss_upper_bound += loss_std * 3

    pose_std = poses.std(axis=0)
    pose_mean = poses.mean(axis=0)

    trans_lower_bound = min(pose_mean[:3]) - np.abs(min(pose_std[:3]))
    if len(np.where(poses[:3] < trans_lower_bound)) > len(poses) / 8:
        trans_lower_bound -= np.abs(min(pose_std[:3])) * 3
    trans_upper_bound = max(pose_mean[:3]) + max(pose_std[:3])
    if len(np.where(poses[:3] > trans_upper_bound)) > len(poses) / 8:
        trans_upper_bound += max(pose_std[:3]) * 3

    rot_lower_bound = min(pose_mean[3:]) - np.abs(min(pose_std[3:]))
    if len(np.where(poses[3:] < rot_lower_bound)) > len(poses) / 8:
        rot_lower_bound -= np.abs(min(pose_std[3:])) * 3
    rot_upper_bound = max(pose_mean[3:]) + max(pose_std[3:])
    if len(np.where(poses[3:] > rot_upper_bound)) > len(poses) / 8:
        rot_upper_bound += max(pose_std[3:]) * 3

    x_ticks = np.array(list(reversed(range(-RUNNING_PLOT_RANGE // 2, RUNNING_PLOT_RANGE // 2 + 1))))

    # write results as video
    with imageio.get_writer(output_path, mode="I") as output_video:
        for i, (tgt_frame, next_frame, warped_frame, depth_img, sequential_diff_img, warped_diff_img) in enumerate(
                result_imgs):

            idxs = x_ticks + i
            valid_idxs = (idxs >= 0) & (idxs < len(result_imgs))
            x_values = x_ticks[valid_idxs]
            idxs = idxs[valid_idxs]

            loss_plot = generate_running_plot(x_values=x_values,
                                              x_label="n-th frame",
                                              x_ticks=x_ticks,
                                              y_values=[losses[idxs]],
                                              y_label="photo loss",
                                              y_ticks=np.linspace(0, loss_upper_bound, 10))

            # generate tx, ty, tz running plot
            trans_plot = generate_running_plot(x_values=x_values,
                                               x_label="n-th frame",
                                               x_ticks=x_ticks,
                                               y_values=list(zip(*[p[:3] for p in poses[idxs]])),
                                               y_label="translation",
                                               y_ticks=np.linspace(trans_lower_bound, trans_upper_bound, 10),
                                               legend_labels=["tx", "ty", "tz"])

            # generate yaw, pitch, roll running plot
            rot_plot = generate_running_plot(x_values=x_values,
                                             x_label="n-th frame",
                                             x_ticks=x_ticks,
                                             y_values=list(zip(*[p[3:] for p in poses[idxs]])),
                                             y_label="rotation",
                                             y_ticks=np.linspace(rot_lower_bound, rot_upper_bound, 10),
                                             legend_labels=["pitch", "yaw", "roll"])

            # resize plots
            loss_plot = cv2.resize(loss_plot, dsize=(500, 300), interpolation=cv2.INTER_LINEAR)
            trans_plot = cv2.resize(trans_plot, dsize=(500, 300), interpolation=cv2.INTER_LINEAR)
            rot_plot = cv2.resize(rot_plot, dsize=(500, 300), interpolation=cv2.INTER_LINEAR)
            second_col = np.vstack((loss_plot, trans_plot, rot_plot))

            first_row = np.hstack((tgt_frame, next_frame, warped_frame))
            second_row = np.hstack((depth_img, sequential_diff_img, warped_diff_img))
            first_col = np.vstack((first_row, second_row))
            # resize stacked frames to match plot size
            first_col = cv2.resize(first_col, dsize=(int(second_col.shape[0] / 2) * 3, second_col.shape[0]),
                                   interpolation=cv2.INTER_LINEAR)

            for _ in range(repeated_frames):
                output_video.append_data(np.concatenate((first_col, second_col), axis=1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("frames_root_path", type=str, help="Path to root dir of frames")
    parser.add_argument("pose_net_path", type=str, help="Path to trained PoseNet")
    parser.add_argument("-d", "--disp-net-path", type=str, help="Path to trained DispNet")
    parser.add_argument("-o", "--output", type=str, help="Path to output video file", default="video.mp4")
    parser.add_argument("--seq-len", type=int, help="Sequence length", default=3)
    parser.add_argument("--step", type=int, help="Number of step", default=1)

    args = parser.parse_args()

    result_imgs, losses, poses = generate_results(args.frames_root_path,
                                                  args.pose_net_path,
                                                  args.disp_net_path,
                                                  args.seq_len,
                                                  args.step)

    create_video(args.output, result_imgs, losses, poses)
