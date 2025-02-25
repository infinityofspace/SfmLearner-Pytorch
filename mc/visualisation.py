import argparse
import glob
import sys
import time
import typing
from pathlib import Path

import cv2
import imageio
import numpy as np
import torch
from imageio import imread
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from scipy.optimize import optimize
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

MARKERS = [".", "v", "^", "<", ">", "1", "2", "3", "4"]


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
        assert root_path.exists()

        paths = [Path(p) for p in glob.glob(str(root_path.joinpath(f"*{file_ending}")))]
        paths.sort(key=lambda p: int(p.name[:p.name.rfind(".")]))
        self.image_paths = np.array(paths)

        demi_length = (seq_length - 1) // 2
        shift_range = np.array([step * i for i in range(-demi_length, demi_length + 1)]).reshape(1, -1)
        indices = shift_range + np.arange(demi_length, len(self.image_paths) - demi_length).reshape(-1, 1)

        self.image_seq_paths = [self.image_paths[i] for i in indices if
                                np.min(i) > 0 and np.max(i) < len(self.image_paths)]

    def generator(self):
        for seq_paths in self.image_seq_paths:
            imgs = [read_img(path) for path in seq_paths]

            yield imgs

    def __iter__(self) -> typing.Generator:
        return self.generator()

    def __len__(self) -> int:
        return len(self.image_paths)


def generate_running_plot(x_values, x_label, x_ticks, y_values, y_label, y_ticks=None, legend_labels=None,
                          markers=None):
    fig, ax = plt.subplots()
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    max_y_val = -np.inf
    min_y_val = np.inf

    for i, y_vals in enumerate(y_values):
        if markers:
            marker = markers[i]
        else:
            marker = None
        max_y_val = max(np.max(y_vals), max_y_val)
        min_y_val = min(np.min(y_vals), min_y_val)
        plt.plot(x_values, y_vals, marker=marker)

    plt.xticks(x_ticks)
    plt.xlabel(x_label)
    ax.invert_xaxis()

    zero_idx = int(np.where(x_ticks == 0)[0])
    gridlines = ax.xaxis.get_gridlines()
    gridlines[zero_idx].set_color("k")
    gridlines[zero_idx].set_linewidth(1)

    if min_y_val > min(y_ticks) and max_y_val < max(y_ticks):
        plt.yticks(y_ticks)
    plt.ylabel(y_label)

    if legend_labels:
        plt.legend(loc="upper right", labels=legend_labels)

    plt.grid()

    fig.tight_layout()
    fig.canvas.draw()
    plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    plot = plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close("all")

    return plot


def sfm_learner(seq_frames, disp_net, pose_net):
    imgs = [np.transpose(img, (2, 0, 1)) for img in seq_frames]

    ref_imgs = []
    for i, img in enumerate(imgs):
        temp_img = torch.from_numpy(img).unsqueeze(0)
        temp_img = ((temp_img / 255 - 0.5) / 0.5).to(device)
        if i == len(imgs) // 2:
            tgt_img = temp_img
            tgt_frame = np.transpose(img, (1, 2, 0)).astype(np.uint8)
            next_frame = np.transpose(imgs[i + 1], (1, 2, 0)).astype(np.uint8)
        else:
            ref_imgs.append(temp_img)

    start_time = time.time()
    pred_disp = 1 / disp_net(tgt_img)[0]
    disp_net_calc_time = time.time() - start_time

    depth_img = pred_disp.squeeze(0).detach().numpy()
    # scale every value to max 1
    depth_img[depth_img > 1] = 1
    depth_img = cv2.cvtColor(depth_img * 255, cv2.COLOR_GRAY2BGR).astype(np.uint8)

    start_time = time.time()
    _, img_poses = pose_net(tgt_img, ref_imgs)
    pose_net_calc_time = time.time() - start_time

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
    tgt_pose_tensor = torch.from_numpy(tgt_pose.astype(np.float32)).unsqueeze(0)

    depth_tensor = pred_disp.detach()

    warped_frame_tensor, valid_points = inverse_warp(tgt_img, depth_tensor, tgt_pose_tensor, INTRINSICS)

    warped_frame = np.transpose((warped_frame_tensor.squeeze(0) * 0.5 + 0.5) * 255,
                                (1, 2, 0)).detach().numpy().astype(np.uint8)

    valid_points = np.transpose(valid_points.numpy(), (1, 2, 0))

    return tgt_frame, \
           next_frame, \
           warped_frame * valid_points, \
           depth_img, \
           valid_points, \
           tgt_pose, \
           disp_net_calc_time, \
           pose_net_calc_time


def sfm_learner_predict(frame_seqs, pose_net_path, disp_net_path, step=1):
    pose_net = load_pose_exp_net(pose_net_path)

    disp_net = load_disp_net(disp_net_path)

    tgt_frames = []
    next_frames = []
    warped_frames = []
    depth_imgs = []
    valid_points_list = []
    poses = []

    calc_timings = []

    for seq_frames in frame_seqs:
        pred_res = sfm_learner(seq_frames, disp_net, pose_net)

        tgt_frames.append(pred_res[0])
        next_frames.append(pred_res[1])
        warped_frames.append(pred_res[2])
        depth_imgs.append(pred_res[3])
        valid_points_list.append(pred_res[4])
        poses.append(pred_res[5])
        calc_timings.append(pred_res[6:8])

    return tgt_frames, next_frames, warped_frames, valid_points_list, poses, depth_imgs, calc_timings


def convert_rgb_to_gray_rgb(img, max=3 * 255, color_map="gray"):
    temp_img = ((np.sum(img.astype(np.float32), axis=2) / max) * 255).astype(np.uint8)
    temp_img = cv2.cvtColor(temp_img, cv2.COLOR_GRAY2RGB).astype(np.uint8)

    return temp_img


def rad(grad):
    return grad * np.pi / 180


def frame_to_tensor(frame):
    return torch.from_numpy(np.transpose(frame, (2, 0, 1)).astype(np.float32)).unsqueeze(0)


def tensor_to_frame(tensor):
    return np.abs(np.transpose(tensor.squeeze(0), (1, 2, 0)).numpy()).astype(np.uint8)


def grid_search_pose(rranges, pose_idxs, cur_img, next_img, depth_img, intrinsics):
    def photometric_loss(vals, *params):
        img, next_img, depth, intrinsics, pose_idxs = params
        pose = np.array([vals[idx] if idx is not None else 0 for idx in pose_idxs])
        pose = torch.from_numpy(pose.astype(np.float32)).unsqueeze(0)
        warped_img, valid_points = inverse_warp(img, depth, pose, intrinsics)

        diff = (next_img - warped_img) * valid_points.unsqueeze(1).float()
        return float(diff.abs().mean())

    depth_tensor = torch.from_numpy(
        np.expand_dims(cv2.cvtColor(depth_img, cv2.COLOR_RGB2GRAY), axis=0).astype(np.float32))
    cur_img_tensor = frame_to_tensor(cur_img)
    next_img_tensor = frame_to_tensor(next_img)

    params = (cur_img_tensor, next_img_tensor, depth_tensor, intrinsics, np.array(pose_idxs))

    start_time = time.time()
    res = optimize.brute(photometric_loss, rranges, args=params, full_output=True, finish=optimize.fmin)
    calc_time = time.time() - start_time

    pose = np.array([res[0][idx] if idx is not None else 0 for idx in pose_idxs])
    pose = torch.from_numpy(pose.astype(np.float32)).unsqueeze(0)

    warped, valid_points = inverse_warp(cur_img_tensor, depth_tensor, pose, INTRINSICS)

    valid_points = np.transpose(valid_points.numpy(), (1, 2, 0))

    return cur_img, next_img, tensor_to_frame(warped) * valid_points, valid_points, pose.squeeze(0).numpy(), calc_time


def generate_diffs(next_frames, warped_frames, valid_points_list):
    diff_imgs = []
    hsv_diff_imgs = []
    warped_imgs = []
    merged_imgs = []
    losses = []
    hs_losses = []

    for next_frame, warped_frame, valid_points in zip(next_frames, warped_frames, valid_points_list):
        warped_imgs.append(warped_frame)

        warped_diff = cv2.absdiff(next_frame, warped_frame) * valid_points
        warped_diff_img = convert_rgb_to_gray_rgb(warped_diff)
        diff_imgs.append(warped_diff_img)

        photo_loss = np.abs(warped_diff).mean()
        losses.append(photo_loss)

        warped_frame_hsv = cv2.cvtColor(warped_frame, cv2.COLOR_RGB2HSV)
        warped_frame_hsv[:, :, 2] = 0
        next_frame_hsv = cv2.cvtColor(next_frame, cv2.COLOR_RGB2HSV)
        next_frame_hsv[:, :, 2] = 0
        warped_hs_diff = cv2.absdiff(next_frame_hsv, warped_frame_hsv) * valid_points
        warped_hs_diff_img = convert_rgb_to_gray_rgb(warped_hs_diff)
        hsv_diff_imgs.append(warped_hs_diff_img)

        hs_photo_loss = np.abs(warped_hs_diff).mean()
        hs_losses.append(hs_photo_loss)

        merged_img = (0.5 * next_frame + 0.5 * warped_frame).astype(np.uint8)
        merged_imgs.append(merged_img)

    return warped_imgs, merged_imgs, diff_imgs, hsv_diff_imgs, losses, hs_losses


def generate_plots(losses, loss_legend, hs_losses, hs_loss_legend, poses, trans_legend, rot_legend):
    losses = np.array(losses)
    hs_losses = np.array(hs_losses)

    poses = np.array(poses)

    loss_std = losses.std(axis=0)
    loss_median = np.median(losses, axis=0)

    loss_lower_bound = loss_median.min() + loss_std.min()
    loss_upper_bound = loss_median.max() + loss_std.max()

    pose_std = poses.std(axis=0)
    pose_median = np.median(poses, axis=0)

    trans_lower_bound = pose_median[:, :3].min() - np.abs(pose_std[:, :3].min())
    trans_upper_bound = pose_median[:, :3].max() + pose_std[:, :3].max()

    rot_lower_bound = pose_median[:, 3:].min() - np.abs(pose_std[:, 3:].min())
    rot_upper_bound = pose_median[:, 3:].max() + pose_std[:, 3:].max()

    x_ticks = np.array(list(reversed(range(-RUNNING_PLOT_RANGE // 2, RUNNING_PLOT_RANGE // 2 + 1))))

    loss_plots = []
    hs_loss_plots = []
    trans_plots = []
    rot_plots = []

    for i in range(len(losses[0])):
        idxs = x_ticks + i
        valid_idxs = (idxs >= 0) & (idxs < len(losses[0]))
        x_values = x_ticks[valid_idxs]
        idxs = idxs[valid_idxs]

        loss_plot = generate_running_plot(x_values=x_values,
                                          x_label="n-th frame",
                                          x_ticks=x_ticks,
                                          y_values=losses[:, idxs],
                                          y_label="photo loss",
                                          y_ticks=np.linspace(loss_lower_bound, loss_upper_bound, 10),
                                          legend_labels=loss_legend)

        loss_plots.append(loss_plot)

        hs_loss_plot = generate_running_plot(x_values=x_values,
                                             x_label="n-th frame",
                                             x_ticks=x_ticks,
                                             y_values=hs_losses[:, idxs],
                                             y_label="hs photo loss",
                                             y_ticks=np.linspace(0, loss_upper_bound, 10),
                                             legend_labels=hs_loss_legend)

        hs_loss_plots.append(hs_loss_plot)

        # generate tx, ty, tz running plot
        trans_y_vals = [list(zip(*p[idxs][:, :3])) for p in poses]
        trans_y_vals = [val for sub in trans_y_vals for val in sub]

        # more than one model
        if len(trans_y_vals) > 3:
            markers = [idx for sub in [MARKERS[i] * 3 for i in range(len(trans_y_vals) // 3)] for idx in sub]
        else:
            markers = None

        trans_plot = generate_running_plot(x_values=x_values,
                                           x_label="n-th frame",
                                           x_ticks=x_ticks,
                                           y_values=trans_y_vals,
                                           y_label="translation",
                                           y_ticks=np.linspace(trans_lower_bound, trans_upper_bound, 10),
                                           legend_labels=trans_legend,
                                           markers=markers)

        trans_plots.append(trans_plot)

        # generate yaw, pitch, roll running plot
        rot_y_vals = [list(zip(*p[idxs][:, 3:])) for p in poses]
        rot_y_vals = [val for sub in rot_y_vals for val in sub]

        # more than one model
        if len(trans_y_vals) > 3:
            markers = [idx for sub in [MARKERS[i] * 3 for i in range(len(trans_y_vals) // 3)] for idx in sub]
        else:
            markers = None

        rot_plot = generate_running_plot(x_values=x_values,
                                         x_label="n-th frame",
                                         x_ticks=x_ticks,
                                         y_values=rot_y_vals,
                                         y_label="rotation",
                                         y_ticks=np.linspace(rot_lower_bound, rot_upper_bound, 10),
                                         legend_labels=rot_legend,
                                         markers=markers)

        rot_plots.append(rot_plot)

    return trans_plots, rot_plots, loss_plots, hs_loss_plots


def create_video_frames(grid_frames, labels, plots, video_height, video_width):
    frames = []

    for grid_list, plot_list in zip(grid_frames, plots):
        grid_frame = np.vstack([np.hstack(row) for row in grid_list])
        plot_frame = np.vstack(plot_list)

        # resize
        plots_size = (int(1.5 * video_height // len(plot_list)), video_height)
        plot_frame = cv2.resize(plot_frame, dsize=plots_size, interpolation=cv2.INTER_CUBIC)

        grid_frame = cv2.resize(grid_frame, dsize=(
            (plot_frame.shape[0] // len(grid_list)) * len(grid_list[0]), plot_frame.shape[0]),
                                interpolation=cv2.INTER_CUBIC)

        # add labels
        for row, labels_list in enumerate(labels):
            for col, l in enumerate(labels_list):
                block_size = grid_frame.shape[1] // len(labels_list)
                cv2.putText(img=grid_frame,
                            text=l,
                            org=(block_size // 2 + col * block_size - 4 * len(l),
                                 15 + row * grid_frame.shape[1] // len(labels_list)),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5,
                            color=(255, 0, 0),
                            thickness=1,
                            lineType=cv2.LINE_AA)

        frame = np.concatenate((grid_frame, plot_frame), axis=1)

        frames.append(frame)

    return frames


def save_video(output_path, frames, repeated_frames=1):
    # write results as video
    with imageio.get_writer(output_path, mode="I") as output_video:
        for frame in frames:
            for _ in range(repeated_frames):
                output_video.append_data(frame)


def main(frames_root_path: str,
         pose_net_path: str,
         disp_net_path: str,
         seq_len: int = 3,
         step: int = 1,
         output: str = "video.mp4",
         gs_tx: typing.Union[float, float, float] = None,
         gs_ty: typing.Union[float, float, float] = None,
         gs_tz: typing.Union[float, float, float] = None,
         gs_pitch: typing.Union[float, float, float] = None,
         gs_yaw: typing.Union[float, float, float] = None,
         gs_roll: typing.Union[float, float, float] = None,
         timings: bool = False):
    loss_legend = ["1f - warped", "0f - 1f", "0f - 1f (valid points)"]

    trans_plot_legend = ["tx", "ty", "tz"]
    rot_plot_legend = ["pitch", "yaw", "roll"]

    grid_frame_labels = [["0-frame", "1-frame", "warped"],
                         ["depth 0f", "0f/2 + 1f/2", "1f/2 + warped/2"],
                         ["depth 1f", "0f - 1f", "1f - warped"],
                         ["depth 0f - depth 1f", "0f - 1f hs", "1f - warped hs"]]

    losses = []
    hs_losses = []
    poses = []

    # warped_imgs, merged_imgs, diff_imgs, hsv_diff_imgs
    result_imgs = []

    print("sfmlearner predict")

    #  tgt_frames, next_frames, warped_frames, valid_points_list, poses, depth_imgs
    *predict_res, nn_calc_timings = sfm_learner_predict(
        SeqFrames(Path(frames_root_path), seq_length=seq_len, step=step),
        pose_net_path,
        disp_net_path)

    if timings:
        n_images = len(predict_res[0])
        total_disp_net_calc_time = sum([calc_times[0] for calc_times in nn_calc_timings])
        total_pose_net_calc_time = sum([calc_times[1] for calc_times in nn_calc_timings])
        print("avg disp net calc time: {:.3f} \tavg pose net calc time: {:.3f}".format(
            total_disp_net_calc_time / n_images,
            total_pose_net_calc_time / n_images))
        print("total nn calc time: {:.3f}".format(total_disp_net_calc_time + total_pose_net_calc_time))

    tgt_imgs = predict_res[0]
    next_imgs = predict_res[1]
    depths = predict_res[5]
    poses.append(predict_res[4])

    *diff_imgs, photo_l, hs_photo_l = generate_diffs(tgt_imgs, next_imgs, np.ones_like(next_imgs))
    result_imgs.append(diff_imgs)
    losses.append(photo_l)
    hs_losses.append(hs_photo_l)

    # sequential diff with valid_points
    *_, photo_l, hs_photo_l = generate_diffs(tgt_imgs, next_imgs, predict_res[3])
    losses.append(photo_l)
    hs_losses.append(hs_photo_l)

    *diff_imgs, photo_l, hs_photo_l = generate_diffs(next_imgs, predict_res[2], predict_res[3])
    result_imgs.append(diff_imgs)
    losses.append(photo_l)
    hs_losses.append(hs_photo_l)

    total_grid_search_calc_time = 0

    if gs_tx or gs_ty or gs_tz or gs_pitch or gs_yaw or gs_roll:
        loss_legend.append("1f - grid search warped")

        trans_plot_legend.extend(["tx gs", "ty gs", "tz gs"])
        rot_plot_legend.extend(["pitch gs", "yaw gs", "roll gs"])

        labels = [["grid search warped"],
                  ["1f/2 + gs warped/2"],
                  ["1f - gs warped"],
                  ["1f - gs warped hs"]]

        grid_frame_labels = [row + labels[i] for i, row in enumerate(grid_frame_labels)]

        print("grid search predict")
        # grid search
        rranges = []
        pose_idxs = [None, None, None, None, None, None]
        pose_idx_count = 0

        if gs_tx:
            rranges.append(slice(rad(gs_tx[0]), rad(gs_tx[1]), gs_tx[2]))
            pose_idxs[0] = pose_idx_count
            pose_idx_count += 1
        if gs_ty:
            rranges.append(slice(rad(gs_ty[0]), rad(gs_ty[1]), gs_ty[2]))
            pose_idxs[1] = pose_idx_count
            pose_idx_count += 1
        if gs_tz:
            rranges.append(slice(rad(gs_tz[0]), rad(gs_tz[1]), gs_tz[2]))
            pose_idxs[2] = pose_idx_count
            pose_idx_count += 1
        if gs_pitch:
            rranges.append(slice(rad(gs_pitch[0]), rad(gs_pitch[1]), gs_pitch[2]))
            pose_idxs[3] = pose_idx_count
            pose_idx_count += 1
        if gs_yaw:
            rranges.append(slice(rad(gs_yaw[0]), rad(gs_yaw[1]), gs_yaw[2]))
            pose_idxs[4] = pose_idx_count
            pose_idx_count += 1
        if gs_roll:
            rranges.append(slice(rad(gs_roll[0]), rad(gs_roll[1]), gs_roll[2]))
            pose_idxs[5] = pose_idx_count
            pose_idx_count += 1

        rranges = tuple(rranges)

        warped_imgs = []
        valid_points = []
        poses_list = []
        grid_search_calc_timings = []
        for img, seq_img, depth_img in zip(tgt_imgs, next_imgs, depths):
            # cur_img, next_img, tensor_to_frame(warped) * valid_points, valid_points, pose.squeeze(0).numpy()
            _, _, warped, v_points, ps, calc_timing = grid_search_pose(rranges, pose_idxs, img, seq_img, depth_img,
                                                                       INTRINSICS)
            warped_imgs.append(warped)
            valid_points.append(v_points)
            poses_list.append(ps)
            grid_search_calc_timings.append(calc_timing)

        if timings:
            total_grid_search_calc_time = sum(grid_search_calc_timings)
            print("avg gs calc time: {:.3f}".format(total_grid_search_calc_time / len(tgt_imgs)))
            print("total gs calc time: {:.3f}".format(total_grid_search_calc_time))

        poses.append(poses_list)

        *diff_imgs, photo_l, hs_photo_l = generate_diffs(next_imgs, warped_imgs, valid_points)
        result_imgs.append(diff_imgs)
        losses.append(photo_l)
        hs_losses.append(hs_photo_l)

    if timings:
        print("total calc time: {:.3f}".format(
            total_disp_net_calc_time + total_pose_net_calc_time + total_grid_search_calc_time))

    plots = list(zip(*generate_plots(losses, loss_legend,
                                     hs_losses, loss_legend,
                                     poses, trans_plot_legend, rot_plot_legend)))

    grid_frames = []

    for i in range(len(tgt_imgs) - 1):
        current_depth = depths[i]
        next_depth = depths[i + 1]
        depth_diff = cv2.absdiff(current_depth, next_depth)
        frame = [[tgt_imgs[i]], [current_depth], [next_depth], [depth_diff]]

        for res in result_imgs:
            for row, imgs in enumerate(res):
                frame[row].append(imgs[i])

        grid_frames.append(frame)

    frames = create_video_frames(grid_frames, grid_frame_labels, plots, 1200, 0)

    save_video(output, frames)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("frames_root_path", type=str, help="Path to root dir of frames")
    parser.add_argument("pose_net_path", type=str, help="Path to trained PoseNet")
    parser.add_argument("disp_net_path", type=str, help="Path to trained DispNet")
    parser.add_argument("-o", "--output", type=str, help="Path to output video file", default="video.mp4")
    parser.add_argument("--seq-len", type=int, help="Sequence length", default=3)
    parser.add_argument("--step", type=int, help="Number of steps", default=1)
    parser.add_argument("--gs-tx", type=int, help="Grid search tx value with given range and step size", nargs=3)
    parser.add_argument("--gs-ty", type=float, help="Grid search ty value with given range and step size", nargs=3)
    parser.add_argument("--gs-tz", type=float, help="Grid search tz value with given range and step size", nargs=3)
    parser.add_argument("--gs-pitch", type=float, help="Grid search pitch value with given range and step size",
                        nargs=3)
    parser.add_argument("--gs-yaw", type=float, help="Grid search yaw value with given range and step size", nargs=3)
    parser.add_argument("--gs-roll", type=float, help="Grid search roll value with given range and step size", nargs=3)
    parser.add_argument("--timings", help="print the calculation timings", action="store_true")

    args = parser.parse_args()

    main(args.frames_root_path,
         args.pose_net_path,
         args.disp_net_path,
         args.seq_len,
         args.step,
         args.output,
         args.gs_tx,
         args.gs_ty,
         args.gs_tz,
         args.gs_pitch,
         args.gs_yaw,
         args.gs_roll,
         args.timings)
