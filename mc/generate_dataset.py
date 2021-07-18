import random
import shutil
from pathlib import Path

import minerl
import cv2
import gym
import imageio
import numpy as np

root_dataset_dir_path = Path("mc_predefined_agent_actions_v3")

env = gym.make("MineRLTreechop-v0")

random_translation_actions = []
for _ in range(200):
    forward = random.randint(0, 1)
    back = random.randint(0, 1) ^ forward
    left = random.randint(0, 1)
    right = random.randint(0, 1) ^ left
    jump = random.randint(0, 1)

    random_translation_actions.extend([{
        "forward": forward,
        "back": back,
        "left": left,
        "right": right,
        "jump": jump
    }] * 3)

random_rotation_actions = []
for _ in range(200):
    pitch = random.randint(-45, 45)
    yaw = random.randint(-45, 45)

    random_rotation_actions.extend([{
        "camera": [pitch, yaw],
    }] * 3)

dataset_actions = [
    {
        "actions": [
                       {
                           "camera": np.array([0, 5])
                       }
                   ] * 72 * 6,
        "action_folder_name": "camera_yaw_5_repeated",
        "samples": 5
    },
    {
        "actions": [
                       {
                           "camera": np.array([0, 30])
                       }
                   ] * 12 * 40,
        "action_folder_name": "camera_yaw_30_repeated",
        "samples": 5
    },
    {
        "actions": ([
                        {
                            "camera": np.array([5, 0])
                        }
                    ] * 18 +
                    [
                        {
                            "camera": np.array([-5, 0])
                        }
                    ] * 18 * 2 +
                    [
                        {
                            "camera": np.array([5, 0])
                        }
                    ] * 18) * 6,
        "action_folder_name": "camera_pitch_5_repeated",
        "samples": 5
    },
    {
        "actions": ([
                        {
                            "camera": np.array([30, 0])
                        }
                    ] * 3 +
                    [
                        {
                            "camera": np.array([-30, 0])
                        }
                    ] * 3 * 2 +
                    [
                        {
                            "camera": np.array([30, 0])
                        }
                    ] * 3) * 40,
        "action_folder_name": "camera_pitch_30_repeated",
        "samples": 5
    },
    {
        "actions": [
                       {
                           "forward": 1
                       }
                   ] * 500,
        "action_folder_name": "forward",
        "samples": 5,
        "min_frames": 120
    },
    {
        "actions": [
                       {
                           "back": 1
                       }
                   ] * 500,
        "action_folder_name": "back",
        "samples": 5,
        "min_frames": 120
    },
    {
        "actions": [
                       {
                           "left": 1
                       }
                   ] * 500,
        "action_folder_name": "left",
        "samples": 5,
        "min_frames": 120
    },
    {
        "actions": [
                       {
                           "right": 1
                       }
                   ] * 500,
        "action_folder_name": "right",
        "samples": 5,
        "min_frames": 120
    },
    {
        "actions": [
                       {
                           "jump": 1
                       }
                   ] * 500,
        "action_folder_name": "jump",
        "samples": 5,
        "min_frames": 120
    },
    {
        "actions": random_translation_actions,
        "action_folder_name": "random_translation_3d",
        "samples": 5
    },
    {
        "actions": random_rotation_actions,
        "action_folder_name": "random_rotation_3d",
        "samples": 5
    },
]

for action_dict in dataset_actions:
    sample = 0
    while sample < action_dict["samples"]:
        obs = env.reset()

        sample_dir_path = root_dataset_dir_path.joinpath(action_dict["action_folder_name"], str(sample))
        sample_dir_path.mkdir(parents=True)

        frames_dir_path = sample_dir_path.joinpath("frames")
        frames_dir_path.mkdir(parents=True)

        sample_actions = []

        invalid_sample = False

        i = 0

        try:
            action = env.action_space.noop()
            obs, reward, done, info = env.step(action)

            last_pov = None

            for act in action_dict["actions"]:
                action = env.action_space.noop()

                cur_pov = obs["pov"]

                if "min_frames" in action_dict and last_pov is not None and i < action_dict["min_frames"] and np.abs(
                        cv2.absdiff(last_pov, cur_pov)).mean() < 0.01:
                    invalid_sample = True
                    # reset env to get min number of frames
                    break

                last_pov = cur_pov

                if imageio.imwrite(str(frames_dir_path.joinpath(str(i))) + ".png", cur_pov):
                    raise Exception("Could not save image")

                action.update(act)
                sample_actions.append(action)

                i += 1

                obs, reward, done, info = env.step(action)
        except RuntimeError as e:
            print(e)
        finally:
            if invalid_sample:
                shutil.rmtree(sample_dir_path)
            else:
                np.save(str(sample_dir_path.joinpath("actions.npy")), np.array(sample_actions))
                sample += 1
