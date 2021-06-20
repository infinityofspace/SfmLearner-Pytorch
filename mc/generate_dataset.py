import minerl
import gym
import cv2
from pathlib import Path
import numpy as np
import imageio

root_dataset_dir_path = Path("mc_predefinied_agent_actions")

env = gym.make("MineRLTreechop-v0")


# minerl-action-name, action-value, action-folder-name, number-of-samples, max-number-of-frames
dataset_actions = [
    ("camera", np.array([0, 5]), "camera_yaw_5", 5, 73),
    ("camera", np.array([0, 10]), "camera_yaw_10", 5, 37),
    ("camera", np.array([0, 20]), "camera_yaw_20", 5, 19),
    ("camera", np.array([0, 30]), "camera_yaw_30", 5, 13),
]


for action_name, value, action_folder_name, n_samples, n_frames  in dataset_actions:
    for sample in range(n_samples):
        obs = env.reset()
        sample_dir_path = root_dataset_dir_path.joinpath(action_folder_name, str(sample))
        sample_dir_path.mkdir(parents=True)
        
        frames_dir_path = sample_dir_path.joinpath("frames")
        frames_dir_path.mkdir(parents=True)
        
        sample_actions = []
        
        i = 0
        action = env.action_space.noop()
        
        try:

            obs, reward, done, info = env.step(action)
            while i < n_frames and not done:
                if imageio.imwrite(str(frames_dir_path.joinpath(str(i))) + ".png", obs["pov"]):
                    raise Exception("Could not save image")
                                    
                action[action_name] = value
                sample_actions.append(action)
                
                i += 1
                
                obs, reward, done, info = env.step(action)
        except RuntimeError as e:
            print(e)
        finally:
            with open(str(sample_dir_path.joinpath("actions.npy")), "wb") as f:
                np.save(f, np.array(sample_actions))

