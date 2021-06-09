import minerl
import gym
import cv2
from pathlib import Path
import numpy as np

root_dataset_dir_path = Path("mc_predefinied_agent_actions")

env = gym.make("MineRLNavigate-v0")


# minerl-action-name, action-value, action-folder-name, number-of-samples, max-number-of-frames
dataset_actions = [
    ("camera", np.array([5, 0]), "camera_pitch_5", 5, 18),
    ("camera", np.array([10, 0]), "camera_pitch_10", 5, 9),
    ("camera", np.array([-5, 0]), "camera_pitch_-5", 5, 18),
    ("camera", np.array([0, 5]), "camera_yaw_5", 5, 80),
    ("camera", np.array([0, 10]), "camera_yaw_10", 10, 50),
    ("camera", np.array([0, 90]), "camera_yaw_90", 5, 10),
    ("forward", 1, "forward", 100, 1000),
    ("left", 1, "left", 100, 1000),
    ("right", 1, "right", 100, 1000),
    ("jump", 1, "jump", 10, 50)
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
                if not cv2.imwrite(str(frames_dir_path.joinpath(str(i))) + ".jpg", obs["pov"]):
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

