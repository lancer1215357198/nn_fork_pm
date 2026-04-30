import glob
import os
from typing import Callable, Optional

import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from custom_env import AirSimMazeEnv

# === 路径配置 ===
MODELS_DIR = r"D:\Others\MyAirsimprojects\models"
LOG_DIR = r"D:\Others\MyAirsimprojects\airsim_logs"

# === 训练超参数 ===
RANDOM_SEED = 42
INITIAL_LEARNING_RATE = 0.0003
BATCH_SIZE = 256
N_STEPS = 2048
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
ENT_COEF = 0.01
TOTAL_TIMESTEPS = 1_000_000
CHECKPOINT_FREQ = 20_000

# === 网络架构 ===
HIDDEN_LAYER_SIZE = 256
NUM_HIDDEN_LAYERS = 2

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """Create linear learning rate schedule.

    Args:
        initial_value: Initial learning rate.

    Returns:
        Function that computes learning rate based on progress.
    """

    def func(progress_remaining: float) -> float:
        # progress_remaining goes from 1.0 (start) to 0.0 (end)
        return progress_remaining * initial_value

    return func


def get_latest_model_path(path_dir: str) -> Optional[str]:
    """Get path to the most recently created model file.

    Args:
        path_dir: Directory to search for model files.

    Returns:
        Path to latest model file, or None if no models found.
    """
    list_of_files = glob.glob(os.path.join(path_dir, '*.zip'))
    if not list_of_files:
        return None
    return max(list_of_files, key=os.path.getctime)


def main() -> None:
    """Main training function."""
    set_random_seed(RANDOM_SEED)

    # Create and wrap environment
    env = DummyVecEnv([lambda: AirSimMazeEnv()])
    env = VecMonitor(env)

    latest_model_path = get_latest_model_path(MODELS_DIR)

    # Network architecture configuration
    policy_kwargs = dict(
        activation_fn=th.nn.Tanh,
        net_arch=dict(
            pi=[HIDDEN_LAYER_SIZE] * NUM_HIDDEN_LAYERS,
            vf=[HIDDEN_LAYER_SIZE] * NUM_HIDDEN_LAYERS
        )
    )

    if latest_model_path:
        print(f"--- 发现存档: {latest_model_path}，继续训练 ---")
        model = PPO.load(latest_model_path, env=env, tensorboard_log=LOG_DIR)
        reset_timesteps = False
    else:
        print("--- 未发现存档，开始【从头训练】(优化版) ---")
        model = PPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            tensorboard_log=LOG_DIR,
            learning_rate=linear_schedule(INITIAL_LEARNING_RATE),
            batch_size=BATCH_SIZE,
            n_steps=N_STEPS,
            gamma=GAMMA,
            gae_lambda=GAE_LAMBDA,
            clip_range=CLIP_RANGE,
            ent_coef=ENT_COEF,
            policy_kwargs=policy_kwargs,
            device="auto"
        )
        reset_timesteps = True

    checkpoint_callback = CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=MODELS_DIR,
        name_prefix='drone_maze_opt'
    )

    print("🚀 优化版训练引擎启动...")
    print(f"配置: Linear LR={INITIAL_LEARNING_RATE}, Net=[{HIDDEN_LAYER_SIZE}]*{NUM_HIDDEN_LAYERS}, Ent={ENT_COEF}")

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=checkpoint_callback,
        reset_num_timesteps=reset_timesteps
    )

    model.save(os.path.join(MODELS_DIR, "drone_maze_final_opt"))
    print("训练结束。")


if __name__ == "__main__":
    main()