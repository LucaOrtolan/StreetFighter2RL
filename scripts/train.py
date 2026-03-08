"""
train.py — Main training script for the Street Fighter II RL agent.

Sets up multiple parallel environments (one per matchup), constructs a PPO
model with a linearly annealed learning rate and clip range, then runs training
with TensorBoard logging, checkpointing, and early stopping.

Usage
-----
    python train.py

TensorBoard
-----------
    tensorboard --logdir=../logs
"""

from stable_baselines3.common.vec_env import SubprocVecEnv
from const import *
from stable_baselines3 import PPO
import os
from callbacks import TrainAndLoggingCallback, EarlyStoppingCallback
from utils import make_env, linear_schedule

# ---------------------------------------------------------------------------
# Directory paths (relative to the project root, one level above this file)
# ---------------------------------------------------------------------------

# Where EarlyStoppingCallback saves the best model checkpoints.
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "train")

# Where TensorBoard logs are written.
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")

# Directory containing the .state save-files that define each matchup.
STATE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/states")


def main():
    game = 'StreetFighterIISpecialChampionEdition-Genesis-v0'
    side = "left" # side for AI to control
    reset_type = "match"  # Episode ends after a full best-of-3 match.
    rendering = False  # No display window during training (faster)
    enable_combo = True # enable special move action space for environment
    null_combo = False # null action space for special move # If True, combo bits are present but always no-op.
    transform_action = False # Use raw MultiBinary action space.
    num_stack = 12 # number of frames to stack
    num_step_frames = 8 # number of emulator frames per step

    # Total number of emulator frames (across all parallel envs) to train for.
    # 100M steps with 10 parallel envs ≈ 10M agent updates.
    num_timesteps = 100_000_000

    # ---------------------------------------------------------------------------
    # Learning rate and clip range schedules
    # Both use linear annealing: start high for exploration, end low for stability.
    # linear_schedule(initial, final) returns a callable f(progress) → float
    # where progress decreases from 1.0 → 0.0 over training.
    # ---------------------------------------------------------------------------
    lr_schedule = linear_schedule(5.0e-5, 2.5e-6)
    clip_range_schedule = linear_schedule(0.075, 0.025)

    winrate_buffer_size = 1000 # sample size for computing winrate rolling average
    patience = 2000 # n° of rollouts to wait for improvements 
    min_improvement = 0.01 # minimum improvement for patience check

    # ---------------------------------------------------------------------------
    # Parallel environment setup
    # Each matchup is trained simultaneously; n_envs = sum(envs_per_matchup.values()).
    # Using multiple envs per matchup provides diversity within each opponent type.
    # The "_8" suffix in state filenames corresponds to arcade difficulty level 8.
    # ---------------------------------------------------------------------------

    envs_per_matchup = {"ryu_vs_ken_8" : 2,   # 2 workers fighting Ken on difficulty 8
                        "ryu_vs_guile_8": 2,
                        "ryu_vs_chunli_8": 2,
                        "ryu_vs_ryu_8": 2,
                        "ryu_vs_sagat_8": 2}   

    # Build the list of factory functions for SubprocVecEnv.
    # Each call to make_env() returns a closure (_init) that SubprocVecEnv
    # will call inside each worker subprocess.
    env_list = []
    for matchup, n_envs in envs_per_matchup.items():
        for _ in range(n_envs):
            state = STATE_DIR + f"/{matchup}.state"
            sub_env = make_env(game, state, side, reset_type, rendering, enable_combo, null_combo, transform_action, num_stack, num_step_frames)
            env_list.append(sub_env)

    # Launch all worker subprocesses; each runs its own copy of the environment.
    env = SubprocVecEnv(env_list)

    model = PPO(
        "CnnPolicy", # Convolutional policy (suitable for pixel observations).
        env,
        device="cuda", 
        verbose=1,
        # n_steps: number of environment steps collected per rollout *per env*.
        # Total samples per update = n_steps × n_envs = 512 × 10 = 5120.
        n_steps=512,
        # batch_size must be a divisor of (n_steps × n_envs).
        # 5120 / 1024 = 5 mini-batches per epoch.
        batch_size=1024, # multiple of n_steps
        gamma=0.94, # Discount factor (lower = more myopic; suits short fights).
        n_epochs=4,  # Number of optimisation passes per rollout buffer.
        learning_rate=lr_schedule,
        clip_range=clip_range_schedule,
        tensorboard_log=LOG_DIR,
    )
    # Log win-rate statistics to TensorBoard.
    logging_callback = TrainAndLoggingCallback(winrate_buffer_size=winrate_buffer_size)

    # Save best model checkpoints and stop training on plateau.
    early_stopping_callback = EarlyStoppingCallback(save_path=CHECKPOINT_DIR,
                                                    delete_previous_best_model=True,
                                                    patience=patience,
                                                    min_improvement=min_improvement)
    
    model.learn(total_timesteps=num_timesteps, callback=[logging_callback, early_stopping_callback])

if __name__ == "__main__":
    main()
