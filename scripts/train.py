from stable_baselines3.common.vec_env import SubprocVecEnv
from const import *
from stable_baselines3 import PPO
import os
from callbacks import TrainAndLoggingCallback
from utils import make_env, linear_schedule

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "train")
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
    
def main():
    game = 'StreetFighterIISpecialChampionEdition-Genesis-v0'
    state = "/home/master26/Documents/StreetFighter2RL/FightLadder/data/sf/curriculum/Level2.20.state"
    side = "left" # side for AI to control
    reset_type = "match"
    rendering = False
    enable_combo = True # enable special move action space for environment
    null_combo = False # null action space for special move
    transform_action = False
    num_stack = 12 # number of frames to stack
    num_step_frames = 8 # number of frames per step
    num_timesteps = 20_000_000
    num_envs = 8 # number of parallel envs
    lr_schedule = linear_schedule(2.5e-4, 2.5e-6)
    clip_range_schedule = linear_schedule(0.15, 0.025)
    winrate_buffer_size = 1000 # sample size for computing winrate rolling average

    env = make_env(game, state, side, reset_type, rendering, enable_combo, null_combo, transform_action, num_stack, num_step_frames)
    env = SubprocVecEnv([env]*num_envs)

    model = PPO(
        "CnnPolicy", 
        env,
        device="cuda", 
        verbose=0,
        n_steps=1024, # 
        batch_size=2048, # multiple of n_steps
        gamma=0.94,
        learning_rate=lr_schedule,
        clip_range=clip_range_schedule,
        tensorboard_log=LOG_DIR,
    )


    callback = TrainAndLoggingCallback(save_path=CHECKPOINT_DIR, 
                                       winrate_buffer_size=winrate_buffer_size)
    
    model.learn(total_timesteps=num_timesteps, callback=callback)

if __name__ == "__main__":
    main()
