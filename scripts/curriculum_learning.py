from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from const import *
from utils import make_env, linear_schedule, SubprocVecEnvCL
from callbacks import CurriculumLearningCallback, TrainAndLoggingCallback
import os

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "train")
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
STATE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/states")

def get_state(character, lvl):
    state_to_load = STATE_DIR + f"/{CHARACTER_MAPPING[character]}.{lvl}.state"
    return state_to_load

def main():
    game = 'StreetFighterIISpecialChampionEdition-Genesis-v0'
    side = "left" # side for AI to control
    reset_type = "match"
    rendering = False
    enable_combo = True # enable special move action space for environment
    null_combo = False # null action space for special move
    transform_action = False
    num_stack = 12 # number of frames to stack
    num_step_frames = 8 # number of frames per step
    num_timesteps = 10_000_000
    lr_schedule = linear_schedule(2.5e-4, 2.5e-6) # alternative approach
    clip_range_schedule = linear_schedule(0.15, 0.025)
    winrate_buffer_size = 1 # sample size for computing winrate rolling average
    patience = 2000  # n° of rollouts to wait for improvements 
    min_improvement = 0.01  # minimum improvement for patience check

    curriculum = {
        "ryu_vs_ken" : {
            "n_envs": 2, 
            "levels": [i for i in range(1, 9)]
            },
        "ryu_vs_chunli": {
            "n_envs": 2,
            "levels": [i for i in range(1, 9)]
            },
        "ryu_vs_guile": {
            "n_envs": 2,
            "levels": [i for i in range(1, 9)]
            },
        "ryu_vs_ryu": {
            "n_envs": 2,
            "levels": [i for i in range(1, 9)]
            },
        "ryu_vs_sagat": {
            "n_envs": 2,
            "levels": [i for i in range(1, 9)]
            }
        }  


    # env_list = []
    # for character in curriculum.items():
    #     char_curr = curriculum[character]
    #     for _ in range(char_curr["n_envs"]):
    #         # state = get_state(character, char_curr["levels"][0])   
    #         state = "ken"
    #         sub_env = make_env(game, state, side, reset_type, rendering, enable_combo, null_combo, transform_action, num_stack, num_step_frames)
    #         env_list.append(sub_env)
    
    envs_per_matchup = {"ryu_vs_ken_8" : 2,           
                    "ryu_vs_guile_8": 2,
                    "ryu_vs_chunli_8": 2,
                    "ryu_vs_ryu_8": 2,
                    "ryu_vs_sagat_8": 2}   

    env_list = []
    for matchup, n_envs in envs_per_matchup.items():
        for _ in range(n_envs):
            state = STATE_DIR + f"/{matchup}.state"
            sub_env = make_env(game, state, side, reset_type, rendering, enable_combo, null_combo, transform_action, num_stack, num_step_frames)
            env_list.append(sub_env)

    env = SubprocVecEnvCL(env_list)

    model = PPO(
        "CnnPolicy", 
        env,
        device="cuda", 
        verbose=1,
        n_steps=512, 
        batch_size=1024, # multiple of n_steps
        gamma=0.94,
        n_epochs=4,
        learning_rate=lr_schedule,
        clip_range=clip_range_schedule,
        tensorboard_log=LOG_DIR,
    )

    logging_callback = TrainAndLoggingCallback(winrate_buffer_size=winrate_buffer_size)

    cl_callback = CurriculumLearningCallback(curriculum=curriculum)
    
    model.learn(total_timesteps=num_timesteps, callback=[logging_callback, cl_callback])

if __name__ == "__main__":
    main()
