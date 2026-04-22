from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from const import *
from utils import make_env, get_state, linear_schedule, SubprocVecEnvCL
from callbacks import CurriculumLearningCallback, TrainAndLoggingCallback, EarlyStoppingCallback
import os

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "train/final_models")
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs/exp1")
STATE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data/states")


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
    num_timesteps = 300_000_000
    lr_schedule = linear_schedule(1.95e-5, 2.5e-6) # alternative approach  # orig: 5.0e-5, 2.5e-6
    clip_range_schedule = linear_schedule(0.0429, 0.025)  # orig: 0.075, 0.025
    winrate_buffer_size = 100 # sample size for computing winrate rolling average
    winrate_threshold = 0.97 # winrate threshold for curriculum learning (was 0.75)
    min_improvement = 0.01 # winrate threshold for saving new model
    patience = 2000
    delete_previous_model = True

    curriculum = {
        "ryu_vs_ken" : {
            "n_envs": 8,
            "levels": [8]
            },
        "ryu_vs_chunli": {
            "n_envs": 8,
            "levels": [8]
            },
        "ryu_vs_guile": {
            "n_envs": 8,
            "levels": [8]
            },
        "ryu_vs_ryu": {
            "n_envs": 8,
            "levels": [8]
            },
        }  

    env_list = []
    for matchup in curriculum.keys():
        char_curr = curriculum[matchup]
        for _ in range(char_curr["n_envs"]):
            state_to_load = get_state(matchup, char_curr["levels"][0])

            sub_env = make_env(game, state_to_load, side, reset_type, rendering, enable_combo=enable_combo, null_combo=null_combo,
                               transform_action=transform_action, num_stack=num_stack, num_step_frames=num_step_frames)

            # print(
            #     f"curriculum_learning.py make_env called with:\n"
            #     f"  game={game}\n"
            #     f"  state_to_load={state_to_load}\n"
            #     f"  side={side}\n"
            #     f"  reset_type={reset_type}\n"
            #     f"  rendering={rendering}\n"
            #     f"  enable_combo={enable_combo}\n"
            #     f"  null_combo={null_combo}\n"
            #     f"  transform_action={transform_action}\n"
            #     f"  num_stack={num_stack}\n"
            #     f"  num_step_frames={num_step_frames}\n"
            # )
            # # original
            # sub_env = make_env(game, state_to_load, side, reset_type, rendering, enable_combo, null_combo, transform_action, num_stack, num_step_frames)

            env_list.append(sub_env)

    env = SubprocVecEnvCL(env_list)

    # model = PPO(
    #     "CnnPolicy",
    #     env,
    #     device="cuda",
    #     verbose=1,
    #     n_steps=512,
    #     batch_size=1024, # multiple of n_steps
    #     gamma=0.94,
    #     n_epochs=4,
    #     learning_rate=lr_schedule,
    #     clip_range=clip_range_schedule,
    #     tensorboard_log=LOG_DIR,
    # )

    # This is a trained model
    model = PPO.load("/home/emeralddawns/Documents/StreetFighter2RL/train/final_models/final_model_difficulty_8.zip",
                     env=env,
                     device="cuda",
                     tensorboard_log=LOG_DIR)

    cl_callback = CurriculumLearningCallback(winrate_buffer_size=winrate_buffer_size, 
                                             save_path=CHECKPOINT_DIR,
                                             min_improvement=min_improvement,
                                             curriculum=curriculum, 
                                             winrate_threshold=winrate_threshold,
                                             patience=patience,
                                             delete_previous_model=delete_previous_model)
    
                                                
    model.learn(total_timesteps=num_timesteps, callback=[cl_callback])

    # export final model
    model_path = os.path.join(CHECKPOINT_DIR, f"final_model_winrate_{cl_callback.winrates['global'].get('winrate')}.zip")
    model.save(model_path)


if __name__ == "__main__":
    main()
