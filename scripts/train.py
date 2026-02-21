import stable_retro as retro
from wrapper import SFWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from const import *
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import time
import os
from collections import deque

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "train")
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")


def make_env(game, state, side, reset_type, rendering, init_level=1, state_dir=None, verbose=False, enable_combo=True, null_combo=False, transform_action=False, num_stack=12, num_step_frames=8):
    def _init():
        env = retro.make(
            game=game, 
            state=state, 
            use_restricted_actions=retro.Actions.FILTERED,
            render_mode = rendering,
            obs_type=retro.Observations.IMAGE,
        )
        env = SFWrapper(env, side=side, rendering=rendering, reset_type=reset_type, init_level=init_level, state_dir=state_dir, verbose=verbose, enable_combo=enable_combo, null_combo=null_combo, transform_action=transform_action, num_stack=num_stack, num_step_frames=num_step_frames)
        env = Monitor(env, LOG_DIR, info_keywords=("matches_won", "enemy_matches_won", "health", "enemy_health"))
        return env
    return _init

class TrainAndLoggingCallback(BaseCallback):
    """To visualize in Tensorboard, run the script from the terminal and run in parallel:
            $ tensorboard --logdir=logs
    """

    def __init__(self, check_freq, save_path, verbose=1, delete_previous_model=False, winrate_buffer_size=100):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.delete_previous_model = delete_previous_model
        self.winrate_buffer_size = winrate_buffer_size
        self.last_model_path = None  # track previous checkpoint
        self.episode_wins = deque(maxlen=winrate_buffer_size)
        self.episode_count = 0

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
        self.start_time = time.time()  # Record start time

    def _on_step(self):
        # Winrate tracking - check episode terminations
        if self.locals.get("infos"):
            for info in self.locals["infos"]:
                # Check if episode ended (works with both Gymnasium and Gym APIs)
                if info.get("episode"):
                    self.episode_count += 1
                    # Get win indicator from info (ensure SFWrapper sets this)
                    outcome = info.get("outcome")
                    # if isinstance(outcome, str):
                    outcome_cat = 1.0 if outcome == "win" else 0.0
                    self.episode_wins.append(outcome_cat)

                    # Log winrate every len(buffer) episodes minimum
                    if len(self.episode_wins) >= self.winrate_buffer_size:
                        winrate = sum(self.episode_wins) / len(self.episode_wins)
                        self.logger.record("rollout/ep_winrate", winrate)
                    self.logger.record("rollout/episode_count", self.episode_count)

        if self.n_calls % self.check_freq == 0:
            # save new checkpoint
            model_path = os.path.join(
                self.save_path,
                f'best_model_{self.n_calls}'
            )
            self.model.save(model_path)

            # delete previous checkpoint if it exists
            if self.delete_previous_model:
                if self.last_model_path is not None and os.path.exists(self.last_model_path):
                    os.remove(self.last_model_path)

            # update pointer
            self.last_model_path = model_path


        return True
    
def main():
    game = 'StreetFighterIISpecialChampionEdition-Genesis-v0'
    state = "ken"
    side = "left" # side for AI to control
    reset_type = "match"
    rendering = False
    verbose = True
    enable_combo = True # enable special move action space for environment
    null_combo = False # null action space for special move
    transform_action = False
    num_stack = 12 # number of frames to stack
    num_step_frames = 8 # number of frames per step
    num_timesteps = 1_000_000

    env = SubprocVecEnv([make_env(game, state, side, reset_type, rendering, enable_combo, null_combo, transform_action, num_stack, num_step_frames)])

    model = PPO(
        "CnnPolicy", 
        env,
        device="cuda", 
        verbose=1,
        n_steps=512,
        batch_size=1024, # 512,
        n_epochs=4,
        gamma=0.94,
        tensorboard_log=LOG_DIR,
    )

    callback = TrainAndLoggingCallback(check_freq=100_000, 
                                       save_path=CHECKPOINT_DIR, 
                                       winrate_buffer_size=100)
    model.learn(total_timesteps=num_timesteps, callback=callback)

if __name__ == "__main__":
    main()
