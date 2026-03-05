import os
from collections import deque
from stable_baselines3.common.callbacks import BaseCallback
from utils import make_env
from const import *


class TrainAndLoggingCallback(BaseCallback):
    """To visualize train stats in Tensorboard, run the script from the terminal and run in parallel:
            $ tensorboard --logdir=logs
    """

    def __init__(self, verbose=1, winrate_buffer_size=100):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.winrate_buffer_size = winrate_buffer_size
        self.last_model_path = None  # track previous checkpoint
        self.episode_wins = deque(maxlen=winrate_buffer_size)   # buffer for calculating global winrate
        self.episode_count = 0

    def _init_callback(self):
        self.logger.record("train/batch_size", self.model.batch_size)
        self.logger.record("train/n_envs", self.model.n_envs)
        self.logger.record("train/n_steps", self.model.n_steps)
        self.logger.record("train/gamma", self.model.gamma)
        self.logger.record("train/n_epochs", self.model.n_epochs)

    def _on_step(self):
        self.logger.record("rollout/episode_count", self.episode_count)
        # create separate buffers for tracking winrate for each matchup
        if self.n_calls ==1:
            self.matchups_wins = {}
            for info in self.locals["infos"]:
                if info["enemy_character"] not in self.matchups_wins.keys():
                    self.matchups_wins[info["enemy_character"]] = deque(maxlen=self.winrate_buffer_size)

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

                    if self.episode_count >= self.winrate_buffer_size:
                        winrate = sum(self.episode_wins) / len(self.episode_wins)
                        self.logger.record("rollout/ep_winrate", winrate)

                        # update winrate per matchup
                        enemy_char = info.get("enemy_character")
                        if enemy_char not in self.matchups_wins.keys():
                            self.matchups_wins[enemy_char] = deque(maxlen=self.winrate_buffer_size)
                        
                        self.matchups_wins[enemy_char].append(outcome_cat)
                        matchup_winrate = sum(self.matchups_wins[enemy_char]) / len(self.matchups_wins[enemy_char])
                        self.logger.record(f"rollout/ep_winrate_{CHARACTER_MAPPING[enemy_char]}", matchup_winrate)

        return True

class EarlyStoppingCallback(BaseCallback):
    """Early stopping if there are no improvements in mean reward per episode after a certain number of rollouts"""

    def __init__(self, save_path, verbose=1, patience=1000, min_improvement=0.1, delete_previous_best_model=False):
        super().__init__(verbose)
        self.save_path = save_path
        self.patience = patience
        self.best_mean_reward = 0
        self.best_model_path = None
        self.delete_previous_best_model = delete_previous_best_model
        self.patience_counter = 0
        self.min_improvement = min_improvement
        self.current_mean_reward = None

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_rollout_end(self):
        try:
            self.current_mean_reward = sum([ep_info["r"] for ep_info in self.model.ep_info_buffer])/len([ep_info["r"] for ep_info in self.model.ep_info_buffer])
        except ZeroDivisionError:
            pass

        # Avoid start saving or counting if the winrate isn't available yet
        if self.logger.name_to_value.get("rollout/ep_winrate") is None:
            pass

        # Check if current mean reward is better than best so far
        if self.current_mean_reward >= self.best_mean_reward + self.min_improvement:
            self.best_mean_reward = self.current_mean_reward
            self.patience_counter = 0  # reset patience counter

            if self.best_model_path is not None:
                if os.path.exists(self.best_model_path) and self.delete_previous_best_model:
                    os.remove(self.best_model_path)
            
            # Save new best model
            current_winrate = self.logger.name_to_value.get("rollout/ep_winrate")
            num_episodes = self.logger.name_to_value.get("rollout/episode_count")

            self.best_model_path = os.path.join(self.save_path, f"best_model_winrate_{current_winrate}_episode_{num_episodes}.zip")
            self.model.save(self.best_model_path)

            if self.verbose > 0:
                print(
                    f"[EarlyStopping] New best ep_rew_mean={self.best_mean_reward:.4f}, "
                    f"saved model to {self.best_model_path}"
                )
        
        # Handle no improvement case
        else: 
            self.patience_counter += 1
    
            
    def _on_step(self):
        # start logging and winrate becomes available
        if self.logger.name_to_value.get("rollout/ep_winrate") is not None:
            self.logger.record("rollout/last_saved_mean_reward", self.best_mean_reward)
            self.logger.record("rollout/patience_counter", self.patience_counter)

    
        # Stop training if (num_episodes since last save) >= patience
        if self.patience_counter >= self.patience:
            if self.verbose > 0:
                print(
                    f"[EarlyStopping] Patience {self.patience} exceeded. "
                    f"Stopping training. Best model: {self.best_model_path}"
                )

            return False
        
        return True 

class OptunaTrainAndLoggingCallback(BaseCallback):
    """Callback for Optuna trials with TensorBoard logging, winrate tracking, and checkpointing."""

    def __init__(self, winrate_buffer_size=50, verbose=1):
        super(OptunaTrainAndLoggingCallback, self).__init__(verbose)
        self.winrate_buffer_size = winrate_buffer_size
        self.episode_wins = deque(maxlen=winrate_buffer_size)
        self.episode_count = 0
        self.last_model_path = None

    def _on_step(self):
        # Winrate tracking using Monitor's episode info
        if self.locals.get("infos"):
            for info in self.locals["infos"]:
                if info.get("episode"):
                    self.episode_count += 1
                    outcome = info.get("outcome", "lose")
                    outcome_cat = 1.0 if outcome == "win" else 0.0
                    self.episode_wins.append(outcome_cat)

                    if len(self.episode_wins) >= self.winrate_buffer_size:
                        winrate = sum(self.episode_wins) / len(self.episode_wins)
                        self.logger.record("rollout/ep_winrate", winrate)
                    self.logger.record("rollout/episode_count", self.episode_count)
        return True

class LearningRateCallback(BaseCallback):
    """Adjusts Learning Rate based on the mean reward per episode"""

    def __init__(self, factor=0.9, patience=100, min_improvement=0.001, verbose=1):
        super(LearningRateCallback, self).__init__(verbose)
        self.min_improvement = min_improvement
        self.patience = patience
        self.factor = factor
        self.best_mean_reward = 0
        self.last_reduce_episode = 0

    def _on_step(self):
        if "rollout/ep_rew_mean" in self.logger.name_to_value:
            current_mean_reward = self.logger.name_to_value["rollout/ep_rew_mean"]

            # Update current best mean reward and reset patience count
            if current_mean_reward >= self.best_mean_reward + self.min_improvement:
                self.best_mean_reward = current_mean_reward
                self.last_reduce_episode = self.logger.name_to_value["rollout/episode_count"]
           
        # Check patience and update lr
        if self.logger.name_to_value["rollout/episode_count"] - self.last_reduce_episode >= self.patience:
            self.model.learning_rate *= self.factor
            self.model._setup_lr_schedule()

            self.last_reduce_episode = self.logger.name_to_value["rollout/episode_count"]

        return True

class CurriculumLearningCallback(BaseCallback):

    def __init__(self, curriculum, target_winrate=.9, verbose=1):
        super().__init__(verbose)
        self.curriculum = curriculum
        self.target_winrate = target_winrate

    def _on_rollout_end(self):
        if self.logger.name_to_value.get("rollout/episode_count") > 10:
            path = "/home/master26/Documents/StreetFighter2RL/data/states/ryu_vs_zangief_8.state"
            states = [path]
            
            self.locals["env"].update_env(states, [0])

    def _on_step(self):
        return True