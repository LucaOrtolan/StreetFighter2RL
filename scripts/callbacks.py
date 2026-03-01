import os
from collections import deque
from stable_baselines3.common.callbacks import BaseCallback
from const import *


class TrainAndLoggingCallback(BaseCallback):
    """To visualize train stats in Tensorboard, run the script from the terminal and run in parallel:
            $ tensorboard --logdir=logs
    """

    def __init__(self, save_path, verbose=1, delete_previous_model=False, winrate_buffer_size=100, improvement_threshold=0.025):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.save_path = save_path
        self.delete_previous_model = delete_previous_model
        self.winrate_buffer_size = winrate_buffer_size
        self.improvement_threshold = improvement_threshold
        self.last_model_path = None  # track previous checkpoint
        self.episode_wins = deque(maxlen=winrate_buffer_size)   # buffer for calculating global winrate
        self.episode_count = 0
        self.last_saved_best_winrate = 0.0

    def _init_callback(self):
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

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

                    winrate = sum(self.episode_wins) / len(self.episode_wins)
                    self.logger.record("rollout/ep_winrate", winrate)

                    # update winrate per matchup
                    enemy_char = info.get("enemy_character")
                    self.matchups_wins[enemy_char].append(outcome_cat)
                    matchup_winrate = sum(self.matchups_wins[enemy_char]) / len(self.matchups_wins[enemy_char])
                    self.logger.record(f"rollout/ep_winrate_{CHARACTER_MAPPING[enemy_char]}", matchup_winrate)

                    # should save condition
                    should_save = (winrate >= 0.99) or (winrate >= self.last_saved_best_winrate + self.improvement_threshold)

                    if should_save:
                        model_path = os.path.join(self.save_path, f'best_model_winrate_{winrate:.3f}_{self.episode_count}')
                        self.model.save(model_path)

                        # Delete previous model if it exists
                        if self.delete_previous_model and self.last_model_path is not None and os.path.exists(self.last_model_path):
                            os.remove(self.last_model_path)

                        # Update pointer to new model
                        self.last_model_path = model_path + ".zip"

                        # Update best winrate
                        self.last_saved_best_winrate = winrate

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


# class CurriculumLearningCallback(TrainAndLoggingCallback):
#     def __init__(self, save_path, curriculum:dict, verbose=1, delete_previous_model=False, 
#                  winrate_buffer_size=100, winrate_threshold=0.9, improvement_threshold=0.025):
#         # Pass ALL required parent parameters
#         super(CurriculumLearningCallback, self).__init__(
#             save_path=save_path,
#             verbose=verbose,
#             delete_previous_model=delete_previous_model,
#             winrate_buffer_size=winrate_buffer_size,
#             improvement_threshold=improvement_threshold
#         )
        
#         self.curriculum = curriculum
#         self.winrate_threshold = winrate_threshold

#     def _on_step(self):
#         self.logger.record("rollout/episode_count", self.episode_count)

#         # Winrate tracking - check episode terminations
#         if self.locals.get("infos"):
#             for info in self.locals["infos"]:
#                 # Check if episode ended (works with both Gymnasium and Gym APIs)
#                 if info.get("episode"):
#                     self.episode_count += 1
#                     # Get win indicator from info (ensure SFWrapper sets this)
#                     outcome = info.get("outcome")
#                     # if isinstance(outcome, str):
#                     outcome_cat = 1.0 if outcome == "win" else 0.0
#                     self.episode_wins.append(outcome_cat)

#                     # Log winrate every len(buffer) episodes minimum
#                     if len(self.episode_wins) >= self.winrate_buffer_size:
#                         winrate = sum(self.episode_wins) / len(self.episode_wins)
#                         self.logger.record("rollout/ep_winrate", winrate)

#                         should_save = (winrate >= 0.99) or (winrate >= self.last_saved_best_winrate + self.improvement_threshold)

#                         if should_save:
#                             model_path = os.path.join(self.save_path, f'best_model_winrate_{winrate:.3f}_{self.n_calls}')
#                             self.model.save(model_path)

#                             # Delete previous model if it exists
#                             if self.delete_previous_model and self.last_model_path is not None and os.path.exists(self.last_model_path):
#                                 os.remove(self.last_model_path)

#                             # Update pointer to new model
#                             self.last_model_path = model_path + ".zip"

#                             # Update best winrate
#                             self.last_saved_best_winrate = winrate

#                             # If winrate threshold is achieved, stop training
#                             if winrate >= self.winrate_threshold:
#                                 return False

#         return True
