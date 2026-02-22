import os
from collections import deque
from stable_baselines3.common.callbacks import BaseCallback


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
