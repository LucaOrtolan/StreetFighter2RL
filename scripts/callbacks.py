"""
callbacks.py — Stable-Baselines3 training callbacks for the SF2 RL project.

Callbacks are called at fixed points during the PPO training loop to perform
logging, checkpointing, early stopping, and learning-rate scheduling.

Available callbacks
-------------------
TrainAndLoggingCallback      – TensorBoard logging of winrate (global and per matchup).
EarlyStoppingCallback        – Saves best model checkpoint and stops training on plateau.
OptunaTrainAndLoggingCallback – Lightweight version for Optuna hyperparameter search.
LearningRateCallback         – Reduces the learning rate on plateau.
CurriculumLearningCallback   – (WIP) Updates the environment state for curriculum learning.
"""

import os
from collections import deque
from stable_baselines3.common.callbacks import BaseCallback
from utils import make_env
from const import *
import torch


class TrainAndLoggingCallback(BaseCallback):
    """
    Logs training statistics to TensorBoard.

    Tracks episode win-rate (globally and broken down per enemy character)
    using a rolling window of the last `winrate_buffer_size` episodes.

    To view logs in TensorBoard, run in a separate terminal:
        $ tensorboard --logdir=logs

    Parameters
    ----------
    verbose : int
        Verbosity level (inherited from BaseCallback).
    winrate_buffer_size : int
        Number of recent episodes to include in the rolling win-rate average.
        Larger values = smoother curve but slower to react to improvements.
    """

    def __init__(self, verbose=1, winrate_buffer_size=100):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.winrate_buffer_size = winrate_buffer_size
        self.last_model_path = None             # Path of the most recently saved checkpoint.
        # Circular buffer: stores 1.0 for a win, 0.0 for a non-win per episode.
        self.episode_wins = deque(maxlen=winrate_buffer_size)
        self.episode_count = 0

    def _init_callback(self):
        """Log static hyperparameters once at the start of training."""
        self.logger.record("train/batch_size", self.model.batch_size)
        self.logger.record("train/n_envs",     self.model.n_envs)
        self.logger.record("train/n_steps",    self.model.n_steps)
        self.logger.record("train/gamma",      self.model.gamma)
        self.logger.record("train/n_epochs",   self.model.n_epochs)

    def _on_step(self):
        """
        Called after every environment step across all parallel envs.

        On the very first call, creates per-matchup win-rate buffers so that
        each enemy character gets its own rolling average tracked separately.

        On every call, inspects `infos` (a list with one dict per parallel env)
        and records completed episode outcomes.
        """
        self.logger.record("rollout/episode_count", self.episode_count)

        # First call: discover all active matchups from initial info dicts
        # and create an empty deque buffer for each one.
        if self.n_calls == 1:
            self.matchups_wins = {}
            for info in self.locals["infos"]:
                enemy = info["enemy_character"]
                if enemy not in self.matchups_wins:
                    self.matchups_wins[enemy] = deque(maxlen=self.winrate_buffer_size)

        # Check every parallel environment's info dict for episode terminations.
        if self.locals.get("infos"):
            for info in self.locals["infos"]:
                # SB3's Monitor wrapper adds an "episode" key to info only when
                # an episode has just finished (compatible with both Gym APIs).
                if info.get("episode"):
                    self.episode_count += 1
                    outcome = info.get("outcome")
                    # Convert outcome string to binary: 1.0 = win, 0.0 = anything else.
                    outcome_cat = 1.0 if outcome == "win" else 0.0
                    self.episode_wins.append(outcome_cat)

                    # Only report win-rate once the buffer is fully populated
                    # to avoid noisy estimates from tiny samples.
                    if self.episode_count >= self.winrate_buffer_size:
                        winrate = sum(self.episode_wins) / len(self.episode_wins)
                        self.logger.record("rollout/ep_winrate", winrate)

                        # Also update the per-matchup win-rate buffer.
                        enemy_char = info.get("enemy_character")
                        if enemy_char not in self.matchups_wins:
                            self.matchups_wins[enemy_char] = deque(maxlen=self.winrate_buffer_size)
                        self.matchups_wins[enemy_char].append(outcome_cat)

                        matchup_winrate = (
                            sum(self.matchups_wins[enemy_char])
                            / len(self.matchups_wins[enemy_char])
                        )
                        # Log under e.g. "rollout/ep_winrate_ryu", "rollout/ep_winrate_ken", etc.
                        self.logger.record(
                            f"rollout/ep_winrate_{CHARACTER_MAPPING[enemy_char]}",
                            matchup_winrate,
                        )

        return True  # Returning False would stop training; True continues.


class EarlyStoppingCallback(BaseCallback):
    """
    Saves the best model checkpoint and stops training when it stops improving.

    After each rollout collection phase, computes the mean episode return from
    SB3's internal `ep_info_buffer`.  If the mean return does not improve by at
    least `min_improvement` within `patience` rollouts, training is halted.

    Best model checkpoints are named:
        best_model_winrate_<winrate>_episode_<n>.zip

    Parameters
    ----------
    save_path : str
        Directory where checkpoint .zip files are saved.
    verbose : int
        Verbosity level.
    patience : int
        Number of rollouts to wait without improvement before stopping.
    min_improvement : float
        Minimum absolute increase in mean reward to count as an improvement.
    delete_previous_best_model : bool
        If True, delete the previous best checkpoint when a new one is saved
        (saves disk space during long runs).
    """

    def __init__(
        self,
        save_path,
        verbose=1,
        patience=1000,
        min_improvement=0.1,
        delete_previous_best_model=False,
    ):
        super().__init__(verbose)
        self.save_path = save_path
        self.patience = patience
        self.best_mean_reward = 0
        self.best_model_path = None
        self.delete_previous_best_model = delete_previous_best_model
        self.patience_counter = 0   # Rollouts elapsed since the last improvement.
        self.min_improvement = min_improvement
        self.current_mean_reward = None

    def _init_callback(self):
        """Create the checkpoint directory if it doesn't exist."""
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_rollout_end(self):
        """
        Called at the end of each rollout collection phase (before the policy update).

        Computes mean episode reward from SB3's internal buffer and decides
        whether to save a new checkpoint or increment the patience counter.
        """
        try:
            # ep_info_buffer contains dicts with key "r" (episode return)
            # for all recently completed episodes across all parallel envs.
            self.current_mean_reward = (
                sum(ep["r"] for ep in self.model.ep_info_buffer)
                / len(self.model.ep_info_buffer)
            )
        except ZeroDivisionError:
            pass  # Buffer empty at the very start of training; skip.

        # Don't start tracking until win-rate is available (enough episodes done).
        if self.logger.name_to_value.get("rollout/ep_winrate") is None:
            return

        if self.current_mean_reward >= self.best_mean_reward + self.min_improvement:
            # New best model found.
            self.best_mean_reward = self.current_mean_reward
            self.patience_counter = 0  # Reset the patience clock.

            # Optionally remove the old best checkpoint to save disk space.
            if self.best_model_path is not None:
                if os.path.exists(self.best_model_path) and self.delete_previous_best_model:
                    os.remove(self.best_model_path)

            # Build a descriptive filename embedding current metrics.
            current_winrate  = self.logger.name_to_value.get("rollout/ep_winrate")
            num_episodes     = self.logger.name_to_value.get("rollout/episode_count")
            self.best_model_path = os.path.join(
                self.save_path,
                f"best_model_winrate_{current_winrate}_episode_{num_episodes}.zip",
            )
            self.model.save(self.best_model_path)

            if self.verbose > 0:
                print(
                    f"[EarlyStopping] New best ep_rew_mean={self.best_mean_reward:.4f}, "
                    f"saved model to {self.best_model_path}"
                )
        else:
            # No improvement: increment the patience counter.
            self.patience_counter += 1

    def _on_step(self):
        """
        Called after every environment step.

        Records diagnostic metrics to TensorBoard once logging is active,
        and halts training if patience is exhausted.
        """
        if self.logger.name_to_value.get("rollout/ep_winrate") is not None:
            self.logger.record("rollout/last_saved_mean_reward", self.best_mean_reward)
            self.logger.record("rollout/patience_counter", self.patience_counter)

        # Stop training if patience is exceeded (return False signals SB3 to stop).
        if self.patience_counter >= self.patience:
            if self.verbose > 0:
                print(
                    f"[EarlyStopping] Patience {self.patience} exceeded. "
                    f"Stopping training. Best model: {self.best_model_path}"
                )
            return False

        return True


class OptunaTrainAndLoggingCallback(BaseCallback):
    """
    Lightweight callback for Optuna hyperparameter search trials.

    Tracks win-rate with a smaller buffer than the full training callback
    to give Optuna a faster (albeit noisier) signal per trial.

    Parameters
    ----------
    winrate_buffer_size : int
        Rolling window size for win-rate estimation.  Smaller = faster
        Optuna convergence, but noisier objective signal.
    verbose : int
        Verbosity level.
    """

    def __init__(self, winrate_buffer_size=50, verbose=1):
        super(OptunaTrainAndLoggingCallback, self).__init__(verbose)
        self.winrate_buffer_size = winrate_buffer_size
        self.episode_wins = deque(maxlen=winrate_buffer_size)
        self.episode_count = 0
        self.last_model_path = None

    def _on_step(self):
        """
        Track episode outcomes and log rolling win-rate for Optuna trials.

        Same logic as TrainAndLoggingCallback._on_step() but without the
        per-matchup breakdown (not needed for Optuna's single objective).
        """
        if self.locals.get("infos"):
            for info in self.locals["infos"]:
                if info.get("episode"):
                    self.episode_count += 1
                    outcome = info.get("outcome", "lose")
                    outcome_cat = 1.0 if outcome == "win" else 0.0
                    self.episode_wins.append(outcome_cat)

                    # Report win-rate once the buffer has enough data.
                    if len(self.episode_wins) >= self.winrate_buffer_size:
                        winrate = sum(self.episode_wins) / len(self.episode_wins)
                        self.logger.record("rollout/ep_winrate", winrate)
                    self.logger.record("rollout/episode_count", self.episode_count)
        return True


class LearningRateCallback(BaseCallback):
    """
    Reduces the learning rate when mean episode reward stops improving.

    Implements a "reduce on plateau" schedule: if the mean reward does not
    improve by at least `min_improvement` within `patience` rollouts, the
    learning rate is multiplied by `factor` (< 1).

    Note: SB3's PPO uses a schedule callable for `learning_rate`.  After
    directly modifying `model.learning_rate`, we call `_setup_lr_schedule()`
    to rebuild the schedule so the optimiser picks up the new value.

    Parameters
    ----------
    factor : float
        Multiplicative factor applied to the LR on each reduction (e.g. 0.9).
    patience : int
        Number of episodes without improvement before reducing the LR.
    min_improvement : float
        Minimum absolute improvement in mean reward to reset patience.
    verbose : int
        Verbosity level.
    """

    def __init__(self, factor=0.9, patience=100, min_improvement=0.001, verbose=1):
        super(LearningRateCallback, self).__init__(verbose)
        self.min_improvement = min_improvement
        self.patience = patience
        self.factor = factor
        self.best_mean_reward = 0
        self.last_reduce_episode = 0  # Episode count at the time of the last LR reduction.

    def _on_step(self):
        """
        Check for improvement and reduce LR if stagnant.

        Reads the current mean reward and episode count directly from the
        TensorBoard logger's name→value dict (populated by SB3 after each
        rollout).
        """
        if "rollout/ep_rew_mean" in self.logger.name_to_value:
            current_mean_reward = self.logger.name_to_value["rollout/ep_rew_mean"]

            # Update best reward and record when we last improved.
            if current_mean_reward >= self.best_mean_reward + self.min_improvement:
                self.best_mean_reward = current_mean_reward
                self.last_reduce_episode = self.logger.name_to_value["rollout/episode_count"]

        # If we haven't improved in `patience` episodes, reduce the LR.
        current_ep = self.logger.name_to_value["rollout/episode_count"]
        if current_ep - self.last_reduce_episode >= self.patience:
            self.model.learning_rate *= self.factor
            # Rebuild the LR schedule so the underlying optimiser uses the new value.
            self.model._setup_lr_schedule()
            # Reset the clock so we don't reduce again immediately.
            self.last_reduce_episode = current_ep

        return True


class CurriculumLearningCallback(BaseCallback):
    """
    (Work in Progress) Updates the environment save-state for curriculum learning.

    The idea: once the agent reaches a target win-rate against the current
    opponent, swap in a harder save-state so the agent faces a more challenging
    scenario.  Currently hardcoded to a single state path for development/testing.

    Parameters
    ----------
    curriculum : list
        Ordered list of (state_path, difficulty_info) tuples defining the curriculum.
    target_winrate : float
        Win-rate threshold at which to advance to the next curriculum stage.
    verbose : int
        Verbosity level.
    """

    def __init__(self, curriculum, target_winrate=0.9, verbose=1):
        super().__init__(verbose)
        self.curriculum = curriculum
        self.target_winrate = target_winrate

    def _on_rollout_end(self):
        """
        Called after each rollout: check if we should advance the curriculum.

        TODO: replace hardcoded path and index with dynamic curriculum logic
        that reads target_winrate and steps through self.curriculum.
        """
        if self.logger.name_to_value.get("rollout/episode_count") > 10:
            # Hardcoded state path — replace with dynamic curriculum advancement.
            path = "/home/master26/Documents/StreetFighter2RL/data/states/ryu_vs_zangief_8.state"
            states = [path]
            # update_env() sends a "update_state" command to the subprocess workers
            # (see SubprocVecEnvCL in utils.py) to hot-swap the save-state without
            # restarting the worker processes.
            self.locals["env"].update_env(states, [0])

    def _on_step(self):
        return True
