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
import re
from collections import deque, Counter
from stable_baselines3.common.callbacks import BaseCallback
from utils import get_state, extract_matchup
from const import *
import pandas as pd
import numpy as np


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
        method = "either",
    ):
        super().__init__(verbose)
        self.save_path = save_path
        self.patience = patience
        self.best_mean_reward = 0
        self.best_winrate = 0
        self.best_model_path = None
        self.delete_previous_best_model = delete_previous_best_model
        self.patience_counter = 0
        self.min_improvement = min_improvement
        self.current_mean_reward = None
        self.current_winrate
        self.method = method


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
            self.current_winrate = self.logger.name_to_value.get("rollout/ep_winrate")
            self.current_mean_reward = sum([ep_info["r"] for ep_info in self.model.ep_info_buffer])/len([ep_info["r"] for ep_info in self.model.ep_info_buffer])
        except ZeroDivisionError:
            pass # Buffer empty at the very start of training; skip.

        # Don't start tracking until win-rate is available (enough episodes done).
        if self.current_winrate is None:
            return

        reset_patience = False
        if self.current_mean_reward >= self.best_mean_reward + self.min_improvement:
            self.best_mean_reward = self.current_mean_reward
            if self.method in ["mean_reward", "either"]:
                reset_patience = True

        if self.current_winrate >= self.best_winrate + self.min_improvement:
            self.best_winrate = self.current_winrate
            if self.method in ["winrate", "either"]:
                reset_patience = True

        if reset_patience:
            # New best model found.
            self.patience_counter = 0  # Reset the patience clock.

            # Optionally remove the old best checkpoint to save disk space.
            if self.best_model_path is not None:
                if os.path.exists(self.best_model_path) and self.delete_previous_best_model:
                    os.remove(self.best_model_path)

            # Build a descriptive filename embedding current metrics.
            num_episodes     = self.logger.name_to_value.get("rollout/episode_count")
            self.best_model_path = os.path.join(
                self.save_path,
                f"best_model_winrate_{self.current_winrate}_episode_{num_episodes}.zip",
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


class CurriculumLearningCallback(TrainAndLoggingCallback):
    """Multi functional Callback for implementing Curriculum Learning. Performs logging for every metric, state update and saves best model."""

    def __init__(self, curriculum, save_path, winrate_threshold=.9, winrate_buffer_size=100, min_improvement=0.01,
                 patience=1000, delete_previous_model=True, verbose=1):
        super(CurriculumLearningCallback, self).__init__(verbose, winrate_buffer_size=winrate_buffer_size)
        ###### curriculum parameters
        self.curriculum = curriculum
        self.winrate_threshold = winrate_threshold
        self.cooldown_duration = winrate_buffer_size  # must be equal to winrate buffer size
        self.last_level_reached = False
        self.max_level = 8
        self.cooldown = {k: self.cooldown_duration for k in self.curriculum.keys()}  # cooldown is computed per matchup

        ###### early stopping parameters
        self.save_path = save_path
        self.best_model_path = None
        self.min_improvement = min_improvement
        self.patience = patience
        self.patience_counter = 0
        self.delete_previous_model = delete_previous_model
        self.best_mean_reward = 0

    def _init_callback(self):
        super()._init_callback()
        for matchup in self.curriculum.keys():
            self.curriculum[matchup]["current_lvl_idx"] = 0  # keeps track of the index of the current level

        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_rollout_end(self):
        current_winrate = self.winrates["global"].get("winrate")
        if current_winrate is not None:
            # make sure all matchups have played enough episodes
            if all(v == 0 for v in self.cooldown.values()):
                # get current difficulty for model save
                # check current level idx
                sample_matchup = list(self.curriculum.keys())[0]
                current_lvl_idx = self.curriculum[sample_matchup]["current_lvl_idx"]
                current_difficulty = self.curriculum[sample_matchup]["levels"][
                    self.curriculum[sample_matchup]["current_lvl_idx"]]

                # reallocate envs
                self.reallocate_envs()

                # raise difficulty and export model is winrate threshold is met
                if (current_winrate >= self.winrate_threshold) & (
                        current_lvl_idx < len(self.curriculum[sample_matchup]["levels"])):
                    # export model
                    model_path = os.path.join(self.save_path, f"final_model_difficulty_{current_difficulty}.zip")
                    self.model.save(model_path)

                    # raise difficulty level and reset envs allocation
                    self.raise_difficulty()

                    # reset cooldown, patience counter and best mean reward
                    self.cooldown = {k: self.cooldown_duration for k in self.curriculum.keys()}
                    self.best_mean_reward = 0
                    self.patience_counter = 0

                # calculate mean reward and export best model if threshold is met
                else:
                    current_mean_reward = self.get_mean_reward()
                    if current_mean_reward is not None:

                        if current_mean_reward >= self.best_mean_reward + self.min_improvement:
                            # replace best mean reward if it's higher
                            self.best_mean_reward = current_mean_reward

                            # delete previous best model
                            if (self.best_model_path is not None) and (os.path.exists(self.best_model_path)) and (
                            self.delete_previous_model):
                                os.remove(self.best_model_path)

                            # save model
                            self.best_model_path = os.path.join(self.save_path,
                                                                f"best_model_difficulty_{current_difficulty}_winrate_{current_winrate}.zip")
                            self.model.save(self.best_model_path)

                            # reset patience counter
                            self.patience_counter = 0

                            if self.verbose > 0:
                                print(
                                    f"New best ep_rew_mean={self.best_mean_reward:.4f}, "
                                    f"saved model to {self.best_model_path}"
                                )

                        # increase patience counter
                        else:
                            self.patience_counter += 1

    def _on_step(self):
        super()._on_step()

        # get state data from the envs
        matchups_running = []
        for env_idx in range(self.model.env.num_envs):
            # get_attr safely fetches from worker processes
            statename = self.model.env.get_attr('statename')[env_idx]
            basename = extract_matchup(statename)
            # # use the following line only for debugging
            # self.logger.record(f"training_curriculum_CHECK/difficulty_{basename.split('_')[-2]}", basename.split('_')[-1])
            matchups_running.append(re.sub(r"_\d+$", "", basename))

        # record envs per matchup
        counter = Counter(matchups_running)

        # log curriculum level for each matchup
        for matchup, value in self.curriculum.items():
            current_level = value["levels"][value["current_lvl_idx"]]
            self.logger.record(f"training_curriculum/difficulty_{matchup.split('_')[-1]}", int(current_level))

            # log env distribution for each matchup
            n_envs = counter[matchup] if matchup in counter else 0
            self.logger.record(f"training_envs/{matchup.split('_')[-1]}", n_envs)

        # reduce cooldown by number of episodes intercurred since last update
        if self.locals.get("infos"):
            for idx, info in enumerate(self.locals["infos"]):
                if info.get("episode"):
                    statename = self.model.env.get_attr("statename")[idx]
                    statename = extract_matchup(statename)
                    matchup = re.sub(r"_\d+$", "", statename)
                    self.cooldown[matchup] = max(self.cooldown[matchup] - 1, 0)
                    self.logger.record(f"training_matchups_cooldowns/cooldown_{matchup.split('_')[-1]}",
                                       self.cooldown[matchup])

        # log best mean reward of the last saved model
        if self.best_mean_reward is not None:
            self.logger.record("train/last_saved_mean_reward", self.best_mean_reward)

        # log patience counter
        self.logger.record("rollout/patience_counter", self.patience_counter)
        # stop training if (number of rollouts since last save) >= patience
        if self.patience_counter >= self.patience:
            while True:
                continue_training = input("Patience limit reached. Continue training? [y/n]")
                if continue_training == "y":
                    self.patience_counter = 0
                    return True
                elif continue_training == "n":
                    return False
                else:
                    print("Type either 'y' or 'n'")

        return True

    def raise_difficulty(self):
        for matchup in self.curriculum.keys():
            # get level from idx
            self.curriculum[matchup]["current_lvl_idx"] += 1

            next_lvl = self.curriculum[matchup]["levels"][self.curriculum[matchup]["current_lvl_idx"]]

            assert next_lvl <= self.max_level, "Curriculum level exceeds maximum level"

        self.reset_env_allocation()

    def reallocate_envs(self):
        # create winrates dict for all matchups
        winrates_dict = {}
        for name, value in self.winrates.items():
            if name != "global":
                # ensure winrates for all matchups are available
                if value.get("winrate") is not None:
                    winrates_dict[name] = value.get("winrate")
                else:
                    return

        winrates_df = pd.DataFrame(winrates_dict, index=[0])
        winrates_df = winrates_df.T.copy()
        winrates_df.columns = ["winrate"]

        # rescale winrates into weights
        # matchup weight increases for lower winrates
        winrates_df["weight"] = (1 - winrates_df["winrate"]).div((1 - winrates_df["winrate"]).sum(), axis=0)

        # calculate envs to allocate based on weights
        winrates_df["allocation"] = np.floor(winrates_df["weight"] * self.model.n_envs).astype(int)

        # ensure each matchup has at least 1 dedicated env
        winrates_df["allocation"] = winrates_df["allocation"].where(winrates_df["allocation"] != 0, 1)

        # if there are envs remaining, allocate them to the matchup with the highest weight
        remaining = self.model.n_envs - winrates_df["allocation"].sum()
        if remaining > 0:
            highest_idx = np.argmax(winrates_df["weight"])
            winrates_df.loc[winrates_df.index[highest_idx], "allocation"] += remaining

        # reallocate the envs
        new_states = []
        for matchup, row in winrates_df.iterrows():
            # fetch the current level being used
            current_diff_lvl = self.curriculum[matchup]["levels"][self.curriculum[matchup]["current_lvl_idx"]]
            envs_to_allocate = row["allocation"].astype(np.int32)
            if envs_to_allocate != 0:
                for _ in range(envs_to_allocate):
                    new_states.append(get_state(matchup, current_diff_lvl))

        envs_to_update = [i for i in range(self.model.env.num_envs)]
        self.locals["env"].update_env(new_states, envs_to_update)

    def reset_env_allocation(self):
        new_states = []
        envs_to_update = [i for i in range(self.model.env.num_envs)]

        for matchup, value in self.curriculum.items():
            for _ in range(value["n_envs"]):
                new_states.append(get_state(matchup, value["levels"][value["current_lvl_idx"]]))

        self.locals["env"].update_env(new_states, envs_to_update)

    def get_mean_reward(self):
        try:
            current_mean_reward = sum([ep_info["r"] for ep_info in self.model.ep_info_buffer]) / len(
                [ep_info["r"] for ep_info in self.model.ep_info_buffer])
            return current_mean_reward
        except ZeroDivisionError:
            return