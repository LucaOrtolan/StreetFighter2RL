import os
import re
from collections import deque, Counter
from stable_baselines3.common.callbacks import BaseCallback
from utils import get_state, extract_matchup
from const import *
import pandas as pd
import numpy as np


class TrainAndLoggingCallback(BaseCallback):
    """To visualize train stats in Tensorboard, run the script from the terminal and run in parallel:
            $ tensorboard --logdir=logs
    """

    def __init__(self, verbose=1, winrate_buffer_size=100):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.winrate_buffer_size = winrate_buffer_size
        self.episode_count = {"global": 0}
        self.winrates = {"global": {
            "buffer": deque(maxlen=winrate_buffer_size),
            "winrate": None}}

    def _init_callback(self):
        self.logger.record("train/batch_size", self.model.batch_size)
        self.logger.record("train/n_envs", self.model.n_envs)
        self.logger.record("train/n_steps", self.model.n_steps)
        self.logger.record("train/gamma", self.model.gamma)
        self.logger.record("train/n_epochs", self.model.n_epochs)

    def _on_step(self):
        # Winrate tracking - check episode terminations
        if self.locals.get("infos"):
            for idx, info in enumerate(self.locals["infos"]):
                # Check if episode ended (works with both Gymnasium and Gym APIs)
                if info.get("episode"):
                    self.episode_count["global"] += 1
                    # Get win indicator from info (ensure SFWrapper sets this)
                    outcome = info.get("outcome")
                    outcome_cat = 1.0 if outcome == "win" else 0.0
                    self.winrates["global"]["buffer"].append(outcome_cat)

                    statename = self.model.env.get_attr("statename")[idx]
                    statename = extract_matchup(statename)
                    matchup = re.sub(r"_\d+$", "", statename) 

                    if matchup not in self.winrates.keys():
                        self.winrates[matchup] = {}
                        self.winrates[matchup]["buffer"] = deque(maxlen=self.winrate_buffer_size)
                    if matchup not in self.episode_count.keys():
                        self.episode_count[matchup] = 0
                    
                    self.winrates[matchup]["buffer"].append(outcome_cat)
                    self.episode_count[matchup] += 1

                    if self.episode_count["global"] >= self.winrate_buffer_size:
                        global_winrate = sum(self.winrates["global"]["buffer"]) / len(self.winrates["global"]["buffer"])
                        self.winrates["global"]["winrate"] = global_winrate

                        # update winrate per matchup                        
                        matchup_winrate = sum(self.winrates[matchup]["buffer"]) / len(self.winrates[matchup]["buffer"])
                        self.winrates[matchup]["winrate"] = matchup_winrate

        # log all episode counts and winrates        
        for matchup, value in self.winrates.items():
            # handle global case
            if matchup == "global":
                self.logger.record(f"time/episode_count", self.episode_count[matchup])
                if value.get("winrate") is not None:
                    self.logger.record(f"rollout/ep_winrate", value.get("winrate"))
                continue

            # send opponent's name to logs for short
            enemy_character = matchup.split("_")[-1]

            self.logger.record(f"time/episode_count_{enemy_character}", self.episode_count[matchup])

            if value.get("winrate") is not None:
                self.logger.record(f"rollout/ep_winrate_{enemy_character}", value.get("winrate"))

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

class CurriculumLearningCallback(TrainAndLoggingCallback):
    """Multi functional Callback for implementing Curriculum Learning. Performs logging for every metric, state update and saves best model."""

    def __init__(self, curriculum, save_path, winrate_threshold=.9, winrate_buffer_size=100, min_improvement=0.01, patience=1000, delete_previous_model=True, verbose=1):
        super(CurriculumLearningCallback, self).__init__(verbose, winrate_buffer_size=winrate_buffer_size)
        ###### curriculum parameters
        self.curriculum = curriculum
        self.winrate_threshold = winrate_threshold
        self.cooldown_duration = winrate_buffer_size # must be equal to winrate buffer size
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
            self.curriculum[matchup]["current_lvl_idx"] = 0 # keeps track of the index of the current level
    
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
                current_difficulty = self.curriculum[sample_matchup]["levels"][self.curriculum[sample_matchup]["current_lvl_idx"]]

                # reallocate envs
                self.reallocate_envs()

                # raise difficulty and export model is winrate threshold is met
                if (current_winrate >= self.winrate_threshold) & (current_lvl_idx < len(self.curriculum[sample_matchup]["levels"])):
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
                            if (self.best_model_path is not None) and (os.path.exists(self.best_model_path)) and (self.delete_previous_model):
                                os.remove(self.best_model_path)

                            # save model
                            self.best_model_path = os.path.join(self.save_path, f"best_model_difficulty_{current_difficulty}_winrate_{current_winrate}.zip")
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
                    self.logger.record(f"training_matchups_cooldowns/cooldown_{matchup.split('_')[-1]}", self.cooldown[matchup])

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

        # rescale winrates into weigths 
        # matchup weight increases for lower winrates
        winrates_df["weight"] = (1 - winrates_df["winrate"]).div((1 - winrates_df["winrate"]).sum(), axis=0)

        # calculate envs to allocate based on weights
        winrates_df["allocation"] = np.floor(winrates_df["weight"] * self.model.n_envs).astype(int)

        # ensure each matchup has at least 1 dedicated env
        winrates_df["allocation"] = winrates_df["allocation"].where(winrates_df["allocation"]!=0, 1)

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
            current_mean_reward = sum([ep_info["r"] for ep_info in self.model.ep_info_buffer])/len([ep_info["r"] for ep_info in self.model.ep_info_buffer])
            return current_mean_reward
        except ZeroDivisionError:
            return
