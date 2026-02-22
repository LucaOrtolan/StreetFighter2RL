import stable_retro as retro
from wrapper import SFWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from const import * 
import optuna
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv
from callbacks import OptunaTrainAndLoggingCallback

OPT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "opt")
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")

state = "/home/master26/Documents/StreetFighter2RL/FightLadder/data/sf/curriculum/Level2.20.state"
n_steps = 100_000
eval_episodes = 10
n_trials = 100

def make_env(game, state, side="left", reset_type="match", rendering=False, init_level=1, state_dir=None, verbose=False, enable_combo=True, null_combo=False, transform_action=False, num_stack=12, num_step_frames=8):
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

# Function to return test hyperparameters - define the object function
def optimize_ppo(trial): 
    return {
        'n_steps':trial.suggest_int('n_steps', 2048, 8192), # These are all wrong
        'gamma':trial.suggest_loguniform('gamma', 0.8, 0.9999),
        'learning_rate':trial.suggest_loguniform('learning_rate', 1e-5, 1e-4),
        'clip_range':trial.suggest_uniform('clip_range', 0.1, 0.4),
        'gae_lambda':trial.suggest_uniform('gae_lambda', 0.8, 0.99),
        'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256, 512, 1024])
    }

# Run a training loop and return mean reward 
def optimize_agent(trial):
    model_params = optimize_ppo(trial) 

    # Create environment 
    env = make_env(sf_game, state)

    env = SubprocVecEnv([env])

    callback = OptunaTrainAndLoggingCallback(
        winrate_buffer_size=5
    )

    # Create algo 
    model = PPO('CnnPolicy', env, device="cuda", tensorboard_log=LOG_DIR, verbose=1, **model_params)
    model.learn(total_timesteps=n_steps, callback=callback)

    # Evaluate model 
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=eval_episodes)
    env.close()

    SAVE_PATH = os.path.join(OPT_DIR, 'trial_{}_best_model'.format(trial.number))
    model.save(SAVE_PATH)

    return mean_reward

def main():
    # Creating the experiment 
    study = optuna.create_study(direction='maximize')
    study.optimize(optimize_agent, n_trials=n_trials, n_jobs=1)

    print("Optimization completed")
    print(f"Best trial: {study.best_trial}")

if __name__ == "__main__":
    main()

