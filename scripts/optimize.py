"""
optimize.py — Optuna hyperparameter search for the Street Fighter II PPO agent.

Runs `n_trials` Optuna trials; each trial samples a PPO hyperparameter config,
trains a model for `n_steps` timesteps, evaluates it over `eval_episodes`
episodes, and returns the mean reward as the objective to maximise.

Usage
-----
    python optimize.py

Results are stored in an in-memory Optuna study (not persisted to a database by
default).  To persist across runs, swap `optuna.create_study()` for one that
uses a storage URL, e.g.:
    optuna.create_study(storage="sqlite:///optuna.db", study_name="sf2_ppo")
"""

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

# Save-state used for all Optuna trials (fixed for comparability).
state = "/home/emeralddawns/Documents/StreetFighter2RL/data/states/ryu_vs_balrog_1.state"

# Training budget per trial (intentionally short to allow many trials).
n_steps = 100_000

# Number of evaluation episodes after training to compute the objective.
eval_episodes = 10

# Total number of Optuna trials to run.
n_trials = 100

# EXP: just verbose. a reminder that it is here
def make_env(game, state, side="left", reset_type="match", rendering=False, init_level=1, state_dir=None, verbose=True, enable_combo=True, null_combo=False, transform_action=False, aggresive_coeff=3.0, dense_coeff=1.0, num_stack=12, num_step_frames=8):
    """
    Environment factory for Optuna trials.

    Functionally identical to utils.make_env() but defined locally so that
    optimize.py can be run independently without importing utils.

    Returns
    -------
    Callable[[], gymnasium.Env]
        Zero-argument factory suitable for SubprocVecEnv.
    """

    def _init():
        env = retro.make(
            game=game, 
            state=state, 
            use_restricted_actions=retro.Actions.FILTERED,
            render_mode = rendering,
            obs_type=retro.Observations.IMAGE,
        )

        env = SFWrapper(env, side=side, rendering=rendering, reset_type=reset_type, init_level=init_level, state_dir=state_dir, verbose=verbose, enable_combo=enable_combo, null_combo=null_combo, transform_action=transform_action, aggresive_coeff=aggresive_coeff, dense_coeff=dense_coeff, num_stack=num_stack, num_step_frames=num_step_frames)

        # Monitor records per-episode stats; info_keywords are extra columns logged.
        env = Monitor(env, LOG_DIR, info_keywords=("matches_won", "enemy_matches_won", "health", "enemy_health"))

        return env
    return _init

# Function to return test hyperparameters - define the object function
def optimize_ppo(trial):
    """
    Sample a PPO hyperparameter configuration from Optuna's search space.

    Called once per trial to draw a candidate configuration.  Optuna's
    suggest_* methods record the sampled values and use them to guide future
    trials via Bayesian optimisation (TPE sampler by default).

    Parameters
    ----------
    trial : optuna.trial.Trial
        The current Optuna trial object.

    Returns
    -------
    dict
        Keyword arguments to pass directly to the PPO constructor.
    """
    # EXP: test some more stuff
    net_arch_name = trial.suggest_categorical('net_arch', ['small', 'medium', 'large', 'asymmetric'])
    net_arch_map = {
        'small':      [64, 64],
        'medium':     [128, 128],
        'large':      [256, 256],
        'asymmetric': [256, 128],
    }
    n_steps_multiplier = trial.suggest_int('n_steps_multiplier', 1, 8)
    batch_size = trial.suggest_categorical('batch_size', [64, 128, 256, 512, 1024])
    return {
        # Number of steps collected per rollout per env (power-of-two range).
        'n_steps': n_steps_multiplier * batch_size,  # trial.suggest_int('n_steps', 2048, 8192),
        # Discount factor: higher = more far-sighted.
        'gamma': trial.suggest_float('gamma', 0.90, 0.999, log=True),
        # Learning rate: log-uniform so small values are sampled proportionally.
        'learning_rate': trial.suggest_float('learning_rate', 5e-6, 3e-4, log=True),
        # PPO clipping parameter: controls how far the new policy can deviate.
        'clip_range': trial.suggest_float('clip_range', 0.1, 0.4),
        # GAE lambda: bias-variance trade-off for advantage estimation.
        'gae_lambda': trial.suggest_float('gae_lambda', 0.8, 0.99),
        # Mini-batch size for the policy gradient update.
        'batch_size': trial.suggest_categorical('batch_size', [64, 128, 256, 512, 1024]),
        'n_epochs': trial.suggest_int('n_epochs', 3, 10),
        'ent_coef': trial.suggest_float('ent_coef', 1e-4, 1e-2, log=True),
        'vf_coef': trial.suggest_float('vf_coef', 0.3, 0.7),
        'max_grad_norm': trial.suggest_float('max_grad_norm', 0.3, 1.0),
        'policy_kwargs': {
            'net_arch': net_arch_map[net_arch_name],
            'ortho_init': trial.suggest_categorical('ortho_init', [True, False]),
        },
        # Env-specific — popped out before passing to PPO
        'aggresive_coeff': trial.suggest_float('aggresive_coeff', 1.0, 5.0),
        'dense_coeff': trial.suggest_float('dense_coeff', 0.5, 2.0),
        'num_stack': trial.suggest_categorical('num_stack', [8, 12, 16]),
        'num_step_frames': 8,  # locked — must match len(SF_COMBOS[i])
    }

# Run a training loop and return mean reward 
def optimize_agent(trial):
    """
    Optuna objective function: train + evaluate one PPO configuration.

    This is called once per trial by `study.optimize()`.  It:
    1. Samples hyperparameters via optimize_ppo(trial).
    2. Builds a SubprocVecEnv with a single worker environment.
    3. Trains a PPO model for `n_steps` timesteps.
    4. Evaluates the policy for `eval_episodes` episodes.
    5. Saves the trained model under `opt/trial_<N>_best_model.zip`.
    6. Returns mean reward as the scalar objective for Optuna to maximise.

    Parameters
    ----------
    trial : optuna.trial.Trial

    Returns
    -------
    float
        Mean evaluation reward (higher is better).
    """
    model_params = optimize_ppo(trial)

    # Single-worker environment (keeps trial runtime predictable).
    n_envs = 4

    # EXP: also test these
    aggresive_coeff  = model_params.pop('aggresive_coeff')
    dense_coeff      = model_params.pop('dense_coeff')
    num_stack        = model_params.pop('num_stack')
    num_step_frames  = model_params.pop('num_step_frames')

    env = SubprocVecEnv([
        make_env(sf_game, state, aggresive_coeff=aggresive_coeff, dense_coeff=dense_coeff, num_stack=num_stack, num_step_frames=num_step_frames)
        for _ in range(n_envs)
    ])

    # # Ensure batch_size divides the rollout buffer size evenly
    # rollout_size = model_params['n_steps'] * n_envs
    # if rollout_size % model_params['batch_size'] != 0:
    #     # Round batch_size down to the nearest valid divisor
    #     model_params['batch_size'] = max(
    #         b for b in [64, 128, 256, 512, 1024] if rollout_size % b == 0
    #     )

    # Lightweight callback: tracks win-rate for TensorBoard without checkpointing.
    callback = OptunaTrainAndLoggingCallback(winrate_buffer_size=5)

    # Construct PPO with the sampled hyperparameters.
    model = PPO(
        'CnnPolicy', env,
        device="cuda",
        tensorboard_log=LOG_DIR,
        verbose=1,
        **model_params,
    )
    model.learn(total_timesteps=n_steps, callback=callback)

    # Evaluate the trained policy (deterministic rollouts).
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=eval_episodes)
    env.close()

    # Persist the model so we can inspect promising trials later.
    SAVE_PATH = os.path.join(OPT_DIR, f'trial_{trial.number}_best_model')
    model.save(SAVE_PATH)

    return mean_reward

def main():
    """
    Run the full Optuna hyperparameter search.

    Creates an in-memory study (maximising mean reward) and runs `n_trials`
    sequential trials.  Prints the best trial configuration on completion.
    """
    # direction="maximize" because our objective is mean reward (higher = better).
    study = optuna.create_study(direction='maximize')
    # n_jobs=1: run trials sequentially.  Increase for parallel search
    # (requires thread-safe environments or separate processes).
    study.optimize(optimize_agent, n_trials=n_trials, n_jobs=1)

    print("Optimization completed")
    print(f"Best trial: {study.best_trial}")

if __name__ == "__main__":
    main()

