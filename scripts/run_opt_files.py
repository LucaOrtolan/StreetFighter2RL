"""
evaluate_trials.py — Evaluate all trial best models and rank them.
AI wrote this so I could recover the results from the Optuna Trials
"""
import csv
import os
import numpy as np
import stable_retro as retro
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from wrapper import SFWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage

# ── Config ────────────────────────────────────────────────────────────────────
MODELS_FOLDER  = "/home/emeralddawns/Documents/StreetFighter2RL/opt"        # folder with all trial_N_best_model.zip
STATE_DIR      = "/home/emeralddawns/Documents/StreetFighter2RL/data/states"
N_EVAL_EPISODES = 5

# Environment settings (must match training)
game            = "StreetFighterIISpecialChampionEdition-Genesis-v0"
side            = "left"
reset_type      = "match"
rendering       = False
enable_combo    = True
null_combo      = False
transform_action = False
num_stack       = 12
num_step_frames = 8

# Opponents to evaluate against:
# The original 2 ("ryu_vs_ken_1", "ryu_vs_ryu_1")
MATCHUPS = ["ryu_vs_ken_1", "ryu_vs_ryu_1"]
# ─────────────────────────────────────────────────────────────────────────────

model = PPO.load(os.path.join(MODELS_FOLDER, "trial_0_best_model.zip"))
print("obs space shape:", model.observation_space.shape)

def get_num_stack_from_model(model):
    """
    obs space is channels-first: (C, H, W)
    num_step_frames is always 8, step = 8 // 2 = 4
    num_stack = C * step
    """
    num_step_frames = 8
    obs_shape = model.observation_space.shape  # (C, H, W)
    n_channels = obs_shape[0]  # first dimension is channels
    step = num_step_frames // 2  # = 4
    num_stack = n_channels * step
    return num_stack, num_step_frames

def make_eval_env(matchup):
    """Create a single evaluation environment (no SubprocVecEnv needed)."""
    state = os.path.join(STATE_DIR, f"{matchup}.state")

    def _init():
        env = retro.make(
            game=game,
            state=state,
            use_restricted_actions=retro.Actions.FILTERED,  # match optimize.py
            obs_type=retro.Observations.IMAGE,
            render_mode=None
        )
        env = SFWrapper(
            env,
            side=side,
            reset_type=reset_type,
            rendering=rendering,
            enable_combo=enable_combo,
            null_combo=null_combo,
            transform_action=transform_action,
            num_stack=num_stack,
            num_step_frames=num_step_frames,
        )
        env = Monitor(env)
        return env

    return _init()

def evaluate_model_on_matchup(model, matchup, n_episodes, num_stack, num_step_frames):
    """Create env, evaluate, then immediately close it."""
    state = os.path.join(STATE_DIR, f"{matchup}.state")

    env = retro.make(
        game=game,
        state=state,
        use_restricted_actions=retro.Actions.FILTERED,
        obs_type=retro.Observations.IMAGE,
        render_mode=None
    )
    env = SFWrapper(
        env,
        side=side,
        reset_type=reset_type,
        rendering=rendering,
        enable_combo=enable_combo,
        null_combo=null_combo,
        transform_action=transform_action,
        num_stack=num_stack,
        num_step_frames=num_step_frames,
    )
    env = Monitor(env)
    # Wrap in DummyVecEnv + VecTransposeImage to match how SB3 trained the model
    env = DummyVecEnv([lambda: env])
    env = VecTransposeImage(env)

    rewards = []
    wins = []

    for _ in range(n_episodes):
        obs = env.reset()  # VecEnv reset returns obs directly, no tuple
        done = False
        total_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)  # VecEnv returns 4 values
            done = done[0]
            total_reward += reward[0]

        rewards.append(total_reward)
        wins.append(info[0].get("matches_won", 0))

    env.close()
    return rewards, wins


def evaluate_model(model, n_episodes):
    """Evaluate model across all matchups sequentially."""
    num_stack, num_step_frames = get_num_stack_from_model(model)

    all_rewards = []
    all_wins = []

    for matchup in MATCHUPS:
        rewards, wins = evaluate_model_on_matchup(
            model, matchup, n_episodes, num_stack, num_step_frames
        )
        all_rewards.extend(rewards)
        all_wins.extend(wins)

    return {
        "avg_reward": float(np.mean(all_rewards)),
        "avg_wins":   float(np.mean(all_wins)),
        "std_reward": float(np.std(all_rewards)),
        "num_stack":  num_stack,
    }


def main():
    model_files = sorted([
        f for f in os.listdir(MODELS_FOLDER)
        if f.startswith("trial_") and f.endswith("_best_model.zip")
    ])
    print(f"Found {len(model_files)} models to evaluate.")

    results = {}

    for i, model_file in enumerate(model_files):
        print(f"[{i+1}/{len(model_files)}] Evaluating {model_file}...")
        try:
            model = PPO.load(os.path.join(MODELS_FOLDER, model_file))
            stats = evaluate_model(model, N_EVAL_EPISODES)
            results[model_file] = stats
            print(f"  → avg_reward={stats['avg_reward']:.3f}, "
                  f"avg_wins={stats['avg_wins']:.2f}, "
                  f"std={stats['std_reward']:.3f}, "
                  f"num_stack={stats['num_stack']}")

        except Exception as e:
            print(f"  ✗ Failed: {e}")
            results[model_file] = {
                "avg_reward": -999, "avg_wins": 0,
                "std_reward": 0, "num_stack": -1
            }

    sorted_models = sorted(
        results.items(),
        key=lambda x: (x[1]["avg_wins"], x[1]["avg_reward"]),
        reverse=True,
    )

    # Save to CSV
    csv_path = os.path.join(MODELS_FOLDER, "trial_rankings.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "rank", "model", "avg_wins", "avg_reward", "std_reward", "num_stack"
        ])
        writer.writeheader()
        for rank, (model_file, stats) in enumerate(sorted_models, 1):
            writer.writerow({
                "rank":       rank,
                "model":      model_file,
                "avg_wins":   round(stats["avg_wins"],   4),
                "avg_reward": round(stats["avg_reward"], 4),
                "std_reward": round(stats["std_reward"], 4),
                "num_stack":  stats["num_stack"],
            })
    print(f"\nResults saved to {csv_path}")

    print("\n" + "="*60)
    print("FINAL RANKINGS")
    print("="*60)
    for rank, (model_file, stats) in enumerate(sorted_models, 1):
        print(f"#{rank:>3}  {model_file:<40}  "
              f"wins={stats['avg_wins']:.2f}  "
              f"reward={stats['avg_reward']:.3f}  "
              f"num_stack={stats['num_stack']}")

    best = sorted_models[0]
    print(f"\nBest model: {best[0]}")
    print(f"   avg_wins={best[1]['avg_wins']:.2f}, "
          f"avg_reward={best[1]['avg_reward']:.3f}, "
          f"num_stack={best[1]['num_stack']}")


if __name__ == "__main__":
    main()