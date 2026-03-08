"""
test_policy.py — Evaluate a saved PPO policy against the SF2 CPU.

Loads a trained model checkpoint, runs it in the environment for a specified
number of episodes, and reports the final win-rate.

Usage
-----
    python test_policy.py

Optional: set `record=<path>` to save BK2 replay files for post-hoc analysis.
"""

from stable_baselines3 import PPO
from wrapper import SFWrapper
import stable_retro as retro
import os
from const import *
from collections import deque


def load_model(model_name):
    """
    Load a PPO model from the project's `saved_models` directory.

    Parameters
    ----------
    model_name : str
        Either just a filename (without extension) relative to `saved_models/`,
        or an absolute path to the .zip file.

    Returns
    -------
    PPO
        The loaded policy model (weights on CPU by default after loading).
    """
    saved_models_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "saved_models")
    model = PPO.load(os.path.join(saved_models_path, model_name))
    return model


def make_env(
    state,
    game=sf_game,
    side="left",
    reset_type="match",
    init_level=1,
    rendering=False,
    num_stack=12,
    num_step_frames=8,
    state_dir=None,
    verbose=False,
    enable_combo=True,
    null_combo=False,
    transform_action=False,
    record=False,
):
    """
    Build a single wrapped SF2 environment for policy evaluation.

    Parameters
    ----------
    state : str
        Path to the .state file defining the matchup to evaluate against.
    game : str
        stable-retro game ID.
    side : str
        "left" or "right" — which player the policy controls.
    reset_type : str
        Episode reset granularity ("round", "match", or "never").
    init_level : int
        Arcade ladder starting level.
    rendering : bool
        Open a display window for live playback.
    num_stack : int
        Number of frames to stack in the observation.
    num_step_frames : int
        Emulator frames advanced per agent step.
    state_dir : str or None
        Directory for save-state I/O.
    verbose : bool
        Print round/match outcomes to stdout.
    enable_combo : bool
        Include special-move combo bits in the action space.
    null_combo : bool
        Include combo bits but always send no-op combos.
    transform_action : bool
        Use MultiDiscrete action space.
    record : str or False
        Path to save BK2 replay files, or False to disable recording.

    Returns
    -------
    SFWrapper
        Fully wrapped environment ready for policy rollouts.
    """
    env = retro.make(
        game=game,
        state=state,
        use_restricted_actions=retro.Actions.FILTERED,
        obs_type=retro.Observations.IMAGE,
        record=record,                          # BK2 replay recording path.
        render_mode="human" if rendering else False,
    )
    env = SFWrapper(
        env, side=side, rendering=rendering, reset_type=reset_type,
        init_level=init_level, state_dir=state_dir, verbose=verbose,
        enable_combo=enable_combo, null_combo=null_combo,
        transform_action=transform_action, num_stack=num_stack,
        num_step_frames=num_step_frames,
    )
    return env


def play(env, model, episodes):
    """
    Run the policy in the environment for `episodes` complete matches.

    Logic
    -----
    Each episode consists of a best-of-3 match.  The loop exits a match
    (sets game_on=False) when either player reaches 2 round wins, then
    resets the environment for the next episode.

    A rolling `frames` deque is used to detect the start of an episode
    (when fewer than `num_stack` frames have been seen, we default to a
    zero-action to let the frame buffer fill before querying the policy).

    Parameters
    ----------
    env : SFWrapper
        The evaluation environment.
    model : PPO
        The loaded policy to evaluate.
    episodes : int
        Number of complete matches to play.
    """
    obs = env.reset()
    # Circular buffer tracking the last num_stack observations (used to detect
    # the warm-up period at the start of each episode).
    frames = deque(maxlen=env.num_stack)
    wins = 0

    for ep in range(episodes):
        print(f"episode = {ep}")
        game_on = True

        while game_on:
            if len(frames) < env.num_stack:
                # Frame buffer not yet full: send a no-op action.
                action = np.zeros(env.action_dim, dtype=np.int8)
            else:
                # Query the policy for the best action given the current obs.
                action = model.predict(obs)[0]

            obs, reward, done, truncated, info = env.step(action)
            frames.append(obs)

            # Check if the match has concluded (either player reached 2 wins).
            if info["matches_won"] == 2 or info["enemy_matches_won"] == 2:
                if info["matches_won"] == 2:
                    wins += 1
                print(f"Player: {info['matches_won']} - CPU: {info['enemy_matches_won']}")
                env.reset()
                game_on = False  # Move on to the next episode.

    env.close()
    print("Simulation completed. Final winrate = %.2f" % (wins / episodes))
    print(f"Wins = {wins} out of {episodes} matches")



def main():
    # Path to the trained model checkpoint (.zip extension is optional for PPO.load).
    model_name = "/home/master26/Documents/StreetFighter2RL/saved_models/mu5_8_winrate_0.155_episode_82645"

    # Save-state defining the opponent and difficulty for evaluation.
    state = "/home/master26/Documents/StreetFighter2RL/data/states/ryu_vs_ryu_8.state"

    # record = os.path.join(os.path.dirname(os.path.dirname(__file__)), "replays")
    # Set to a directory path to record BK2 replay files, or False to skip.
    record    = False
    rendering = True
    episodes  = 1

    env = make_env(state=state, rendering=rendering, record=record)
    model = load_model(model_name)
    play(env, model, episodes)


if __name__=="__main__":
    main()