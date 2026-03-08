"""
random_agent.py — Sanity-check script using a random-action agent.

Runs the SF2 environment with uniformly random actions to verify:
  • The environment loads and resets correctly.
  • The action space is compatible with random sampling.
  • Episode termination and info dicts work as expected.

This is useful for smoke-testing wrapper changes before running expensive
training runs.

Usage
-----
    python random_agent.py
"""

from wrapper import SFWrapper
import stable_retro as retro
from const import *


def make_env(
    game,
    state,
    side,
    reset_type="round",
    init_level=1,
    rendering=False,
    num_stack=12,
    num_step_frames=8,
    state_dir=None,
    verbose=False,
    enable_combo=True,
    null_combo=False,
    transform_action=False,
):
    """
    Build a wrapped SF2 environment for random-agent testing.

    Unlike the factory in utils.py, this returns the live environment directly
    (not a closure) since we only need one env and no subprocess parallelism.

    Parameters
    ----------
    game : str
        stable-retro game ID.
    state : str
        Path to the .state file defining the starting matchup.
    side : str
        "left" or "right" — which player the agent controls.
    reset_type : str
        Episode reset granularity ("round", "match", or "never").
    init_level : int
        Arcade ladder starting level.
    rendering : bool
        Open a display window for live visualisation.
    num_stack : int
        Number of frames to stack in the observation.
    num_step_frames : int
        Emulator frames advanced per agent step.
    state_dir : str or None
        Directory for save-state I/O (not used during random rollouts).
    verbose : bool
        Print round/match outcomes to stdout.
    enable_combo : bool
        Include combo bits in the action space.
    null_combo : bool
        Include combo bits but always send no-op combos.
    transform_action : bool
        Use a MultiDiscrete action space instead of MultiBinary.

    Returns
    -------
    SFWrapper
        Fully wrapped environment ready for interaction.
    """
    env = retro.make(
        game=game,
        state=state,
        use_restricted_actions=retro.Actions.FILTERED,
        obs_type=retro.Observations.IMAGE,
    )
    env = SFWrapper(
        env, side=side, rendering=rendering, reset_type=reset_type,
        init_level=init_level, state_dir=state_dir, verbose=verbose,
        enable_combo=enable_combo, null_combo=null_combo,
        transform_action=transform_action, num_stack=num_stack,
        num_step_frames=num_step_frames,
    )
    return env


def random_agent(env, episodes=1):
    """
    Run the environment for `episodes` episodes using uniformly random actions.

    Samples actions from `env.action_space` at every step.  Prints the final
    match score (player rounds won vs CPU rounds won) after each episode.

    Parameters
    ----------
    env : SFWrapper
        The wrapped environment to interact with.
    episodes : int
        Number of full episodes (matches or rounds, depending on reset_type) to run.
    """
    obs = env.reset()
    done = False

    for game in range(episodes):
        while not done:
            # Draw a uniformly random action from the MultiBinary or MultiDiscrete space.
            action = env.action_space.sample()
            obs, reward, done, truncated, info = env.step(action)

        # Print final score for this episode.
        print(f"Player: {info['matches_won']} - CPU: {info['enemy_matches_won']}")

    env.close()

def main():
    game = sf_game
    state = "/home/emeralddawns/Documents/StreetFighter2RL/data/states/ryu_vs_balrog_1.state"
    side = "left" # side for AI to control
    reset_type = "match"
    rendering = True
    verbose = True
    enable_combo = True # enable special move action space for environment
    null_combo = False # null action space for special move
    transform_action = False
    num_stack = 12 # number of frames to stack
    num_step_frames = 8 # number of frames per step
    episodes = 1

    env = make_env(game, state=state, side=side, reset_type=reset_type, rendering=rendering, verbose=verbose, enable_combo=enable_combo, null_combo=null_combo, transform_action=transform_action, num_stack=num_stack, num_step_frames=num_step_frames)

    random_agent(env, episodes)

if __name__ == "__main__":
    main()




