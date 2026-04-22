"""
fight.py — Run two trained PPO models against each other in a 1v1 fight.

Both models are loaded from .zip files and placed into a shared Street Fighter II
environment using side="both" so each controls one character simultaneously.

The side="both" path in SFWrapper has a one-line obs bug (returns the method
reference instead of calling it). FightEnv fixes that by pulling the stacked
frame buffer from FrameStackObservation.obs_queue after each step.

Usage
-----
    python fight.py --model1 path/to/model1.zip --model2 path/to/model2.zip

    # with options (paths are relative to cwd, or use absolute paths)
    python fight.py \
        --model1 ../train/best_model_winrate_0.851_episode_8059.zip \
        --model2 ../train/final_models/final_model_difficulty_8.zip \
        --state  ../data/states/ryu_vs_ryu_8.state \
        --episodes 20 \
        --render \
        --verbose
"""

import sys
import os
import argparse
from collections import deque

import numpy as np
import stable_retro as retro
from stable_baselines3 import PPO

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)
sys.path.insert(0, SCRIPTS_DIR)

from wrapper import SFWrapper  # noqa: E402
from const import sf_game      # noqa: E402

STATES_DIR = os.path.join(PROJECT_ROOT, "data", "states")
DEFAULT_STATE = os.path.join(STATES_DIR, "player_ryu_vs_player_ryu.state")


def mirror_obs(obs):
    """
    Horizontally flip an observation so a right-side model sees
    the screen from the same perspective it was trained on (left side).
    """
    return obs[:, ::-1, :].copy()

# ---------------------------------------------------------------------------
# Two-player environment
# ---------------------------------------------------------------------------

class FightEnv(SFWrapper):
    """
    SFWrapper subclass for 2-model fights.

    The only change is fixing the side='both' observation bug in SFWrapper.step():
    the original code accidentally returns `self._get_obs` (the unbound method)
    instead of `self._get_obs(obs)`. We reconstruct the observation from
    FrameStackObservation's internal obs_queue after the parent step completes.
    """

    def __init__(self, *args, debug_emulator=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._emu_frame = 0
        if debug_emulator:
            from const import BUTTONS
            original_step = self.env.step

            def _logged_step(action):
                mid = len(action) // 2
                p1_btns = [BUTTONS[i] for i, b in enumerate(action[:mid]) if b]
                p2_btns = [BUTTONS[i] for i, b in enumerate(action[mid:]) if b]
                p1_str = "+".join(p1_btns) if p1_btns else "NO-OP"
                p2_str = "+".join(p2_btns) if p2_btns else "NO-OP"
                print(f"    EMU {self._emu_frame:>5}: P1={p1_str:<40}  P2={p2_str}")
                self._emu_frame += 1
                return original_step(action)

            self.env.step = _logged_step

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self._latest_obs = obs
        return obs, info

    def step(self, action):
        result = super().step(action)

        if self.side != "both":
            obs, reward, done, truncated, info = result
            self._latest_obs = obs
            return obs, reward, done, truncated, info

        # side="both" returns (obs_fn_or_obs, r1, r2, done, truncated, info)
        obs_or_fn, r1, r2, done, truncated, info = result

        if callable(obs_or_fn):
            # Bug: obs_or_fn is self._get_obs method, not the actual observation.
            # Reconstruct from FrameStackObservation's frame buffer (self.env.obs_queue).
            try:
                raw_stacked = np.array(list(self.env.obs_queue))
                obs = self._get_obs(raw_stacked)
            except Exception:
                obs = self._latest_obs
        else:
            obs = obs_or_fn

        self._latest_obs = obs
        return obs, r1, r2, done, truncated, info


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------

def make_fight_env(state, rendering=False, num_stack=12, num_step_frames=8,
                   enable_combo=True, debug_emulator=False):
    retro_env = retro.make(
        game=sf_game,
        state=state,
        use_restricted_actions=retro.Actions.FILTERED,
        obs_type=retro.Observations.IMAGE,
        render_mode="human" if rendering else False,
        players=2,
    )
    return FightEnv(
        retro_env,
        side="both",
        reset_type="match",
        rendering=rendering,
        num_stack=num_stack,
        num_step_frames=num_step_frames,
        enable_combo=enable_combo,
        verbose=False,
        debug_emulator=debug_emulator,
    )


# ---------------------------------------------------------------------------
# Fight loop
# ---------------------------------------------------------------------------

def _decode_action(action, action_dim):
    """Return a human-readable summary of a binary action vector."""
    from const import BUTTONS
    base = action[:12]
    buttons = [BUTTONS[i] for i, b in enumerate(base) if b]
    summary = "+".join(buttons) if buttons else "NO-OP"
    if action_dim > 12:
        combo_id = int(4 * action[-3] + 2 * action[-2] + action[-1])
        if combo_id < 6:
            summary += f" [combo {combo_id}]"
    return summary


def fight(model1_path, model2_path, state=DEFAULT_STATE, episodes=10,
          rendering=False, verbose=False, debug_actions=False,
          debug_emulator=False, name1="Model 1", name2="Model 2"):
    """
    Run model1 (left / P1) against model2 (right / P2) for N episodes.

    Parameters
    ----------
    model1_path : str   Path to the first model .zip file.
    model2_path : str   Path to the second model .zip file.
    state       : str   Path to the .state file defining the matchup.
    episodes    : int   Number of complete best-of-3 matches to play.
    rendering   : bool  Open a display window for live playback.
    verbose     : bool  Print per-episode results.
    name1       : str   Display name for Player 1.
    name2       : str   Display name for Player 2.

    Returns
    -------
    dict  {name1: int, name2: int, 'draw': int}
    """
    print(f"Loading {name1}: {model1_path}")
    model1 = PPO.load(model1_path)
    print(f"Loading {name2}: {model2_path}")
    model2 = PPO.load(model2_path)
    print(f"State file: {state}")
    print(f"Episodes:   {episodes}")
    print()

    env = make_fight_env(state, rendering=rendering, debug_emulator=debug_emulator)

    wins = {name1: 0, name2: 0, "draw": 0}

    obs, info = env.reset()
    # Tracks how many observations have been seen since last reset.
    # We send no-ops while the frame buffer is warming up, matching test_policy.py.
    frames = deque(maxlen=env.num_stack)

    for ep in range(episodes):
        game_on = True

        while game_on:
            if len(frames) < env.num_stack:
                # Frame buffer not yet full — send no-ops for both players.
                # Combo bits must be 111 (=7, out-of-range) to mean "no combo";
                # all-zeros would decode as combo index 0 (Hadouken).
                action1 = np.zeros(env.action_dim, dtype=np.int8)
                action2 = np.zeros(env.action_dim, dtype=np.int8)
                action1[-3:] = 1
                action2[-3:] = 1
            else:
                action1 = model1.predict(obs, deterministic=True)[0]
                action2 = model2.predict(obs, deterministic=True)[0]

                # action2_mirrored = action2.copy()
                # action2_mirrored[6] = action2[7]  # LEFT ← RIGHT
                # action2_mirrored[7] = action2[6]  # RIGHT ← LEFT

            if debug_actions and len(frames) >= env.num_stack:
                a1_str = _decode_action(action1, env.action_dim)
                a2_str = _decode_action(action2, env.action_dim)
                print(f"  step {len(frames):>5}  {name1}: {a1_str:<30}  {name2}: {a2_str}")

            combined = np.hstack([action1, action2])
            obs, _r1, _r2, done, truncated, info = env.step(combined)
            frames.append(obs)

            m1_wins = info.get("matches_won", 0)
            m2_wins = info.get("enemy_matches_won", 0)

            match_over = (m1_wins == 2 or m2_wins == 2) or done or truncated
            if match_over:
                if m1_wins > m2_wins:
                    winner = name1
                elif m2_wins > m1_wins:
                    winner = name2
                else:
                    winner = "draw"

                wins[winner] += 1
                ep_num = ep + 1

                if verbose:
                    print(f"  Episode {ep_num:>3}: {winner} wins  "
                          f"({name1} rounds: {m1_wins} | {name2} rounds: {m2_wins})")

                obs, info = env.reset()
                frames.clear()
                game_on = False

    env.close()

    total = wins[name1] + wins[name2] + wins["draw"]
    m1_rate = wins[name1] / total if total else 0.0
    m2_rate = wins[name2] / total if total else 0.0

    col = max(len(name1), len(name2), 5) + 2
    width = col + 18
    print()
    print("=" * width)
    print(f"  Results after {episodes} episode(s)")
    print("=" * width)
    print(f"  {name1:<{col}}: {wins[name1]:>4}  ({m1_rate:.1%})")
    print(f"  {name2:<{col}}: {wins[name2]:>4}  ({m2_rate:.1%})")
    print(f"  {'Draw':<{col}}: {wins['draw']:>4}")
    print("=" * width)

    return wins


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run two trained PPO models against each other in SF2.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model1", required=True,
                        help="Path to Player 1 model (.zip)")
    parser.add_argument("--name1", default="Model 1",
                        help="Display name for Player 1")
    parser.add_argument("--model2", required=True,
                        help="Path to Player 2 model (.zip)")
    parser.add_argument("--name2", default="Model 2",
                        help="Display name for Player 2")
    parser.add_argument("--state", default=DEFAULT_STATE,
                        help="Path to game .state file")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of matches to play")
    parser.add_argument("--render", action="store_true",
                        help="Open a display window")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-episode outcomes")
    parser.add_argument("--debug-actions", action="store_true",
                        help="Print both agents' actions every step")
    parser.add_argument("--debug-emulator", action="store_true",
                        help="Print the raw button array sent to the emulator every frame")
    parser.add_argument("--list-states", action="store_true",
                        help="List available state files and exit")

    args = parser.parse_args()

    if args.list_states:
        states = sorted(f for f in os.listdir(STATES_DIR) if f.endswith(".state"))
        print(f"Available states in {STATES_DIR}:")
        for s in states:
            print(f"  {s}")
        return

    fight(
        model1_path=args.model1,
        model2_path=args.model2,
        state=args.state,
        episodes=args.episodes,
        rendering=args.render,
        verbose=args.verbose,
        debug_actions=args.debug_actions,
        debug_emulator=args.debug_emulator,
        name1=args.name1,
        name2=args.name2,
    )


if __name__ == "__main__":
    main()
