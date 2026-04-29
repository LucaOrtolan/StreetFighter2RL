"""
debug_action.py — Send one button-combination and watch what it does.

Starts the game with rendering, waits for the round to begin (warmup no-ops),
then repeats the chosen action for --steps agent-steps so you can see its effect.
Per-step reward/HP are printed to help diagnose what the move does.

Usage
-----
    # By button string
    python scripts/debug_action.py --action "B+A+UP+RIGHT+X"

    # By index into the built-in KNOWN_ACTIONS list
    python scripts/debug_action.py --index 5

    # Print all known actions and their effective buttons (after START filter)
    python scripts/debug_action.py --list

    # Custom state file, more steps, larger no-op gap between each repeat
    python scripts/debug_action.py --action "B+A+UP+RIGHT+X" \\
        --state data/states/ryu_vs_ken_8.state --steps 60 --gap 5
"""


# python debug_action.py --index 0

import sys
import os
import argparse

import numpy as np
import stable_retro as retro

SCRIPTS_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)
sys.path.insert(0, SCRIPTS_DIR)

from const   import BUTTONS, sf_game  # noqa: E402
from wrapper import SFWrapper          # noqa: E402

DEFAULT_STATE = os.path.join(PROJECT_ROOT, "data", "states", "ryu_vs_ryu_8.state")

# ---------------------------------------------------------------------------
# Button combinations captured from training logs — fill in as you identify them
# ---------------------------------------------------------------------------
KNOWN_ACTIONS = [
    "B+A+MODE+START+UP+RIGHT+C+Z",    # 0
    "B+A+MODE+START+UP+RIGHT+Z",       # 1
    "START+UP+LEFT+Y",                 # 2
    "B+A+START+UP+RIGHT+C+Y+Z",        # 3
    "B+A+START+UP+RIGHT+Y+Z",          # 4
    "B+A+UP+RIGHT+C+Z",                # 5
    "START+UP+RIGHT+Z",                # 6
    "A+MODE+UP+RIGHT+C+Z",             # 7
    "B+A+START+UP+RIGHT+C+Z",          # 8
    "A+START+UP+RIGHT+Z",              # 9
    "A+MODE+START+UP+RIGHT+Z",         # 10
    "UP+LEFT+RIGHT",                   # 11
    "B+A+START+UP+RIGHT+X",            # 12
    "B+A+START+UP+RIGHT+Z",            # 13
    "B+A+UP+DOWN+LEFT+RIGHT+C+Z",      # 14
    "B+A+UP+RIGHT+C",                  # 15
    "B+UP+LEFT+RIGHT+X",               # 16
    "A+START+UP+RIGHT+Y+Z",            # 17
    "B+A+START+UP+RIGHT+Y",            # 18
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_action(action_str: str) -> np.ndarray:
    """
    Convert a '+'-separated button string into a 15-bit action vector.

    Bits 0-11 : base buttons aligned with BUTTONS.
    Bits 12-14: combo bits set to [1,1,1] = index 7 (>= 6 = no special move).
    Note: START (index 3) is always zeroed inside SFWrapper.step(), so it has
    no effect — but it still shows up in the raw vector as requested by the user.
    """
    pressed = {b.strip() for b in action_str.upper().split("+")}
    base    = [1 if b in pressed else 0 for b in BUTTONS]
    vec     = np.array(base + [1, 1, 1], dtype=np.int8)  # combo=111 → no-op
    return vec


def effective_buttons(vec: np.ndarray) -> str:
    """Return the button string actually sent to the emulator (START stripped)."""
    v = vec.copy()
    v[3] = 0  # mirror SFWrapper's START filter
    pressed = [BUTTONS[i] for i in range(12) if v[i]]
    return "+".join(pressed) if pressed else "NO-OP"


def make_env(state: str) -> SFWrapper:
    retro_env = retro.make(
        game=sf_game,
        state=state,
        use_restricted_actions=retro.Actions.FILTERED,
        obs_type=retro.Observations.IMAGE,
        render_mode="human",
    )
    return SFWrapper(
        retro_env,
        side="left",
        reset_type="never",   # don't auto-reset so we can keep watching
        rendering=True,
        enable_combo=True,
        verbose=True,
    )


# ---------------------------------------------------------------------------
# Main run
# ---------------------------------------------------------------------------

def run(action_str: str, state: str, warmup: int, steps: int, gap: int) -> None:
    vec       = parse_action(action_str)
    effective = effective_buttons(vec)
    noop      = np.zeros(15, dtype=np.int8)
    noop[-3:] = 1  # no-op combo bits

    print()
    print(f"  Requested : {action_str}")
    print(f"  Effective : {effective}  (START filtered by wrapper)")
    print(f"  Binary    : {vec.tolist()}")
    print(f"  Warmup    : {warmup} no-op steps  |  Repeat : {steps}  |  Gap : {gap}")
    print()

    env      = make_env(state)
    obs, _   = env.reset()

    print(f"Warming up ({warmup} no-op steps)…")
    for _ in range(warmup):
        obs, _, done, _, _ = env.step(noop)
        if done:
            obs, _ = env.reset()

    print(f"Sending '{effective}' for {steps} steps (gap={gap} no-ops between each)…")
    print(f"  {'step':>4}  {'reward':>8}  {'player HP':>9}  {'enemy HP':>8}")
    print(f"  {'-'*4}  {'-'*8}  {'-'*9}  {'-'*8}")

    for i in range(steps):
        obs, reward, done, _, info = env.step(vec)
        hp      = info.get("health",       "?")
        enemy   = info.get("enemy_health", "?")
        print(f"  {i+1:>4}  {reward:>+8.4f}  {hp:>9}  {enemy:>8}")
        if done:
            print("  [episode ended]")
            break
        for _ in range(gap):
            obs, _, done2, _, _ = env.step(noop)
            if done2:
                break

    print("\nAction sequence complete. Keeping window open — press Ctrl+C to exit.")
    try:
        while True:
            _, _, done, _, _ = env.step(noop)
            if done:
                obs, _ = env.reset()
    except KeyboardInterrupt:
        pass

    env.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send a single button-combination action and observe it in SF2.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    src = parser.add_mutually_exclusive_group()
    src.add_argument(
        "--action",
        default=None,
        help="Button combo string, e.g. 'B+A+UP+RIGHT+X'",
    )
    src.add_argument(
        "--index",
        type=int,
        default=None,
        help="Index into KNOWN_ACTIONS (0-based); use --list to see them",
    )

    parser.add_argument("--list",   action="store_true",
                        help="Print all known actions and exit")
    parser.add_argument("--state",  default=DEFAULT_STATE,
                        help="Path to .state file")
    parser.add_argument("--warmup", type=int, default=40,
                        help="No-op steps before the action (lets the round load)")
    parser.add_argument("--steps",  type=int, default=30,
                        help="Number of times to repeat the action")
    parser.add_argument("--gap",    type=int, default=3,
                        help="No-op frames between each action repeat")

    args = parser.parse_args()

    if args.list:
        print(f"\n{'#':>3}  {'Requested string':<45}  Effective (START stripped)")
        print(f"  {'-'*3}  {'-'*45}  {'-'*40}")
        for i, a in enumerate(KNOWN_ACTIONS):
            print(f"  [{i:>2}]  {a:<45}  {effective_buttons(parse_action(a))}")
        print()
        return

    if args.index is not None:
        if not (0 <= args.index < len(KNOWN_ACTIONS)):
            parser.error(f"--index must be in 0..{len(KNOWN_ACTIONS)-1}")
        action_str = KNOWN_ACTIONS[args.index]
    elif args.action:
        action_str = args.action
    else:
        parser.error("Specify --action <string>, --index <n>, or --list")

    run(action_str, args.state, args.warmup, args.steps, args.gap)


if __name__ == "__main__":
    main()
