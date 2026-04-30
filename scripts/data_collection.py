"""
data_collection.py — Collect match analytics across four model matchups.

Runs N episodes for each of:
  1. model1 (P1) vs CPU
  2. model2 (P1) vs CPU
  3. model1 (P1) vs model2 (P2)
  4. model2 (P1) vs model1 (P2)

Outputs three CSV files:
  matches.csv — one row per completed match (rounds won, match winner)
  rounds.csv  — one row per completed round (final HP, HP ratio, round winner)
  actions.csv — one row per (matchup, episode, round, player, action) with step count

Round boundaries are detected the same way FightEnv does it: watching for the
during_transition flag to flip False → True, then reading the final HP from info.

Usage
-----
    python data_collection.py \\
        --model1 path/to/model1.zip \\
        --model2 path/to/model2.zip \\
        --episodes 20 \\
        --output-dir ../logs/analytics
"""

import sys
import os
import csv
import argparse
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
from stable_baselines3 import PPO

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPTS_DIR)
sys.path.insert(0, SCRIPTS_DIR)

from const import sf_game              # noqa: E402
from fight import FightEnv, make_fight_env, make_cpu_env, _decode_action  # noqa: E402

STATES_DIR = os.path.join(PROJECT_ROOT, "data", "states")
DEFAULT_CPU_STATE = os.path.join(STATES_DIR, "ryu_vs_ryu_8.state")
DEFAULT_PVP_STATE = os.path.join(STATES_DIR, "player_ryu_vs_player_ryu.state")
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "logs", "analytics")

FULL_HP = 176


# ---------------------------------------------------------------------------
# Analytics collector
# ---------------------------------------------------------------------------

class AnalyticsCollector:
    """
    Accumulates per-round, per-match, and per-action data across all matchups.

    Round boundaries are detected externally (fight loops) and reported via
    record_round_end(). Action counts are accumulated per round and flushed
    to action_rows when each round ends. Match outcomes are derived from the
    accumulated round win counts.
    """

    def __init__(self):
        self.round_rows = []
        self.match_rows = []
        self.action_rows = []

        self._matchup = self._p1_name = self._p2_name = None
        self._ep = 0
        self._round = 0
        self._p1_actions: dict[str, int] = {}
        self._p2_actions: dict[str, int] = {}
        self._p1_rounds = 0
        self._p2_rounds = 0

    def start_episode(self, matchup: str, p1_name: str, p2_name: str, ep: int):
        self._matchup = matchup
        self._p1_name = p1_name
        self._p2_name = p2_name
        self._ep = ep
        self._round = 0
        self._p1_actions = {}
        self._p2_actions = {}
        self._p1_rounds = 0
        self._p2_rounds = 0

    def record_action(self, p1_action: str, p2_action: str | None = None):
        """Record one agent step. p2_action=None for vs-CPU (CPU inputs unknown)."""
        self._p1_actions[p1_action] = self._p1_actions.get(p1_action, 0) + 1
        if p2_action is not None:
            self._p2_actions[p2_action] = self._p2_actions.get(p2_action, 0) + 1

    def record_round_end(self, p1_hp: int, p2_hp: int):
        """
        Called when during_transition flips False → True.

        HP values at transition: the loser's HP is negative (clamped to 0 here).
        Action counts accumulated since the last round end are flushed and reset.
        """
        self._round += 1
        p1c = max(0, p1_hp)
        p2c = max(0, p2_hp)

        if p1_hp > p2_hp:
            winner = "p1"
            self._p1_rounds += 1
        elif p2_hp > p1_hp:
            winner = "p2"
            self._p2_rounds += 1
        else:
            winner = "draw"

        self.round_rows.append({
            "matchup":     self._matchup,
            "p1_name":     self._p1_name,
            "p2_name":     self._p2_name,
            "episode":     self._ep,
            "round":       self._round,
            "round_winner": winner,
            "p1_hp_final": p1c,
            "p2_hp_final": p2c,
            "p1_hp_ratio": round(p1c / FULL_HP, 4),
            "p2_hp_ratio": round(p2c / FULL_HP, 4),
        })

        for action, count in self._p1_actions.items():
            self.action_rows.append({
                "matchup":     self._matchup,
                "p1_name":     self._p1_name,
                "p2_name":     self._p2_name,
                "episode":     self._ep,
                "round":       self._round,
                "player":      "p1",
                "player_name": self._p1_name,
                "action":      action,
                "count":       count,
            })
        for action, count in self._p2_actions.items():
            self.action_rows.append({
                "matchup":     self._matchup,
                "p1_name":     self._p1_name,
                "p2_name":     self._p2_name,
                "episode":     self._ep,
                "round":       self._round,
                "player":      "p2",
                "player_name": self._p2_name,
                "action":      action,
                "count":       count,
            })

        self._p1_actions = {}
        self._p2_actions = {}

    def record_match_end(self):
        p1r, p2r = self._p1_rounds, self._p2_rounds
        winner = "p1" if p1r > p2r else ("p2" if p2r > p1r else "draw")
        self.match_rows.append({
            "matchup":      self._matchup,
            "p1_name":      self._p1_name,
            "p2_name":      self._p2_name,
            "episode":      self._ep,
            "p1_rounds_won": p1r,
            "p2_rounds_won": p2r,
            "match_winner": winner,
        })

    def save(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        _write_csv(os.path.join(output_dir, "matches.csv"), self.match_rows)
        _write_csv(os.path.join(output_dir, "rounds.csv"),  self.round_rows)
        _write_csv(os.path.join(output_dir, "actions.csv"), self.action_rows)
        print(f"\nCSVs written to {output_dir}/")
        print(f"  matches.csv : {len(self.match_rows)} rows")
        print(f"  rounds.csv  : {len(self.round_rows)} rows")
        print(f"  actions.csv : {len(self.action_rows)} rows")


def _write_csv(path: str, rows: list[dict]):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Fight loops
# ---------------------------------------------------------------------------

def collect_vs_cpu(
    model,
    state: str,
    episodes: int,
    analytics: AnalyticsCollector,
    matchup: str,
    p1_name: str,
    rendering: bool = False,
    deterministic: bool = True,
    quiet: bool = False,
):
    """
    Run model (P1) vs CPU for `episodes` complete best-of-3 matches.

    Uses make_cpu_env() — same env setup as fight.py.
    CPU controls P2 natively; its button inputs are not recorded.

    Round boundaries are detected by watching during_transition flip False → True.
    Actions are only recorded during active gameplay (not in transition).
    """
    env = make_cpu_env(state, rendering=rendering)

    MAX_STEPS_PER_EPISODE = 5000

    obs, info = env.reset()

    for ep in range(episodes):
        analytics.start_episode(matchup, p1_name, "CPU", ep + 1)
        game_on = True
        prev_in_transition = True   # matches reset() → during_transition=True
        step = 0

        while game_on:
            step += 1
            if step > MAX_STEPS_PER_EPISODE:
                print(f"  WARNING: {matchup} ep {ep + 1} exceeded {MAX_STEPS_PER_EPISODE} steps — forcing reset", flush=True)
                analytics.record_match_end()
                obs, info = env.reset()
                game_on = False
                continue

            action = model.predict(obs, deterministic=deterministic)[0]

            # Record action only during active gameplay (not in transition).
            if not prev_in_transition:
                analytics.record_action(_decode_action(action, env.action_dim))

            obs, _reward, done, truncated, info = env.step(action)

            # Round end: during_transition just flipped False → True.
            if not prev_in_transition and env.during_transition:
                analytics.record_round_end(
                    info.get("health", 0),
                    info.get("enemy_health", 0),
                )
            prev_in_transition = env.during_transition

            # End the match when either side wins 2 rounds (same logic as FightEnv).
            # Using round counts rather than relying solely on `done` avoids a race
            # condition where the arcade ladder resets victory counters in RAM before
            # the env can set done=True, which causes extra rounds to be counted.
            match_over = (
                analytics._p1_rounds >= 2 or analytics._p2_rounds >= 2
                or done or truncated
            )
            if match_over:
                analytics.record_match_end()
                last = analytics.match_rows[-1]
                if not quiet:
                    print(f"  [{matchup}] ep {ep + 1:>4}:  winner={last['match_winner']:<4}  "
                          f"{p1_name} {last['p1_rounds_won']} – CPU {last['p2_rounds_won']}", flush=True)
                obs, info = env.reset()
                game_on = False

    env.close()


def collect_pvp(
    model1,
    model2,
    state: str,
    episodes: int,
    analytics: AnalyticsCollector,
    matchup: str,
    p1_name: str,
    p2_name: str,
    rendering: bool = False,
    deterministic: bool = True,
    quiet: bool = False,
):
    """
    Run model1 (P1) vs model2 (P2) for `episodes` complete best-of-3 matches.

    Uses FightEnv (side="both"), which tracks p1_rounds_won / p2_rounds_won.
    Both players' actions are decoded and recorded each step.

    Round boundaries are detected by watching during_transition flip False → True,
    matching the same logic inside FightEnv.step().
    """
    env = make_fight_env(state, rendering=rendering)

    MAX_STEPS_PER_EPISODE = 5000

    obs, info = env.reset()

    for ep in range(episodes):
        analytics.start_episode(matchup, p1_name, p2_name, ep + 1)
        game_on = True
        prev_in_transition = True
        step = 0

        while game_on:
            step += 1
            if step > MAX_STEPS_PER_EPISODE:
                print(f"  WARNING: {matchup} ep {ep + 1} exceeded {MAX_STEPS_PER_EPISODE} steps — forcing reset", flush=True)
                analytics.record_match_end()
                obs, info = env.reset()
                game_on = False
                continue

            action1 = model1.predict(obs, deterministic=deterministic)[0]
            action2 = model2.predict(obs, deterministic=deterministic)[0]

            if not prev_in_transition:
                analytics.record_action(
                    _decode_action(action1, env.action_dim),
                    _decode_action(action2, env.action_dim),
                )

            combined = np.hstack([action1, action2])
            obs, _r1, _r2, done, truncated, info = env.step(combined)

            if not prev_in_transition and env.during_transition:
                analytics.record_round_end(
                    info.get("health", 0),
                    info.get("enemy_health", 0),
                )
            prev_in_transition = env.during_transition

            match_over = (
                env.p1_rounds_won == 2 or env.p2_rounds_won == 2
                or done or truncated
            )
            if match_over:
                analytics.record_match_end()
                last = analytics.match_rows[-1]
                if not quiet:
                    print(f"  [{matchup}] ep {ep + 1:>4}:  winner={last['match_winner']:<8}  "
                          f"{p1_name} {last['p1_rounds_won']} – {p2_name} {last['p2_rounds_won']}", flush=True)
                obs, info = env.reset()
                game_on = False

    env.close()


# ---------------------------------------------------------------------------
# Parallel shard runner
# ---------------------------------------------------------------------------

def _run_shard(args):
    """
    Subprocess worker. Loads models from paths, runs a slice of episodes,
    offsets episode numbers to avoid collisions on merge, returns rows.
    Must be a module-level function so multiprocessing can pickle it.
    """
    (mode, model1_path, model2_path, state,
     n_episodes, matchup, p1_name, p2_name,
     rendering, deterministic, ep_offset) = args

    model1 = PPO.load(model1_path)
    model2 = PPO.load(model2_path) if model2_path else None

    collector = AnalyticsCollector()

    if mode == "cpu":
        collect_vs_cpu(
            model1, state, n_episodes, collector,
            matchup=matchup, p1_name=p1_name,
            rendering=rendering, deterministic=deterministic,
            quiet=False,
        )
    else:
        collect_pvp(
            model1, model2, state, n_episodes, collector,
            matchup=matchup, p1_name=p1_name, p2_name=p2_name,
            rendering=rendering, deterministic=deterministic,
            quiet=False,
        )

    for row in collector.match_rows + collector.round_rows + collector.action_rows:
        row["episode"] += ep_offset

    return matchup, p1_name, p2_name, collector.match_rows, collector.round_rows, collector.action_rows


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------

def print_summary(match_rows: list[dict], round_rows: list[dict]):
    match_stats  = defaultdict(lambda: {"p1": 0, "p2": 0, "draw": 0, "total": 0})
    round_stats  = defaultdict(lambda: {"p1": 0, "p2": 0, "draw": 0, "total": 0})

    for r in match_rows:
        s = match_stats[r["matchup"]]
        s[r["match_winner"]] += 1
        s["total"] += 1

    for r in round_rows:
        s = round_stats[r["matchup"]]
        s[r["round_winner"]] += 1
        s["total"] += 1

    print("\n" + "=" * 62)
    print("  SUMMARY")
    print("=" * 62)
    for matchup in match_stats:
        ms = match_stats[matchup]
        rs = round_stats[matchup]
        mt = ms["total"] or 1
        rt = rs["total"] or 1
        p1_name = matchup.split("_vs_")[0]
        p2_name = matchup.split("_vs_")[1] if "_vs_" in matchup else "P2"
        print(f"\n  {matchup}")
        print(f"    Match winrate  — {p1_name}: {ms['p1'] / mt:.1%} ({ms['p1']}/{mt})"
              f"   {p2_name}: {ms['p2'] / mt:.1%} ({ms['p2']}/{mt})"
              + (f"   draw: {ms['draw']}" if ms["draw"] else ""))
        print(f"    Round winrate  — {p1_name}: {rs['p1'] / rt:.1%} ({rs['p1']}/{rt})"
              f"   {p2_name}: {rs['p2'] / rt:.1%} ({rs['p2']}/{rt})"
              + (f"   draw: {rs['draw']}" if rs["draw"] else ""))
    print("=" * 62)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Collect SF2 match analytics for four model matchups.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model1",      required=True,
                        help="Path to model 1 .zip")
    parser.add_argument("--name1",       default="Luca",
                        help="Display name for model 1")
    parser.add_argument("--model2",      required=True,
                        help="Path to model 2 .zip")
    parser.add_argument("--name2",       default="Krystal",
                        help="Display name for model 2")
    parser.add_argument("--episodes",    type=int, default=20,
                        help="Matches per matchup")
    parser.add_argument("--cpu-state",   default=DEFAULT_CPU_STATE,
                        help="State file for vs-CPU matchups")
    parser.add_argument("--pvp-state",   default=DEFAULT_PVP_STATE,
                        help="State file for model vs model matchups")
    parser.add_argument("--output-dir",  default=DEFAULT_OUTPUT_DIR,
                        help="Directory to write CSV files into")
    parser.add_argument("--render",      action="store_true",
                        help="Open a display window")
    parser.add_argument("--stochastic",  action="store_true",
                        help="Sample actions from the policy distribution instead of taking the mean")
    parser.add_argument("--n-envs",      type=int, default=1,
                        help="Parallel workers per matchup (also parallelises all 4 matchups simultaneously)")
    args = parser.parse_args()

    print(f"Episodes per matchup : {args.episodes}")
    print(f"CPU state            : {args.cpu_state}")
    print(f"PvP state            : {args.pvp_state}")
    print(f"Output               : {args.output_dir}")
    print(f"Workers per matchup  : {args.n_envs}")

    analytics = AnalyticsCollector()

    # Each entry: (matchup_id, p1_name, p2_name, mode, model1_path, model2_path, state)
    matchup_defs = [
        (f"{args.name1}_vs_CPU",          args.name1, "CPU",        "cpu", args.model1, None,         args.cpu_state),
        (f"{args.name2}_vs_CPU",          args.name2, "CPU",        "cpu", args.model2, None,         args.cpu_state),
        (f"{args.name1}_vs_{args.name2}", args.name1, args.name2,   "pvp", args.model1, args.model2,  args.pvp_state),
        (f"{args.name2}_vs_{args.name1}", args.name2, args.name1,   "pvp", args.model2, args.model1,  args.pvp_state),
    ]

    if args.n_envs == 1:
        # --- Sequential (original behaviour) ---
        print(f"Loading {args.name1}: {args.model1}")
        model1 = PPO.load(args.model1)
        print(f"Loading {args.name2}: {args.model2}")
        model2 = PPO.load(args.model2)

        model_map = {args.model1: model1, args.model2: model2}

        for matchup, p1_name, p2_name, mode, m1_path, m2_path, state in matchup_defs:
            sep = "=" * 56
            print(f"\n{sep}")
            print(f"  {p1_name} (P1) vs {p2_name} (P2)   [{args.episodes} episodes]")
            print(sep)
            if mode == "cpu":
                collect_vs_cpu(
                    model_map[m1_path], state, args.episodes, analytics,
                    matchup=matchup, p1_name=p1_name,
                    rendering=args.render, deterministic=not args.stochastic,
                )
            else:
                collect_pvp(
                    model_map[m1_path], model_map[m2_path], state, args.episodes, analytics,
                    matchup=matchup, p1_name=p1_name, p2_name=p2_name,
                    rendering=args.render, deterministic=not args.stochastic,
                )

            rows = [r for r in analytics.match_rows if r["matchup"] == matchup]
            total = len(rows)
            p1w = sum(1 for r in rows if r["match_winner"] == "p1")
            p2w = sum(1 for r in rows if r["match_winner"] == "p2")
            drw = sum(1 for r in rows if r["match_winner"] == "draw")
            print(f"\n  Result: {p1_name} {p1w/total:.1%}  {p2_name} {p2w/total:.1%}"
                  + (f"  draw {drw/total:.1%}" if drw else "")
                  + f"  ({total} matches)")

    else:
        # --- Parallel: 4 matchups × n_envs workers = n_envs*4 total processes ---
        n = args.n_envs
        all_shards = []
        shard_mu_index = []  # index into matchup_defs for each shard

        for mu_idx, (matchup, p1_name, p2_name, mode, m1_path, m2_path, state) in enumerate(matchup_defs):
            base, rem = divmod(args.episodes, n)
            for i in range(n):
                n_ep = base + (1 if i < rem else 0)
                ep_offset = i * base + min(i, rem)
                all_shards.append((
                    mode, m1_path, m2_path, state,
                    n_ep, matchup, p1_name, p2_name,
                    args.render, not args.stochastic, ep_offset,
                ))
                shard_mu_index.append(mu_idx)

        total_workers = len(all_shards)
        print(f"\nSpawning {total_workers} workers ({n} per matchup × 4 matchups)...")

        # Track accumulated shards per matchup so we can print as each one finishes.
        mu_data     = defaultdict(lambda: {"mr": [], "rr": [], "ar": [], "p1": "", "p2": "", "done": 0})
        mu_shards   = defaultdict(int)
        for mu_idx in shard_mu_index:
            mu_shards[matchup_defs[mu_idx][0]] += 1

        with Pool(total_workers, maxtasksperchild=1) as pool:
            for matchup, p1_name, p2_name, mr, rr, ar in pool.imap_unordered(_run_shard, all_shards):
                d = mu_data[matchup]
                d["mr"] += mr
                d["rr"] += rr
                d["ar"] += ar
                d["p1"] = p1_name
                d["p2"] = p2_name
                d["done"] += 1

                n_done  = d["done"]
                n_total = mu_shards[matchup]

                if n_done < n_total:
                    print(f"  [{matchup}] shard {n_done}/{n_total} done "
                          f"({len(d['mr'])} matches so far)")
                else:
                    # All shards for this matchup complete — print final result.
                    total = len(d["mr"])
                    p1w = sum(1 for r in d["mr"] if r["match_winner"] == "p1")
                    p2w = sum(1 for r in d["mr"] if r["match_winner"] == "p2")
                    drw = total - p1w - p2w
                    print(f"\n  {matchup} complete: {p1_name} {p1w/total:.1%}  "
                          f"{p2_name} {p2w/total:.1%}"
                          + (f"  draw {drw/total:.1%}" if drw else "")
                          + f"  ({total} matches)")

        for matchup, p1_name, p2_name, *_ in matchup_defs:
            d = mu_data[matchup]
            analytics.match_rows += d["mr"]
            analytics.round_rows += d["rr"]
            analytics.action_rows += d["ar"]

    print_summary(analytics.match_rows, analytics.round_rows)
    analytics.save(args.output_dir)


if __name__ == "__main__":
    main()
