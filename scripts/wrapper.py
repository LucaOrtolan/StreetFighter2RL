"""
wrapper.py — Gymnasium environment wrappers for Street Fighter II RL.

SFWrapper
---------
The core wrapper around the stable-retro Street Fighter II environment.
It handles:
  • Observation pre-processing  – frame stacking, downsampling, grayscale-cycling
  • Action space definition     – raw MultiBinary or discrete (transformed) actions,
                                   plus optional special-move (combo) actions
  • Reward shaping              – dense HP-delta rewards scaled by an aggression
                                   coefficient, plus terminal win/loss bonuses
  • Episode management          – round and match status tracking, configurable
                                   reset granularity ("round" | "match" | "never")
  • Two-player support          – "both" side lets a single agent control both
                                   characters simultaneously

Monitor2P
---------
Extends SB3's Monitor wrapper to track per-episode rewards for *both* players
when running in two-player mode (side="both").
"""

from const import *
from gymnasium import Wrapper, Env
from gymnasium.wrappers import FrameStackObservation
from gymnasium.spaces import Box, MultiDiscrete, MultiBinary
import numpy as np
import gzip
import time
import math
from stable_baselines3.common.monitor import Monitor
from typing import Optional, Tuple, Union
from stable_baselines3.common.type_aliases import GymObs, GymStepReturn
import cv2
from rewards import get_rewards

class SFWrapper(Wrapper):
    """
    Custom Gymnasium wrapper for Street Fighter II (via stable-retro).

    Converts the raw emulator environment into a form suitable for deep RL:
    - Stacks `num_stack` frames and sub-samples every `num_step_frames//2`
      frames so the agent sees motion without excessive redundancy.
    - Downsamples each frame from (224, 256) to (100, 128) by taking every
      other pixel (slice [::2, ::2]).
    - Cycles through R/G/B channels across stacked frames so colour information
      is preserved despite converting to a single-channel representation.
    - Maps 3-bit combo flags in the action vector to full special-move sequences.
    - Computes a shaped reward that rewards dealing damage and penalises
      taking damage, with an `aggresive_coeff` multiplier on enemy damage.

    Parameters
    ----------
    env : gymnasium.Env
        The raw retro environment returned by `retro.make(...)`.
    side : str
        Which player the agent controls: "left", "right", or "both".
    reset_type : str
        Granularity of episode resets:
          "round"  – reset after every individual round (fastest learning signal)
          "match"  – reset only after a full best-of-3 match
          "never"  – let the game run indefinitely (curriculum / evaluation use)
    init_level : int
        Starting arcade ladder level (1 = easiest CPU opponent).
    rendering : bool
        Whether to open a display window for human observation.
    num_stack : int
        Number of raw frames to keep in the frame buffer.
    num_step_frames : int
        Number of emulator frames advanced per agent step.
        Must equal `len(SF_COMBOS[i])` (currently 8) so combo sequences fit.
    state_dir : str or None
        Directory for saving/loading game states; used by `save_state_to_file`.
    verbose : bool
        If True, print round/match outcomes to stdout.
    enable_combo : bool
        If True, the last 3 bits of the action vector encode a combo index.
    null_combo : bool
        If True, allocate combo bits but always send a no-op combo (useful for
        ablation studies keeping the action-space size constant).
    transform_action : bool
        If True, use a MultiDiscrete action space (single integer per player)
        instead of MultiBinary; requires a custom `action_transformer`.
    """
    def __init__(self, env, side, reset_type="round", init_level=1, rendering=False, num_stack=12, num_step_frames=8, state_dir=None, verbose=False, enable_combo=True, null_combo=False, transform_action=False, aggresive_coeff=3.0, dense_coeff=1.0):
        super(SFWrapper, self).__init__(env)

        # Wrap the base env in SB3's FrameStackObservation so we receive
        # a deque of the last `num_stack` raw RGB frames on every step/reset.
        self.env = FrameStackObservation(env, stack_size=num_stack)
    
        assert side in ['left', 'right', 'both'], "side should be 'left', 'right' or 'both'"
        self.side = side

        # Frame-stack / step parameters.
        self.num_stack = num_stack
        self.num_step_frames = num_step_frames

        # Reward shaping coefficients.
        # aggresive_coeff > 1 makes dealing damage more valuable than avoiding it.
        self.aggresive_coeff = aggresive_coeff
        # dense_coeff scales the per-step HP-delta reward (applied during a round).
        self.dense_coeff = dense_coeff

        # Cumulative emulator frame counter for the current episode.
        self.total_timesteps = 0

        # Maximum HP value for both characters (read from game RAM).
        self.full_hp = 176
        self.prev_agent_hp = self.full_hp   # HP at the previous agent step.
        self.prev_enemy_hp = self.full_hp

        # Seconds to sleep between rendered frames (controls playback speed).
        self.frame_rate = 0.01

        # ---------------------------------------------------------------------------
        # Observation space
        # Shape: (H, W, C) = (100, 128, num_stack // (num_step_frames // 2))
        # Each channel is one downsampled, channel-cycled frame from the stack.
        # The sub-sampling index `range(0, num_stack, num_step_frames // 2)` gives
        # the frames spaced `num_step_frames//2` apart, providing temporal coverage
        # without redundancy.
        # ---------------------------------------------------------------------------
        self.observation_space = Box(
            low=0, high=255,
            shape=(100, 128, len(range(0, self.num_stack, self.num_step_frames // 2))),
            dtype=np.uint8,
        )

        # ---------------------------------------------------------------------------
        # Action space
        # Base: 12 binary buttons (one-hot over BUTTONS).
        # Combo extension: +3 bits when enable_combo or null_combo is True.
        #   The 3 extra bits form a 3-bit integer (0–7) indexing into SF_COMBOS.
        #   Values >= len(SF_COMBOS) (6) are treated as "no combo".
        # ---------------------------------------------------------------------------
        self.action_dim = 12 + 3 if (enable_combo or null_combo) else 12  # 3 bits for combos

        if transform_action:
            # Discrete mode: agent outputs a single integer per player.
            # The integer is split into direction + attack (+ optional combo).
            if enable_combo or null_combo:
                self.n_actions = MultiDiscrete([len(DIRECTIONS_BUTTONS) + len(ATTACKS_BUTTONS) + len(SF_COMBOS)])
            else: 
                self.action_space = MultiDiscrete([len(DIRECTIONS_BUTTONS) + len(ATTACKS_BUTTONS)])

            def action_transformer(action):
                """
                Convert a discrete integer action into a binary button array.

                Integers in [0, len(DIRECTIONS))                   → direction only
                Integers in [len(DIRECTIONS), len(DIR)+len(ATK))   → attack + direction=[]
                Integers >= len(DIR)+len(ATK)                      → combo (3-bit index)
                """
                players_action = []
                for player_action in action:
                    if player_action >= len(DIRECTIONS_BUTTONS) + len(ATTACKS_BUTTONS):
                        # Combo action: clear the 12 base buttons,
                        # encode the combo index into 3 bits via binary_repr.
                        button_bits = [0 for _ in range(12)]
                        combo_bits = [int(i) for i in np.binary_repr(player_action)]
                    elif player_action >= len(DIRECTIONS_BUTTONS):
                        # Attack action (no direction pressed).
                        direction_buttons = []
                        attack_buttons = ATTACKS_BUTTONS[player_action - len(DIRECTIONS_BUTTONS)]
                        button_bits = [int(b in direction_buttons + attack_buttons) for b in BUTTONS]
                        combo_bits = [1 for _ in range(3)]  # combo=111 → no-op combo
                    else:
                        # Directional action (no attack pressed).
                        direction_buttons = DIRECTIONS_BUTTONS[player_action]
                        attack_buttons = []
                        button_bits = [int(b in direction_buttons + attack_buttons) for b in BUTTONS]
                        combo_bits = [1 for _ in range(3)]  # combo=111 → no-op combo

                players_action.append(np.array(button_bits + combo_bits))
                return np.hstack(players_action)

            self.action_transformer = action_transformer

        else:
            # MultiBinary mode: each bit corresponds directly to a button / combo bit.
            self.action_space = MultiBinary(self.action_dim)
            self.action_transformer = None

        # Episode / environment configuration.
        self.reset_type = reset_type
        self.rendering = rendering
        self.init_level = init_level
        self.state_dir = state_dir
        self.verbose = verbose
        self.enable_combo = enable_combo
        self.null_combo = null_combo

        # EXP: damage_taken tuning
        self.max_damage_taken = 0.0
        self.max_damage_dealt = 0.0

    def save_state_to_file(self, name="test.state"):
        """
        Dump the current emulator save-state to a gzip-compressed .state file.

        Useful during curriculum learning to capture a specific game snapshot
        that can be reloaded as a training starting point.

        Parameters
        ----------
        name : str
            Filename (relative to `self.state_dir`) to write the state to.
        """
        content = self.env.em.get_state()
        print(f"Save state to {os.path.join(self.state_dir, name)}")
        with gzip.open(os.path.join(self.state_dir, name), 'wb') as f:
            f.write(content)

    def _get_obs(self, obs):
        """
        Pre-process the raw stacked-frame observation for the neural network.

        Steps:
        1. Sub-sample the frame stack: pick every `num_step_frames // 2`-th frame.
           With num_stack=12 and num_step_frames=8, this yields 3 frames.
        2. Spatially downsample each frame by 2× in H and W (slice [::2, ::2]).
           224×320 → 112×160 after retro crops → ~100×128 after wrapper.
        3. Cycle through R/G/B channels (index `i % 3`) so that successive
           channels in the output encode different colour channels, preserving
           colour information across the stack in a compact single-channel format.

        Parameters
        ----------
        obs : np.ndarray or dict
            Raw observation from FrameStackObservation.
            Shape when array: (num_stack, H, W, 3).

        Returns
        -------
        np.ndarray
            Shape (H//2, W//2, num_selected_frames) — the network input.
        """
        if isinstance(obs, dict):
            return {
                'observations': np.stack(
                    [o[::2, ::2, i % 3]
                     for (i, o) in enumerate(obs['observations'][::(self.num_step_frames // 2)])],
                    axis=-1,
                ),
                'actions': np.squeeze(obs['actions'][::(self.num_step_frames // 2)], axis=-1),
            }
        else:
            return np.stack(
                [o[::2, ::2, i % 3]
                 for (i, o) in enumerate(obs[::(self.num_step_frames // 2)])],
                axis=-1,
            )

    # ---------------------------------------------------------------------------
    # Core Gymnasium interface
    # ---------------------------------------------------------------------------

    def reset(self, **kwargs):
        """
        Reset the environment to the beginning of a new episode.

        Resets all HP tracking, status flags, level counter, and the frame
        buffer via the underlying FrameStackObservation wrapper.

        Returns
        -------
        obs : np.ndarray
            Pre-processed initial observation.
        info : dict
            Info dict from the underlying environment.
        """
        obs, info = self.env.reset()

        # Reset HP trackers to full health for both fighters.
        self.prev_agent_hp = self.full_hp
        self.prev_enemy_hp = self.full_hp

        # Restore level to the configured starting level.
        self.level = self.init_level
        self.save_state = False

        # match_status / round_status track whether we are inside an active
        # match/round (START_STATUS) or in the transition screen (END_STATUS).
        self.match_status = START_STATUS
        self.round_status = END_STATUS

        # during_transition=True while the game is showing a non-playable
        # transition (e.g. KO screen, continue screen).  No reward is given
        # and custom_done is determined only by win/loss counts.
        self.during_transition = True

        self.round_num = 0      # Rounds completed in the current match.
        self.extra_round = False  # True if a draw forced an extra deciding round.
        self.total_timesteps = 0

        # EXP: damage taken tuning
        self.max_damage_taken = 0.0
        self.max_damage_dealt = 0.0

        return self._get_obs(obs), info

    def update_status(self, info, bonus=False):
        """
        Update internal match/round state machines based on the latest game RAM.

        Called once per emulator frame inside `step()` to keep `match_status`
        and `round_status` accurate.  Both are two-state machines:

          round_status:  END_STATUS → START_STATUS when a new round begins
                                      (both HPs full, timer > 0)
                         START_STATUS → END_STATUS when a round ends
                                      (any HP < 0 or timer == 0)

          match_status:  END_STATUS → START_STATUS when a new match begins
                                      (both victory counters reset to 0)
                         START_STATUS → END_STATUS when a match ends
                                      (a player reaches 2 round wins, or
                                       max_round rounds played)

        Parameters
        ----------
        info : dict
            RAM-derived game state from stable-retro, containing keys:
            'health', 'enemy_health', 'matches_won', 'enemy_matches_won',
            'round_timer'.
        bonus : bool
            If True, treat this as a bonus stage (single-round match).
        """
        max_round = 1 if bonus else 3  # Best-of-1 for bonus, best-of-3 otherwise.

        agent_hp       = info["health"]
        enemy_hp       = info["enemy_health"]
        agent_victories  = info["matches_won"]
        enemy_victories  = info["enemy_matches_won"]
        round_countdown  = info["round_timer"]
        timesup = (round_countdown <= 0)

        # --- Match status state machine ---
        if self.match_status == END_STATUS and (agent_victories == 0 and enemy_victories == 0):
            # Both victory counters reset → a new match has started.
            self.match_status = START_STATUS
            self.save_state = True  # Flag to snapshot the state for curriculum.
            self.level += 1
            self.round_num = 0
            self.extra_round = False

        elif self.match_status == START_STATUS and (
            self.round_num == max_round
            or agent_victories == 2
            or enemy_victories == 2
        ):
            # A player reached 2 wins, or all rounds played → match is over.
            self.match_status = END_STATUS
            if self.verbose:
                if agent_victories < enemy_victories:
                    print(f"Level {self.level} is over and player loses")
                elif agent_victories > enemy_victories:
                    print(f"Level {self.level} is over and player wins")
                else:
                    print(f"Draw level {self.level}")
                    if (not bonus) and (not self.extra_round):
                        # Force one more round to break the tie.
                        self.match_status = START_STATUS
                        self.round_num -= 1
                        self.extra_round = True
                        print(f"One more round for draw level {self.level}")

        # --- Round status state machine ---
        if self.round_status == END_STATUS and (
            agent_hp == self.full_hp and enemy_hp == self.full_hp and round_countdown > 0
        ):
            # Both fighters at full HP and timer ticking → new round underway.
            self.round_status = START_STATUS
            self.prev_agent_hp = self.full_hp
            self.prev_enemy_hp = self.full_hp

        elif self.round_status == START_STATUS and ((agent_hp < 0 or enemy_hp < 0) or timesup):
            # A fighter's HP hit 0 (or < 0 due to simultaneous KO) or time ran out.
            self.round_status = END_STATUS
            self.round_num += 1
            if self.verbose:
                if agent_hp < enemy_hp:
                    print(f"The round is over and player loses")
                elif agent_hp > enemy_hp:
                    print(f"The round is over and player wins")
                else:
                    print(f"Draw round")

        # Sanity checks: match/round statuses must be consistent with each other.
        if self.match_status == END_STATUS:
            info["match"] = "start" if self.match_status == START_STATUS else "end"
            info["round"] = "start" if self.round_status == START_STATUS else "end"
            assert self.round_status == END_STATUS, info
        if self.round_status == START_STATUS:
            info["match"] = "start" if self.match_status == START_STATUS else "end"
            info["round"] = "start" if self.round_status == START_STATUS else "end"
            assert self.match_status == START_STATUS, info

    def step(self, action):
        """
        Advance the environment by one agent step (= `num_step_frames` emulator frames).

        The method:
        1. Optionally transforms the action via `action_transformer`.
        2. Decodes the combo bits (last 3 bits) into a full combo sequence, or
           repeats the base 12-button action for `num_step_frames` frames.
        3. Steps the emulator `num_step_frames` times, updating status each frame.
        4. Computes a shaped reward based on HP changes and win/loss outcome.
        5. Returns (obs, reward, done, truncated, info).

        Reward structure
        ----------------
        During a round:
          reward = dense_coeff × (aggresive_coeff × Δenemy_hp − Δagent_hp)
          (positive when the agent is dealing more damage than receiving)

        Round end — agent wins:
          reward = +full_hp^((agent_hp+1)/(full_hp+1)) × aggresive_coeff

        Round end — agent loses:
          reward = −full_hp^((agent_hp+1)/(full_hp+1)) × aggresive_coeff

        Round end — draw:
          reward = +1

        All rewards are scaled by 0.001 before being returned to keep gradient
        magnitudes stable.

        Parameters
        ----------
        action : np.ndarray
            Binary action vector of length `action_dim`.

        Returns (single-player side)
        -------
        obs        : np.ndarray  – pre-processed stacked frame observation
        reward     : float       – shaped reward for the controlling player
        done       : bool        – True if the episode should reset
        truncated  : bool        – True if the episode hit a time limit
        info       : dict        – game RAM state + episode metadata
        """

        # info = {}
        # info["damage_taken"] = 0
        # info["damage_dealt"] = 0
        # info["damage_reward"] = 0.0

        # Apply optional discrete-to-binary action transformation.
        if self.action_transformer is not None:
            action = self.action_transformer(action)

        # Validate action vector shape.
        if self.side == "both":
            assert action.shape[-1] == 2 * self.action_dim, \
                f"action_shape[-1]={action.shape[-1]}, 2 * self.action_dim={2*self.action_dim}"
        else:
            assert action.shape[-1] == self.action_dim, \
                f"action_shape[-1]={action.shape[-1]}, self.action_dim={self.action_dim}"

        custom_done = False

        # Filter out the START/pause button to prevent the agent from pausing.
        action[3] = 0

        # ---------------------------------------------------------------------------
        # Build the action sequence for the underlying emulator.
        # The emulator expects a full 24-bit button array (12 per player).
        # For combos, the pre-computed SF_COMBOS sequence is used directly.
        # For regular actions, the same 12-bit array is repeated num_step_frames times.
        # ---------------------------------------------------------------------------

        if self.side == "left":
            if self.enable_combo:
                # Decode last 3 bits as binary index into SF_COMBOS.
                combo_id = int(4 * action[-3] + 2 * action[-2] + action[-1])
            else:
                combo_id = len(SF_COMBOS)  # Out-of-range → no combo.

            if combo_id >= len(SF_COMBOS):
                # Regular action: pad the right player's buttons with zeros.
                action_seq = [
                    np.hstack([action[:12], np.zeros_like(action[:12])])
                    for _ in range(self.num_step_frames)
                ]
            else:
                # Combo action: use the pre-defined button sequence.
                combo = SF_COMBOS[combo_id]
                assert self.num_step_frames == len(combo)
                action_seq = [
                    np.hstack([combo[t], np.zeros_like(combo[t])])
                    for t in range(self.num_step_frames)
                ]

        elif self.side == 'right':
            if self.enable_combo:
                combo_id = int(4 * action[-3] + 2 * action[-2] + action[-1])
            else:
                combo_id = len(SF_COMBOS)

            if combo_id >= len(SF_COMBOS):
                # Pad the left player's buttons with zeros.
                action_seq = [
                    np.hstack([np.zeros_like(action[:12]), action[:12]])
                    for _ in range(self.num_step_frames)
                ]
            else:
                combo = SF_COMBOS[combo_id]
                assert self.num_step_frames == len(combo)
                action_seq = [
                    np.hstack([np.zeros_like(combo[t]), combo[t]])
                    for t in range(self.num_step_frames)
                ]

        else:  # side == "both": interleave both players' actions.
            action[self.action_dim + 3] = 0  # Filter START for player 2 as well.
            if self.enable_combo:
                combo_ids = [
                    int(4 * action[self.action_dim - 3] + 2 * action[self.action_dim - 2] + action[self.action_dim - 1]),
                    int(4 * action[-3] + 2 * action[-2] + action[-1]),
                ]
            else:
                combo_ids = [len(SF_COMBOS), len(SF_COMBOS)]

            action_seqs = []
            for player_id, combo_id in enumerate(combo_ids):
                if combo_id >= len(SF_COMBOS):
                    action_seq = [
                        action[player_id * self.action_dim: player_id * self.action_dim + 12]
                        for _ in range(self.num_step_frames)
                    ]
                else:
                    combo = SF_COMBOS[combo_id]
                    assert self.num_step_frames == len(combo)
                    action_seq = combo
                action_seqs.append(action_seq)

            # Combine per-player sequences into a single interleaved array.
            action_seq = [
                np.hstack([a1, a2])
                for a1, a2 in zip(action_seqs[0], action_seqs[1])
            ]

        # ---------------------------------------------------------------------------
        # Step the emulator num_step_frames times, holding the action sequence.
        # update_status() is called each frame to keep status flags current.
        # ---------------------------------------------------------------------------
        self.hp_before_step = self.prev_agent_hp
        for i in range(self.num_step_frames):
            obs, _reward, _done, truncated, info = self.env.step(action_seq[i])
            self.update_status(info)
            if self.rendering:
                self.env.render()
                time.sleep(self.frame_rate)

        # Read final game state from the last emulator step.
        agent_hp       = info["health"]
        enemy_hp       = info["enemy_health"]
        agent_victories  = info["matches_won"]
        enemy_victories  = info["enemy_matches_won"]
        round_countdown  = info["round_timer"]
        timesup = (round_countdown <= 0)

        self.total_timesteps += self.num_step_frames

        # ---------------------------------------------------------------------------
        # Reward computation and done flag logic.
        # The first branch handles the non-playable transition period between
        # rounds/matches; the second branch handles active gameplay.
        # ---------------------------------------------------------------------------

        if self.during_transition and (
            self.match_status == END_STATUS or self.round_status == END_STATUS
        ):
            # --- Transition / cut-scene period ---
            # No learning signal; only determine whether to end the episode.
            custom_done = False
            custom_reward = 0
            custom_reward_inverse = 0

            if (enemy_victories == 2) or (
                (self.match_status == END_STATUS) and (enemy_victories >= agent_victories)
            ):
                # Agent lost the match → reset if reset_type != "never".
                custom_done = not self.reset_type == "never"

            if (agent_victories == 2) or (
                (self.match_status == END_STATUS) and (agent_victories > enemy_victories)
            ):
                # Agent won the match.
                custom_done = self.reset_type == "match"
                if self.level == 15:
                    # Agent beat the entire arcade ladder — game complete!
                    print(f"Player wins the game")
                    custom_done = True
                    self.save_state = True
                    self.level += 1

        else:
            # --- Active gameplay ---
            self.during_transition = False  # We are now inside a live round.

            if (agent_hp < 0 and enemy_hp < 0) or (timesup and agent_hp == enemy_hp):
                # Double KO or time-out draw.
                custom_reward = 1
                custom_reward_inverse = 1
                if self.reset_type == "round":
                    custom_done = True
                else:
                    custom_done = False
                    self.during_transition = True

            elif agent_hp < 0 or (timesup and agent_hp < enemy_hp):
                # Agent lost the round.
                # Reward decreases as remaining enemy HP increases (punishes giving
                # up lots of HP before dying).
                custom_reward = (
                    -math.pow(self.full_hp, (agent_hp + 1) / (self.full_hp + 1))
                    * self.aggresive_coeff
                )
                custom_reward_inverse = math.pow(self.full_hp, (enemy_hp + 1) / (self.full_hp + 1))

                if self.reset_type == "round":
                    custom_done = True
                else:
                    self.during_transition = True
                    if (enemy_victories >= 2) or (
                        (self.match_status == END_STATUS) and (enemy_victories > agent_victories)
                    ):
                        custom_done = not self.reset_type == "never"

            elif enemy_hp < 0 or (timesup and agent_hp > enemy_hp):
                # Agent won the round.
                # Reward scales with the agent's remaining HP (incentivises
                # winning without taking damage).
                custom_reward = (
                    math.pow(self.full_hp, (agent_hp + 1) / (self.full_hp + 1))
                    * self.aggresive_coeff
                )
                custom_reward_inverse = -math.pow(self.full_hp, (agent_hp + 1) / (self.full_hp + 1))

                if self.reset_type == "reset":
                    custom_done = True
                else:
                    self.during_transition = True
                    if (agent_victories >= 2) or (
                        (self.match_status == END_STATUS) and (agent_victories > enemy_victories)
                    ):
                        custom_done = self.reset_type == "match"
                        if self.level == 15:
                            print("Player wins the game")
                            custom_done = True
                            self.save_state = True
                            self.level += 1

            else:
                # --- Mid-round: dense reward based on HP delta ---
                # Reward = aggresive_coeff × damage dealt − damage received.
                # prev_*_hp is updated here so next step's delta is relative to now.
                damage_taken = max(0, self.prev_agent_hp - agent_hp)
                damage_dealt = max(0, self.prev_enemy_hp - enemy_hp)


                # HERE: Experiment with mid-round rewards here
                # custom_reward = self.dense_coeff * (
                #     self.aggresive_coeff * damage_dealt
                #     - damage_taken
                # )
                #
                # custom_reward_inverse = self.dense_coeff * (
                #     self.aggresive_coeff * damage_taken
                #     - damage_dealt
                # )

                # EXP: damage taken tuning
                if self.max_damage_taken < damage_taken:
                    self.max_damage_taken = damage_taken
                if self.max_damage_dealt < damage_dealt:
                    self.max_damage_dealt = damage_dealt

                # default reward structure (used by Luca)
                default_rewards = True
                custom_reward = get_rewards(self.dense_coeff, self.aggresive_coeff, damage_taken, damage_dealt,
                                            defaults=default_rewards, max_damage_taken=self.max_damage_taken)
                custom_reward_inverse = get_rewards(self.dense_coeff, self.aggresive_coeff, damage_dealt, damage_taken,
                                                    defaults=default_rewards, max_damage_taken=self.max_damage_dealt)

                self.prev_agent_hp = agent_hp
                self.prev_enemy_hp = enemy_hp
                custom_done = False

            # Annotate info with current level and state for logging / curriculum.
            info["level"] = self.level
            info["match"] = "start" if self.match_status == START_STATUS else "end"
            info["round"] = "start" if self.round_status == START_STATUS else "end"

            # Record episode outcome for win-rate tracking (used by callbacks).
            if custom_done:
                info["outcome"] = (
                    "win" if (agent_hp > enemy_hp) else ("lose" if (agent_hp < enemy_hp) else "draw")
                )

        self.last_info = info

        # Scale rewards by 0.001 to keep values in a numerically stable range.
        if self.side == "left":
            return self._get_obs(obs), 0.001 * custom_reward, custom_done, truncated, info
        elif self.side == "right":
            return self._get_obs(obs), 0.001 * custom_reward_inverse, custom_done, truncated, info
        else:
            # Two-player mode: return separate rewards for each agent.
            return (
                self._get_obs,
                0.001 * custom_reward,
                0.001 * custom_reward_inverse,
                custom_done,
                truncated,
                info,
            )

    def _opponent_attacking(self):
        # True if the agent took damage in the last step
        return self.last_info.get("health", self.hp_before_step) < self.hp_before_step

# ---------------------------------------------------------------------------
# Two-player monitor
# ---------------------------------------------------------------------------

class Monitor2P(Monitor):
    """
    Extended SB3 Monitor for two-player (side="both") environments.

    The standard Monitor only tracks a single reward stream.  Monitor2P adds
    a parallel `rewards_other` buffer and `episode_returns_other` list so the
    episode return for *player 2* is also logged alongside player 1's.

    The step signature matches SFWrapper.step() when side="both":
      (obs, reward_p1, reward_p2, done, info)
    """

    def __init__(
        self,
        env: Env,
        filename: Optional[str] = None,
        allow_early_resets: bool = True,
        reset_keywords: Tuple[str, ...] = (),
        info_keywords: Tuple[str, ...] = (),
        override_existing: bool = True,
    ):
        super().__init__(
            env, filename, allow_early_resets, reset_keywords, info_keywords, override_existing
        )
        # Secondary reward buffer and history (mirrors Monitor's self.rewards / episode_returns).
        self.rewards_other = None
        self.episode_returns_other = []

    def reset(self, **kwargs) -> GymObs:
        """Reset the environment and clear both reward buffers."""
        self.rewards_other = []
        return super().reset(**kwargs)

    def step(self, action: Union[np.ndarray, int]) -> GymStepReturn:
        """
        Step the environment and record both players' per-step rewards.

        On episode end, appends an 'ro' (reward other) key to the episode
        info dict in addition to the standard 'r' (reward) key.
        """
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")

        # Unpack the two-player reward tuple from SFWrapper.
        observation, reward, reward_other, done, info = self.env.step(action)

        self.rewards.append(reward)
        self.rewards_other.append(reward_other)

        if done:
            self.needs_reset = True
            ep_rew       = sum(self.rewards)
            ep_rew_other = sum(self.rewards_other)
            ep_len       = len(self.rewards)

            ep_info = {
                "r":  round(ep_rew, 6),
                "ro": round(ep_rew_other, 6),   # Player 2 episode return.
                "l":  ep_len,
                "t":  round(time.time() - self.t_start, 6),
            }
            for key in self.info_keywords:
                ep_info[key] = info[key]

            self.episode_returns.append(ep_rew)
            self.episode_returns_other.append(ep_rew_other)
            self.episode_lengths.append(ep_len)
            self.episode_times.append(time.time() - self.t_start)

            ep_info.update(self.current_reset_info)
            if self.results_writer:
                self.results_writer.write_row(ep_info)
            info["episode"] = ep_info

        self.total_steps += 1
        return observation, reward, reward_other, done, info
