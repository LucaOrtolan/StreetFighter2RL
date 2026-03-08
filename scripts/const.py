"""
const.py — Global constants for the Street Fighter II RL project.

Defines game metadata, action spaces (buttons, directional inputs, special-move
combo sequences), and a human-readable character mapping used across training,
evaluation, and environment wrappers.
"""

import stable_retro as retro
import os
import numpy as np


# ---------------------------------------------------------------------------
# Game / ROM metadata
# ---------------------------------------------------------------------------

# Round numbers (within a multi-round match) that correspond to bonus/break stages.
# These levels have different rules (single round, no HP-based reset).
SF_BONUS_LEVEL = [4, 8, 12]

# Default save-state to load when no explicit state path is provided.
SF_DEFAULT_STATE = "Champion.Level1.RyuVsGuile"

# Absolute path to stable-retro's ROM data directory.
retro_directory = os.path.dirname(retro.__file__)

# Relative path inside the retro package where SF2 game data lives.
sf_game_dir = "data/stable/StreetFighterIISpecialChampionEdition-Genesis"

# Full path to the SF2 game-state directory (used to list available .state files).
SF_STATE_DIR = os.path.join(retro_directory, sf_game_dir)

# Retro environment ID for Street Fighter II Special Champion Edition on Genesis.
sf_game = "StreetFighterIISpecialChampionEdition-Genesis-v0"


# ---------------------------------------------------------------------------
# Match / round status flags
# These are used in SFWrapper.update_status() to track whether a match or
# round is currently in progress (START) or has ended (END).
# ---------------------------------------------------------------------------

START_STATUS = 0  # A match/round is currently active.
END_STATUS = 1    # A match/round has finished (transition state).


# ---------------------------------------------------------------------------
# Controller button layout
# This is the canonical button order expected by stable-retro for the
# Sega Genesis 6-button controller.  All action arrays must align with this.
# ---------------------------------------------------------------------------

BUTTONS = ['B', 'A', 'MODE', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'C', 'Y', 'X', 'Z']


# ---------------------------------------------------------------------------
# Special-move combo definitions
# Each combo is a sequence of button-press sub-sequences (one per input frame).
# Combos are stored as raw button name lists, then converted to binary arrays
# below so they can be fed directly to the emulator.
# ---------------------------------------------------------------------------

# Human-readable combo definitions.
# Each entry is a list of "sub-steps"; each sub-step is a list of button names
# that must be held simultaneously during that frame window.
SF_COMBOS_BUTTONS = [
    [['DOWN'], ['DOWN', 'RIGHT'], ['RIGHT'], ['X']],  # Hadouken facing right
    [['RIGHT'], ['DOWN'], ['DOWN', 'RIGHT'], ['X']],  # Shoryuken facing right
    [['DOWN'], ['DOWN', 'LEFT'], ['LEFT'], ['A']],    # Tatsumaki facing right
    [['DOWN'], ['DOWN', 'LEFT'], ['LEFT'], ['X']],    # Hadouken facing left
    [['LEFT'], ['DOWN'], ['DOWN', 'LEFT'], ['X']],    # Shoryuken facing left
    [['DOWN'], ['DOWN', 'RIGHT'], ['RIGHT'], ['A']],  # Tatsumaki facing left
]

# Convert each combo into a list of binary numpy arrays (one array per frame).
# Each array has length == len(BUTTONS), with 1 where a button is pressed.
# Each sub-step is repeated for 2 frames to match num_step_frames=8
# (4 sub-steps × 2 frames = 8 total frames per combo).
SF_COMBOS = []
for combo_buttons in SF_COMBOS_BUTTONS:
    action_seq = []
    for combo_button in combo_buttons:
        # Build a binary button vector: 1 if button in combo_button else 0.
        button = [int(b in combo_button) for b in BUTTONS]
        for _ in range(2):  # Hold each sub-step for 2 consecutive frames.
            action_seq.append(np.array(button))
    SF_COMBOS.append(action_seq)


# ---------------------------------------------------------------------------
# Discrete action space components
# Used when transform_action=True to map a single integer to a button combo.
# ---------------------------------------------------------------------------

# All possible directional inputs, including diagonals and neutral.
DIRECTIONS_BUTTONS = [
    [],                              # No direction (neutral)
    ['UP'], ['DOWN'],
    ['LEFT'], ['RIGHT'],
    ['UP', 'LEFT'], ['UP', 'RIGHT'],
    ['DOWN', 'LEFT'], ['DOWN', 'RIGHT'],
]

# All possible attack button presses (one at a time).
ATTACKS_BUTTONS = [
    [],           # No attack (neutral)
    ['B'], ['A'], ['C'],  # Light / medium / strong punch (Genesis mapping)
    ['Y'], ['X'], ['Z'],  # Light / medium / strong kick
]


# ---------------------------------------------------------------------------
# Character-select screen navigation
# Maps character names to the sequence of D-pad inputs needed to highlight
# them on the character-select grid starting from Ryu (top-left).
# ---------------------------------------------------------------------------

# Named movements → button arrays for the select screen.
SELECT_CHARACTER_MOVEMENTS = {
    'NO_OP': [],
    'START': ['START'],
    'LEFT': ['LEFT'],
    'RIGHT': ['RIGHT'],
    'UP': ['UP'],
    'DOWN': ['DOWN'],
}

# Pre-convert each movement to a binary button array.
SELECT_CHARACTER_BUTTONS = {}
for k, v in SELECT_CHARACTER_MOVEMENTS.items():
    SELECT_CHARACTER_BUTTONS[k] = np.array([int(b in v) for b in BUTTONS])

# Navigation sequences from Ryu's position to each character.
# Ryu is at grid position (row=0, col=0); grid has 2 rows × 6 cols.
SELECT_CHARACTER_SEQUENCES = {
    'Ryu':     [],
    'Honda':   ['RIGHT'],
    'Blanka':  ['RIGHT'] * 2,
    'Guile':   ['RIGHT'] * 3,
    'Balrog':  ['RIGHT'] * 4,
    'Vega':    ['RIGHT'] * 5,
    'Ken':     ['DOWN'],
    'Chunli':  ['DOWN'] + ['RIGHT'],
    'Zangief': ['DOWN'] + ['RIGHT'] * 2,
    'Dhalsim': ['DOWN'] + ['RIGHT'] * 3,
    'Sagat':   ['DOWN'] + ['RIGHT'] * 4,
    'Bison':   ['DOWN'] + ['RIGHT'] * 5,
}


# ---------------------------------------------------------------------------
# Character index → name mapping
# stable-retro returns integer IDs for the current enemy character.
# This dict lets us convert those IDs to human-readable strings for logging.
# ---------------------------------------------------------------------------

CHARACTER_MAPPING = {
    0:  "ryu",
    1:  "honda",
    2:  "blanka",
    3:  "guile",
    4:  "ken",
    5:  "chunli",
    6:  "zangief",
    7:  "dhalsim",
    8:  "bison",
    9:  "sagat",
    10: "balrog",
    11: "vega",
}