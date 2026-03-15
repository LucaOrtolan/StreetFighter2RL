import numpy as np

def get_rewards(dense_coeff : float, aggresive_coeff: float, damage_taken: float, damage_dealt: float, defaults=True, max_damage_taken=0.0) -> np.ndarray:

    # Defaults
    if defaults:
        custom_reward = dense_coeff * (aggresive_coeff * damage_dealt - damage_taken)

    else:
        eps = 10**-6
        # EXP: 1 - Minimal Penalty for lower amounts of damage
        tuned_damage_taken = ((max_damage_taken/(damage_taken + eps))**3) * damage_taken
        custom_reward = dense_coeff * (aggresive_coeff * damage_dealt - tuned_damage_taken)

    return custom_reward
