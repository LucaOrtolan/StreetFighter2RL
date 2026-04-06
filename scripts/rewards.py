import numpy as np


def sigmoid_01(x, k, midpoint):
    sig = lambda t: 1 / (1 + np.exp(-t))
    lo = sig(-k * midpoint)
    hi = sig(k * (1 - midpoint))
    return (sig(k * (x - midpoint)) - lo) / (hi - lo)


def get_midround_rewards(dense_coeff : float, aggresive_coeff: float, damage_taken: float, damage_dealt: float, rewards_scheme="default", max_damage_taken=0.0) -> np.ndarray:

    # Defaults
    if rewards_scheme == "default":
        custom_reward = dense_coeff * (aggresive_coeff * damage_dealt - damage_taken)

        # Originals written as
        # custom_reward = self.dense_coeff * (self.aggresive_coeff * damage_dealt - damage_taken)
        # custom_reward_inverse = self.dense_coeff * (self.aggresive_coeff * damage_taken - damage_dealt)

    elif rewards_scheme == "sigmoid":
        eps = 10 ** -6
        # tuned_damage_taken = sigmoid_01(max_damage_taken / (damage_taken + eps), k, m) * damage_taken
        # first 2: m=0.5, k=8
        # second 2: m=0.4, k=14
        m=0.3
        k=20.0
        # not tired now. not sensical. flip it next time. maybe flip it now.
        tuned_damage_taken = sigmoid_01(damage_taken / (max_damage_taken + eps), k, m) * damage_taken

        custom_reward = dense_coeff * (aggresive_coeff * damage_dealt - tuned_damage_taken)
    else:
        eps = 10**-6
        # EXP: 1 - Minimal Penalty for lower amounts of damage
        # tuned_damage_taken = ((max_damage_taken/(damage_taken + eps))**3) * damage_taken
        # tuned_damage_taken = ((damage_taken / (max_damage_taken + eps)) ** 3) * damage_taken

        # EXP 2: - Higher damage amounts have similar penalties
        # tuned_damage_taken = ((max_damage_taken / (damage_taken + eps)) ** 0.25) * damage_taken
        tuned_damage_taken = ((damage_taken / (max_damage_taken + eps)) ** 0.15) * damage_taken

        custom_reward = dense_coeff * (aggresive_coeff * damage_dealt - tuned_damage_taken)

    return custom_reward
