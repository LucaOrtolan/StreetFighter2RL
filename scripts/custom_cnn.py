from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import gymnasium as gym
import torch

# Attention is all you need

class SFCnn(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[-1]  # 3 for your stacked frames
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Compute flattened output size dynamically
        with torch.no_grad():
            sample = torch.zeros(1, *observation_space.shape).permute(0, 3, 1, 2)
            n_flatten = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim),
            nn.ReLU()
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # SB3 passes (N, H, W, C) — permute to (N, C, H, W) for PyTorch Conv2d
        return self.linear(self.cnn(obs.permute(0, 3, 1, 2).float()))