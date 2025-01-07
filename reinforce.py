import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F

env = gym.make("CartPole-v1", render_mode="human")
