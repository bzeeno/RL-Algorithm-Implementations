import torch
import torch.nn as nn
from torch.optim import Adam

import gymnasium as gym

from collections import deque
import numpy as np

def main():
    env_name = "CartPole-v1"
    num_updates = 250
    num_collect_steps = 2048  # Steps per collection batch
    lr = 3e-4
    clip_epsilon = 0.2
    entropy_coeff = 0.01
    epochs = 4

    # Environment setup
    env = gym.make(env_name)
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n  # Discrete actions

    # Initialize networks and optimizers
    policy_model = Policy(input_dim, output_dim)
    value_model = Value(input_dim)
    policy_optim = Adam(policy_model.parameters(), lr=lr)
    value_optim = Adam(value_model.parameters(), lr=lr)

    for update in range(num_updates):
        # Collect experiences
        experiences = collect_experiences(
            env, policy_model, value_model, num_collect_steps
        )
        # Train networks
        train(
            experiences, policy_model, value_model,
            policy_optim, value_optim,
            epsilon=clip_epsilon,
            entropy_coeff=entropy_coeff,
            num_epochs=epochs
        )
        # Periodic evaluation
        if update % 50 == 0:
            avg_reward = test_policy(policy_model, env)
            print(f"Update {update}, Avg Reward: {avg_reward:.2f}")

    avg_reward = test_policy(policy_model, env)
    print(f"Update {update}, Avg Reward: {avg_reward:.2f}")

    torch.save(policy_model, "models/PPO/policy.pt")
    torch.save(value_model, "models/PPO/value.pt")

def test_policy(policy, env, num_episodes=10):
    total_reward = 0
    for _ in range(num_episodes):
        state, _ = env.reset()
        done = False
        while not done:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state)
                action_dist = get_action_probs(policy, state_tensor)
                action = action_dist.sample()
            state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            total_reward += reward
    return total_reward / num_episodes

class Policy(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, output_dim)
        )

    def forward(self, obs):
        return self.layers(obs)

class Value(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, obs):
        return self.layers(obs)

def get_action_probs(policy_model, obs):
    obs = torch.tensor(obs, dtype=torch.float32)
    logits = policy_model(obs)
    return torch.distributions.Categorical(logits=logits)

def collect_experiences(env, policy_model, value_model, max_steps):
    states = []
    log_probs = []
    actions = []
    values = []
    rewards = []
    dones = []

    state, _ = env.reset()

    for _ in range(max_steps):
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32)
            # Get action distribution and value estimate
            action_dist = get_action_probs(policy_model, state_tensor)
            value = value_model(state_tensor)
            # Sample action
            action = action_dist.sample()
            # Get log_prob
            log_prob = action_dist.log_prob(action)
        # Take action
        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        # Store experience
        states.append(state)
        log_probs.append(log_prob)
        actions.append(action)
        values.append(value)
        rewards.append(reward)
        dones.append(done)
        # Get next state
        state = next_state if not done else env.reset()[0]
    return {
        'states': torch.tensor(np.array(states), dtype=torch.float32),
        'log_probs': torch.stack(log_probs),
        'actions': torch.tensor(actions, dtype=torch.long),
        'values': torch.stack(values).flatten(),
        'rewards': torch.tensor(rewards, dtype=torch.float32),
        'dones': torch.tensor(dones, dtype=torch.float32)
    }

def train(experiences, policy_model, value_model, policy_optimizer, value_optimizer, gamma=0.99, epsilon=0.2, entropy_coeff=0.01, num_epochs=4):
    # Get tensors from experiences
    states = experiences['states']
    actions = experiences['actions']
    old_log_probs = experiences['log_probs'].detach()
    rewards = experiences['rewards']
    dones = experiences['dones']
    old_values = experiences['values'].detach()
    
    # Calculate advantages and returns
    returns = deque()
    advantages = deque()
    R = 0 # return
    for reward, value, done in zip(reversed(rewards), reversed(old_values), reversed(dones)):
        R = reward + ( ( gamma * R ) * ( 1 - done ) )
        returns.appendleft(R)
        advantages.appendleft(R - value.item())
    returns = torch.tensor(returns, dtype=torch.float32)
    advantages = torch.tensor(advantages, dtype=torch.float32)
    
    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Training Epochs
    for _ in range(num_epochs):
        # Calculate Losses
        # policy ratio
        action_dists = get_action_probs(policy_model, states)
        new_log_probs = action_dists.log_prob(actions)
        entropy = action_dists.entropy().mean()
        ratios = (new_log_probs - old_log_probs).exp()
        # Policy loss
        p_loss1 = -ratios * advantages
        p_loss2 = -torch.clamp(ratios, 1-epsilon, 1+epsilon) * advantages
        policy_loss = torch.max(p_loss1, p_loss2).mean() - (entropy_coeff * entropy)
        # Value loss
        values = value_model(states).flatten()
        value_loss = ((returns - values)**2).mean()

        # Update policy
        policy_optimizer.zero_grad()
        policy_loss.backward()
        # Update value
        value_optimizer.zero_grad()
        value_loss.backward()
        # Step
        policy_optimizer.step()
        value_optimizer.step()

if __name__ == "__main__":
    main()