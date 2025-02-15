import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

def create_environment(env_id, seed, idx, capture_video=False):
    def _thunk():
        env = gym.make(env_id, render_mode="rgb_array")
        env_seed = seed + idx
        env.reset(seed=env_seed)
        env.action_space.seed(env_seed)
        env.observation_space.seed(env_seed)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                run_name = f"{env_id}-seed{seed}"
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        return env
    return _thunk

class Model(nn.Module):
    def __init__(self, state_space, action_space, h_size):
        super(Model, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(state_space, h_size), nn.GELU(),
            nn.Linear(h_size, h_size), nn.GELU(),
            nn.Linear(h_size, 1)
        )
        self.actor = nn.Sequential(
            nn.Linear(state_space, h_size), nn.GELU(),
            nn.Linear(h_size, h_size), nn.GELU(),
            nn.Linear(h_size, action_space)
        )
    
    def get_value_from_critic(self, obs):
        return self.critic(obs)
    
    def get_action(self, obs, action=None):
        logits = self.actor(obs)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(obs)

def ppo(model, optimizer, envs, hyperparameters, batch_size, num_updates, num_mini_batches, device):
    # Initialize variables (obs, action, reward, logprob(old), advantage, value_target)
    observations = torch.zeros((hyperparameters["max_T"], hyperparameters["num_envs"]) + envs.single_observation_space.shape, device=device)
    actions = torch.zeros((hyperparameters["max_T"], hyperparameters["num_envs"]) + envs.single_action_space.shape, device=device)
    rewards = torch.zeros((hyperparameters["max_T"], hyperparameters["num_envs"]), device=device)
    old_log_probs = torch.zeros((hyperparameters["max_T"], hyperparameters["num_envs"]), device=device)
    advantages = torch.zeros((hyperparameters["max_T"], hyperparameters["num_envs"]), device=device)
    value_targets = torch.zeros((hyperparameters["max_T"], hyperparameters["num_envs"]), device=device)

    # value estimates will be used to calculate the advantage
    value_estimates = torch.zeros((hyperparameters["max_T"], hyperparameters["num_envs"]), device=device)

    num_updates = hyperparameters["total_timesteps"] // (hyperparameters["max_T"] * hyperparameters["num_envs"])

    for i in range(num_updates):
        obs, _ = envs.reset()  # Reset environments and unpack the tuple
        obs = torch.tensor(obs, dtype=torch.float32).to(device)  # Convert to tensor and move to device
        # Collect data from each environment
        for t in range(hyperparameters["max_T"]):
            # Get action
            with torch.no_grad():
                action, old_log_prob, entropy, value = model.get_action(obs)
            # Take action
            next_obs, next_reward, done, truncated, _ = envs.step(action.cpu().numpy())
            next_obs = torch.tensor(next_obs, dtype=torch.float32).to(device)  # Convert to tensor and move to device
            next_reward = torch.tensor(next_reward, dtype=torch.float32).to(device)  # Convert rewards to tensor
            # Store data
            observations[t] = next_obs
            actions[t] = action
            rewards[t] = next_reward
            old_log_probs[t] = old_log_prob
            value_estimates[t] = value.flatten()
        
        # Calculate advantage
        for t in reversed(range(hyperparameters["max_T"])):
            returns = torch.zeros_like(rewards).to(device)
            est_next_value = model.get_value_from_critic(next_obs).flatten()
            if t == hyperparameters["max_T"] - 1:
                next_return = est_next_value
            else:
                next_return = returns[t + 1]
            returns[t] = rewards[t] + hyperparameters["gamma"] * next_return
        advantages = returns - value_estimates

        # Train the model
        batch_obs = observations.reshape(observations.shape[0] * observations.shape[1], -1)
        batch_actions = actions.reshape(actions.shape[0] * actions.shape[1], -1)
        batch_old_log_probs = old_log_probs.reshape(-1)
        batch_advantages = advantages.reshape(-1)
        batch_value_targets = returns.reshape(-1)
        train_indices = np.arange(batch_size)
        for epoch in range(hyperparameters["num_training_epochs"]):
            np.random.shuffle(train_indices)
            for start in range(0, batch_size, hyperparameters["mini_batch_size"]):
                end = start + hyperparameters["mini_batch_size"]
                mini_batch_indices = train_indices[start:end]
                mini_batch_obs = batch_obs[mini_batch_indices]
                mini_batch_actions = batch_actions.long()[mini_batch_indices]
                mini_batch_old_log_probs = batch_old_log_probs[mini_batch_indices]
                mini_batch_advantages = batch_advantages[mini_batch_indices]
                mini_batch_value_targets = batch_value_targets[mini_batch_indices]

                # Calculate loss
                _, log_prob, entropy, value = model.get_action(mini_batch_obs, mini_batch_actions)
                ratio = torch.exp(log_prob - mini_batch_old_log_probs)
                clipped_ratio = torch.clamp(ratio, 1 - hyperparameters["epsilon"], 1 + hyperparameters["epsilon"])
                L_clip = torch.min(ratio * mini_batch_advantages, clipped_ratio * mini_batch_advantages).mean()
                L_value = nn.MSELoss()(value, mini_batch_value_targets)
                L_entropy = entropy.mean()
                loss = -(L_clip + hyperparameters["s_const"] * L_entropy - hyperparameters["v_const"] * L_value)

                # Backpropagate
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print(f"Epoch: {epoch}, Loss: {loss.item()}, L_clip: {L_clip.item()}, L_value: {L_value.item()}, L_entropy: {L_entropy.item()}")
    envs.close()

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_id = "CartPole-v1"
    random_seed = 42
    num_envs = 8

    envs = gym.vector.SyncVectorEnv(
        [create_environment(env_id, random_seed, i) for i in range(num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "Only discrete action spaces are supported"

    # Get the state space and action space
    s_size = envs.single_observation_space.shape[0]
    a_size = envs.single_action_space.n
    # Set hyperparameters
    cartpole_hyperparameters = {
    "num_envs": num_envs,
    "max_T": 128,
    "h_size": 64,
    "model_lr": 1e-2,
    "env_id": env_id,
    "state_space": s_size,
    "action_space": a_size,
    "total_timesteps": 10000,
    "num_training_epochs": 10,
    "gamma": 0.99,
    "mini_batch_size": 32,
    "epsilon": 0.2,
    "v_const": 0.5,
    "s_const": 0.01
    }
    # Get number of mini-batches
    batch_size = int(cartpole_hyperparameters["max_T"] * num_envs)
    num_updates = cartpole_hyperparameters["total_timesteps"] // batch_size
    num_mini_batches = batch_size // cartpole_hyperparameters["mini_batch_size"]

    # Initialize model and optimizer
    cartpole_ppo_model = Model(cartpole_hyperparameters["state_space"], cartpole_hyperparameters["action_space"], cartpole_hyperparameters["h_size"]).to(device)
    cartpole_ppo_optimizer = optim.AdamW(cartpole_ppo_model.parameters(), lr=cartpole_hyperparameters["model_lr"])

    # Call PPO function
    ppo(cartpole_ppo_model, cartpole_ppo_optimizer, envs, cartpole_hyperparameters, batch_size, num_updates, num_mini_batches, device)
    envs.close()
    torch.save(cartpole_ppo_model, "models/cartpole_ppo_model.pth")