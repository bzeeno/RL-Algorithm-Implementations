import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.optim as optim
from collections import deque
import numpy as np

class Policy(nn.Module):
    def __init__(self, s_size, a_size, h_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(s_size, h_size),
            nn.GELU(),
            nn.Linear(h_size, a_size),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.layers(x)
    
    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state)
        a_dist = Categorical(probs)
        action = a_dist.sample()
        return action.item(), a_dist.log_prob(action)

def reinforce(env, policy, optimizer, n_episodes, max_t, gamma, print_freq):
    scores_deque = deque(maxlen=100)
    scores = []
    # Generate episodes
    for i_episode in range(n_episodes):
        state, _ = env.reset()
        log_probs = []
        rewards = []
        
        # Generate an episode
        for t in range(max_t):
            action, log_prob = policy.act(state)
            log_probs.append(log_prob)
            state, reward, done, truncated, _ = env.step(action)
            rewards.append(reward)
            if done or truncated:
                break

        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        # calculate reward-to-go for each timestep
        returns = deque(maxlen=max_t)
        n_steps = len(rewards)
        for t in reversed(range(n_steps)):
            G_t1 = (returns[0] if len(returns) > 0 else 0) # G_t+1: The cummulative discounted return at the next time step
            returns.appendleft(rewards[t] + (gamma * G_t1))
        # Normalize returns
        eps = np.finfo(np.float32).eps.item()
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + eps)

        # Calculate loss
        loss = []
        for log_prob_t, reward_t in zip(log_probs, returns):
            loss.append(-log_prob_t * reward_t)
        loss = torch.cat(loss).sum()

        # Gradient descent
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print if needed
        if i_episode % print_freq == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))

    return scores
        
def evaluate_agent(env, max_steps, n_eval_episodes, policy):
  """
  Evaluate the agent for ``n_eval_episodes`` episodes and returns average reward and std of reward.
  :param env: The evaluation environment
  :param n_eval_episodes: Number of episode to evaluate the agent
  :param policy: The Reinforce agent
  """
  episode_rewards = []
  for episode in range(n_eval_episodes):
    state, _ = env.reset()
    step = 0
    done = False
    total_rewards_ep = 0
    
    for step in range(max_steps):
      action, _ = policy.act(state)
      new_state, reward, done, truncated, info = env.step(action)
      total_rewards_ep += reward
        
      if done or truncated:
        break
      state = new_state
    episode_rewards.append(total_rewards_ep)
  mean_reward = np.mean(episode_rewards)
  std_reward = np.std(episode_rewards)

  return mean_reward, std_reward

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env_id = "CartPole-v1"
    env = gym.make(env_id, render_mode="human")
    eval_env = gym.make(env_id)

    # Get the state space and action space
    s_size = env.observation_space.shape[0]
    a_size = env.action_space.n


    cartpole_hyperparameters = {
    "h_size": 16,
    "n_training_episodes": 10,
    "n_evaluation_episodes": 10,
    "max_t": 1000,
    "gamma": 1.0,
    "lr": 1e-2,
    "env_id": env_id,
    "state_space": s_size,
    "action_space": a_size,
    }

    cartpole_policy = Policy(cartpole_hyperparameters["state_space"], cartpole_hyperparameters["action_space"], cartpole_hyperparameters["h_size"]).to(device)
    cartpole_optimizer = optim.Adam(cartpole_policy.parameters(), lr=cartpole_hyperparameters["lr"])

    scores = reinforce(env, cartpole_policy,
                   cartpole_optimizer,
                   cartpole_hyperparameters["n_training_episodes"], 
                   cartpole_hyperparameters["max_t"],
                   cartpole_hyperparameters["gamma"], 
                   100)
    
    print("scores: ", scores)
    
    mean_reward, std_reward = evaluate_agent(eval_env, 
               cartpole_hyperparameters["max_t"], 
               cartpole_hyperparameters["n_evaluation_episodes"],
               cartpole_policy)
    
    print("mean_reward: ", mean_reward)
    print("std_reward: ",  std_reward)

    torch.save(cartpole_policy, "models/reinforce_model.pt")

    
