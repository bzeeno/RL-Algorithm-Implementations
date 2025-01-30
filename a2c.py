import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.optim as optim
import numpy as np

class Actor(nn.Module):
    def __init__(self, s_size, a_size, h_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(s_size, h_size),
            nn.GELU(),
            nn.Linear(h_size, a_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.layers(x)

    def act(self, state):
        a_probs = self.forward(state)
        a_dist = Categorical(a_probs)
        action = a_dist.sample()
        return action.item(), a_dist.log_prob(action)

class Critic(nn.Module):
    def __init__(self, s_size, h_size):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(s_size, h_size),
            nn.GELU(),
            nn.Linear(h_size, 1)
        )
    
    def forward(self, state):
        return self.layers(state)

def a2c(actor, critic, actor_optimizer, critic_optimizer, n_episodes, max_t, gamma, print_freq):
    actor_losses, critic_losses, returns = [], [], []
    for i_episode in range(n_episodes):
        state, _ = env.reset()
        state = torch.from_numpy(state).float().to(device)
        ep_return = 0

        for t in range(max_t):
            # actor takes action
            action, log_prob = actor.act(state)
            # environment update
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            reward_tensor = torch.tensor([reward], device=device, dtype=torch.float32)
            next_state = torch.from_numpy(next_state).float().to(device)
            # critic estimates q-value
            value = critic(state)
            # calculate reward-to-go and advantage
            next_value = critic(next_state).detach()
            td_target = (reward_tensor + gamma * next_value if not done else reward_tensor)
            advantage = td_target - value.detach()
            # update actor params
            actor_loss = -log_prob * advantage.detach()
            actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_optimizer.step()

            # update critic params
            critic_loss = F.mse_loss(value, td_target)
            critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_optimizer.step()

            # update for next step
            state = next_state
            ep_return += reward
            if done:
                break
        
        # Gather stats for episode 
        actor_losses.append(actor_loss.item())
        critic_losses.append(critic_loss.item())
        returns.append(ep_return)
        # print if necessary
        if i_episode % print_freq == 0:
            print('Episode: {}\tAverage Actor Loss: {:.2f}\tAverage Critic Loss: {:.2f}\tReturn: {:.2f}'.format(i_episode, np.mean(actor_losses), np.mean(critic_losses), ep_return))

    return actor_losses, critic_losses, returns

def evaluate_agent(actor, env, max_t, n_eval_episodes):
    returns = []
    for i_epsiode in range(n_eval_episodes):
        state, _ = env.reset()
        ep_return = 0
        
        for t in range(max_t):
            state = torch.from_numpy(state).float().to(device)
            with torch.no_grad():
                action, _ = actor.act(state)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_return += reward
            if done:
                break

        returns.append(ep_return)

    mean_reward = np.mean(returns)
    std_reward = np.std(returns)
    return mean_reward, std_reward

if __name__ == '__main__':
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
    "actor_lr": 1e-2,
    "critic_lr": 1e-2,
    "env_id": env_id,
    "state_space": s_size,
    "action_space": a_size,
    }

    cartpole_actor = Actor(cartpole_hyperparameters["state_space"], cartpole_hyperparameters["action_space"], cartpole_hyperparameters["h_size"]).to(device)
    cartpole_critic = Critic(cartpole_hyperparameters["state_space"], cartpole_hyperparameters["h_size"]).to(device)

    cartpole_actor_optimizer = optim.AdamW(cartpole_actor.parameters(), lr=cartpole_hyperparameters["actor_lr"])
    cartpole_critic_optimizer = optim.AdamW(cartpole_critic.parameters(), lr=cartpole_hyperparameters["critic_lr"]) 

    actor_losses, critic_losses, returns = a2c(
        cartpole_actor,
        cartpole_critic,
        cartpole_actor_optimizer,
        cartpole_critic_optimizer,
        cartpole_hyperparameters["n_training_episodes"],
        cartpole_hyperparameters["max_t"],
        cartpole_hyperparameters["gamma"],
        100
    )

    mean_reward, std_reward = evaluate_agent(
        cartpole_actor, 
        eval_env, 
        cartpole_hyperparameters["max_t"], 
        cartpole_hyperparameters["n_evaluation_episodes"]
    )

    print("Mean Reward: ", mean_reward)
    print("STD Reward: ", std_reward)

    torch.save(cartpole_actor, "models/A2C/actor.pt")
    torch.save(cartpole_critic, "models/A2C/critic.pt")
