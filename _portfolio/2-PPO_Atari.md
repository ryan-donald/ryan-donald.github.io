---
title: "Proximal Policy Optimization, Deep Reinforcement Learning"
excerpt: "Implementation of the Proximal Policy Optimization (PPO) algorithm, tested with the Atari Pong Gymnasium environment<br/><img src='/images/ppo_pong_website.gif'>"
collection: portfolio
---
This notebook contains my implementation of the Proximal Policy Optimization algorithm, trained within the Atari ALE/Pong-v5 Gymnasium environment, and verified within the Atari PongNoFrameskip-v4 Gymnasium environment.

Information about both environments can be found [here](https://ale.farama.org/environments/pong/).

In this environment, the agent is tasked with playing the classic game of Pong against a computer opponent. The agent receives a +1 reward each time it scores a point, a -1 reward each time the computer scores a point, and a 0 reward for every other timestep. A perfect agent will score +21 every episode, and a pure random agent will score -21 every episode.

Basic import functions:


```python
import numpy as np
import gymnasium as gym

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

import matplotlib.pyplot as plt
```

The cell below enables PyTorch to run on a CUDA enabled GPU if available.


```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```

    Using device: cuda


The cell below implements the Actor network and the Critic network The Actor network inputs an 84x84x4 state representing four stacked 84x84 gray scale images from the environment, and outputs a softmax layer representing the probabilities for each action in the policy. The Critic network inputs the same 84x84x4 state representing four stacked 84x84 gray scale images from the environment, and outputs a single value based upon the state.  
  
The state is represented as four consecutive 84x84 gray scale images from the environment for a few reasons:  
1. The image is downscaled to 84x84 as determined in the paper "Playing Atari with Deep Learning" by Mnih et al, from Google DeepMind. This reduces the number of pixels improving computation time without losing too much information to determine the state.
2. In the same purpose of reducing the size of the state, the image is gray scaled as color information is not important, reducing the number of pixels to one third.
3. Four consecutive images are stacked on top of eachother to allow the network to learn based on the velocity of the ball, as well as position. With only a single image, it is impossible to determine the velocity of the ball.  


```python
class Actor(nn.Module):
    
    def __init__(self, action_dim):
        super(Actor, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
                
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        probs = F.softmax(self.fc(x), dim=-1)
        return probs

class Critic(nn.Module):

    def __init__(self):
        super(Critic, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        value = self.fc(x)
        return value
```

The PPOAgent instantiates an Actor network and a Critic network, as well as a shared optimizer, meant to introduce stability into the training step of the networks. Three additional functions are provided. First, select_action() allows for a state vector to be input, and will input that through the actor network, and sample an action based upon the policy. This function returns the action vector, the log probability of that action, and the entropy of the distribution for that action. Second, the compute_gae() function allows for the computation of the generalized advantage estimation for each timestep in the collected batch of experiences, returning a returns vector and an advantages vector. Lastly, the update() function performs the minibatch generation and update loop for the two networks based upon the PPO algorithm.


```python
class PPOAgent:

    def __init__(self, action_dim, lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_epsilon=0.2):
        self.actor = Actor(action_dim).to(device)
        self.critic = Critic().to(device)
        self.optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()), 
            lr=lr
        )
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
    
    def select_action(self, state):
        
        with torch.no_grad():
            probs = self.actor(state)
        
        dist = Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action.squeeze(), log_prob.squeeze(), entropy.squeeze()
    
    def compute_gae(self, rewards, values, dones, next_value):
        num_steps = rewards.shape[0]
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros_like(next_value)

        for step in reversed(range(num_steps)):
            if step == num_steps - 1:
                next_val = next_value
            else:
                next_val = values[step + 1]
                
            delta = rewards[step] + self.gamma * next_val * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.gae_lambda * (1 - dones[step]) * gae
            advantages[step] = gae
        
        returns = advantages + values
        return advantages, returns

    def update(self, states, actions, log_probs_old, returns, advantages, epochs=4, batch_size=64, entropy_coef=0.01):
        b_states = states.reshape(-1, 4, 84, 84)
        b_actions = actions.reshape(-1)
        b_log_probs_old = log_probs_old.reshape(-1)
        b_returns = returns.reshape(-1)
        b_advantages = advantages.reshape(-1)

        dataset_size = b_states.shape[0]

        for epoch in range(epochs):
            indices = np.random.permutation(dataset_size)

            for start in range(0, dataset_size, batch_size):
                end = start + batch_size
                batch_indices = indices[start:end]

                batch_states = b_states[batch_indices]
                batch_actions = b_actions[batch_indices]
                batch_log_probs_old = b_log_probs_old[batch_indices]
                batch_returns = b_returns[batch_indices]
                batch_advantages = b_advantages[batch_indices]
                batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-8)

                probs = self.actor(batch_states)
                dist = Categorical(probs)
                log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                ratios = torch.exp(log_probs - batch_log_probs_old)

                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages

                actor_loss = -torch.min(surr1, surr2).mean()

                values = self.critic(batch_states).squeeze()
                critic_loss = 0.5 * F.mse_loss(values, batch_returns)

                loss = actor_loss + critic_loss - entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()), 0.5)
                self.optimizer.step()
```

The make_atari_env() function below allows for the creation of vectorized Atari environments, with additional control of each individual environment in the vectorized environments when compared to the make_vec() function I used in the LunarLander environment.


```python
def make_atari_env(env_name, num_envs=8, render_mode=None):
    
    def make_env(i):
        def _init():
            env = gym.make(env_name, frameskip=1, render_mode=render_mode)
            env = gym.wrappers.RecordVideo(env, video_folder="videos", episode_trigger=lambda x: True, name_prefix="ppo_pong_"+ str(i)) if render_mode == "rgb_array" else env
            
            env = gym.wrappers.AtariPreprocessing(
                env,
                noop_max=30,
                frame_skip=4,
                screen_size=84,
                terminal_on_life_loss=True,
                grayscale_obs=True,
                grayscale_newaxis=False,
                scale_obs=True
            )
            
            env = gym.wrappers.FrameStack(env, num_stack=4)
            
            return env
        return _init
    
    envs = gym.vector.SyncVectorEnv([make_env(i) for i in range(num_envs)])
    return envs
```

The function below, run_ppo(), performs the entire training loop of PPO for a given continuous state and action space Gymnasium environment, from collecting the experiences for each batch, computing the advantages, and performing the update loop. Additionally, this will save the final and best performing actor network and critic network.


```python
def run_ppo_atari():
    num_envs = 8
    env_name = "PongNoFrameskip-v4"
    envs = make_atari_env(env_name, num_envs=num_envs)

    action_dim = envs.single_action_space.n
    agent = PPOAgent(action_dim, lr=2.5e-4, gamma=0.99, gae_lambda=0.95, clip_epsilon=0.1)

    state, info = envs.reset()
    state = torch.FloatTensor(np.array(state)).to(device)

    episode_rewards = []
    current_episode_rewards = np.zeros(num_envs)

    total_timesteps = 10_000_000
    steps_per_rollout = 128
    num_updates = total_timesteps // (steps_per_rollout * num_envs)
    batch_size = 256
    num_epochs = 4

    # to load saved models that are either partially or fully trained    
    # agent.actor.load_state_dict(torch.load("ppo_atari_training_actor.pth", map_location=device))
    # agent.critic.load_state_dict(torch.load("ppo_atari_training_critic.pth", map_location=device))

    for update in range(num_updates):

        states = torch.zeros((steps_per_rollout, num_envs, 4, 84, 84)).to(device)
        actions = torch.zeros((steps_per_rollout, num_envs), dtype=torch.long).to(device)
        log_probs = torch.zeros((steps_per_rollout, num_envs)).to(device)
        rewards = torch.zeros((steps_per_rollout, num_envs)).to(device)
        dones = torch.zeros((steps_per_rollout, num_envs)).to(device)
        values = torch.zeros((steps_per_rollout, num_envs)).to(device)

        for step in range(steps_per_rollout):
            states[step] = state
            with torch.no_grad():
                action, log_prob, entropy = agent.select_action(state)
                value = agent.critic(state).squeeze()

            next_state, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
            done = np.logical_or(terminated, truncated)

            actions[step] = action
            log_probs[step] = log_prob
            rewards[step] = torch.FloatTensor(reward).to(device)
            dones[step] = torch.FloatTensor(done).to(device)
            values[step] = value

            current_episode_rewards += reward
            state = torch.FloatTensor(np.array(next_state)).to(device)

            for env_idx in range(num_envs):
                if done[env_idx]:
                    episode_rewards.append(current_episode_rewards[env_idx])
                    current_episode_rewards[env_idx] = 0

        with torch.no_grad():
            next_value = agent.critic(state).squeeze()

        advantages, returns = agent.compute_gae(rewards, values, dones, next_value)
        
        agent.update(states, actions, log_probs, returns, advantages, 
                    epochs=num_epochs, batch_size=batch_size, entropy_coef=0.01)
        
        avg_reward = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
        if (update + 1) % 10 == 0:
            print(f"Update {update + 1}/{num_updates}, Average Reward: {avg_reward:.2f}, Episodes: {len(episode_rewards)}, Total Timesteps: {(update + 1) * steps_per_rollout}")

            if avg_reward >= 19.5 and (len(episode_rewards) >= 100):
                print(f"Solved in {update + 1} updates!")
                break

    torch.save(agent.actor.state_dict(), "ppo_atari_actor.pth")
    torch.save(agent.critic.state_dict(), "ppo_atari_critic.pth")
    envs.close()
    return episode_rewards
```

In the cell below, the function run_trained_agent() allows for us to visually see the performance of our trained policy from the above cell. Additionally, if the render_mode flag is set to "rgb_array", it will instead save an mp4 recording of each episode.


```python
def run_trained_agents():
    num_envs = 1
    env_name = "PongNoFrameskip-v4"
    envs = make_atari_env(env_name, num_envs=num_envs, render_mode="human")

    action_dim = envs.single_action_space.n
    agent = PPOAgent(action_dim, lr=2.5e-4, gamma=0.99, gae_lambda=0.95, clip_epsilon=0.1)

    state, info = envs.reset()
    state = torch.FloatTensor(np.array(state)).to(device)
    
    # To load saved models that are either partially or fully trained   
    agent.actor.load_state_dict(torch.load("ppo_atari_training_actor.pth", map_location=device))
    agent.critic.load_state_dict(torch.load("ppo_atari_training_critic.pth", map_location=device))

    
    for ep in range(10):
        current_episode_rewards = np.zeros(num_envs)
        done = False
        while not done:

            with torch.no_grad():
                action, log_prob, entropy = agent.select_action(state)
                value = agent.critic(state).squeeze()

            next_state, reward, terminated, truncated, info = envs.step(np.atleast_1d(action.cpu().numpy()))
            done = np.logical_or(terminated, truncated).any()
            current_episode_rewards += reward
            state = torch.FloatTensor(np.array(next_state)).to(device)

            # time.sleep(0.1)
        
        print(current_episode_rewards)

    envs.close()
```


```python
# run_ppo_atari()
run_trained_agents()
```
