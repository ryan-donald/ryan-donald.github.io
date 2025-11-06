---
title: "Proximal Policy Optimization within IsaacLab"
excerpt: "Implementation of the Proximal Policy Optimization (PPO) algorithm, tested with multiple environments within IsaacLab<br><img src='/images/isaac_drawer.gif' width='50%'>"
collection: portfolio
---
Previously, I have implemented the PPO algorithm to work within multiple gymnasium environments, including Atari Pong. I adjusted my PPO implementation to work with multiple environments within the IsaacLab, a gym-like framework within NVIDIA IsaacSim. A number of adjustments had to be made to improve the performance of the algorithm, as these environments provide a more difficult problem, and highlighted some issues my previous implementation had.  
  
My implementation of PPO that I am using can be found [here](https://github.com/ryan-donald/PPO_IsaacLab).
  
These issues I found in my implementation and subsequently fixed are as follows:  
- I found that the initialization of the weights in both my Actor and Critic networks was insufficient. While solving the Atari environment, I noticed this as well, and found that I should be utilizing Orthogonal weight initialization, and this worked for that environment. However, I still had issues in this environment, and I found that I had missed that the weights of the output layer should have a different gain, *0.01* for the Actor, and *1* for the Critic.  
- I also was not utilizing KL divergence for an adaptive learning rate. I found that this improved the performance of my PPO algorithm and allowed it to develop a better solution for each environment.  
- Lastly, my implementation was struggling when compared to some of the implementations provided with IsaacLab, and I found that Observation Normalization could improve the performance. Previously, I have utilized min/max normalization, however these environments do not have a defined min/max for their observations. I am utilizing the RSL-RL implementation of Welfords Algorithm to do this.

The plots below show the training progress for each environment:  
<p float="left">
    <img src='/images/training_plot_episodes_Isaac-Cartpole-v0.svg' onerror this.src='/images/training_plot_episodes_Isaac-Cartpole-v0.png' width='49%'> <img src='/images/isaac_cartpole.gif' width="49%">  
</p>
<p float="left">
    <img src='/images/training_plot_episodes_Isaac-Reach-Franka-v0.svg' onerror this.src='/images/training_plot_episodes_Isaac-Reach-Franka-v0.png' width='49%'> <img src='/images/isaac_reach.gif' width="49%"> 
</p>
<p float="left">
    <img src='/images/training_plot_episodes_Isaac-Lift-Cube-Franka-v0.svg' onerror this.src='/images/training_plot_episodes_Isaac-Lift-Cube-Franka-v0.png' width='49%'> <img src='/images/isaac_lift.gif' width="49%">  
</p>
<p float="left">
    <img src='/images/training_plot_episodes_Isaac-Open-Drawer-Franka-v0.svg' onerror this.src='/images/training_plot_episodes_Isaac-Open-Drawer-Franka-v0.png' width='49%'> <img src='/images/isaac_drawer.gif' width="49%">  
</p>
 <p float="left">
    <img src='/images/training_plot_episodes_Isaac-Repose-Cube-Allegro-Direct-v0.svg' onerror this.src='/images/training_plot_episodes_Isaac-Repose-Cube-Allegro-Direct-v0.png' width='49%'> <img src='/images/isaac_repose_allegro.gif' width="49%">  
</p>