---
layout: page
title: My experience implementing and optimizing my own Proximal Policy Optimization (PPO) implementation
description: Applying my implementation of PPO to be used with Isaac Lab and mjlab, fixing various issues and optimizing my code in the process.
img: assets/img/thumb_isaac_lab.jpg
importance: 2
category: Reinforcement Learning
---

I previously had worked on a very simple PPO implementation for the ALE gymnasium environments. It was able to learn those environments, but I wanted to apply my algorithm to robots with the goal of eventually performing sim2real transfer. I also was very interested in optimizing the performance of the implementation and training, mainly because I find the engineering challenge intellectually intriguing and I was initially training on my RTX 3070, which is not very performant hardware. I also wanted to challenge myself to see if I could produce an implementation that was more performant than the libraries that ship with Isaac Lab without sacrificing clarity in the source code.

My implementation of PPO can be found [here](https://github.com/ryan-donald/ppo).

## Various bug fixes.

My implementation already worked across a number of standard Gymnasium environments including Atari Pong However, the tasks within Isaac Lab are significantly more difficult for an agent to learn and as a result a number of issues in my implementation came to light.

1. I was already using orthogonal initialization, which I had found out I drastically improved performance on the Atari environments, but I had missed that the output layer needs a different gain from the rest of the network, 0.01 for the actor and 1 for the critic.

2. I was initially using a constant learning rate, and a scheduled cosine learning rate decay during training. I found during my research that adaptive learning rate schedulers are also used, which increase or decrease the learning rate based on the KL values for a specific update. This type of scheduler applies a pressure to the agent against convergence which may seem like backwards thinking, but in practice it helps prevent the agent from converging to a non-optimal policy.

3. I had been using min/max normalization on my observations, but this requires an observation space that is bounded. These environments do not have defined bounds for every observation, such as object positions or goal positions. I switched to a running mean and variance estimator, using the Welford's algorithm implementation which other libraries also use, as it is an efficient method of calculating the running mean and std deviation of the distribution.

4. My GAE implementation was treating truncated and terminated episodes the same, so an episode that ended on truncation would have an advantage as though the next step had a reward of 0. In practice, the next step is likely to have a similar reward, so we just treat the critic value as though it is the reward it would have received.

## Improving the performance

Once it was learning reliably I decided to look into the performance of the code with a profiler, namely the Tracy profiler that Isaac Lab uses. Through this, I was able to get a grasp of what portions of the code were taking up the most execution time, and what portions of the code I could actually improve the efficiency of. Due to the fact that the physics simulation was a large portion of the execution time, and these libraries were shipped with Isaac Lab, I did not expect that I would be able to make my implementation significantly faster than the provided ones. I was wrong on this, as many of these implementations decided on giving up small performance gains that 'improved' readability, which added up.

1. Sampling actions with torch.distributions.Normal turned out to be roughly 14 times slower than writing the Gaussian log probability out by hand. This gets called once per environment step, so it dominated the per-iteration overhead.

2. I was calling .item() on the KL divergence once per minibatch, which forces a GPU to CPU synchronization and stalls the pipeline on every update. This was one of my first times really using and profiling code that is running on a GPU, so I learned a lot about limiting synchronization between the CPU and GPU. Another optimization was to enqueue the loss.backward() call on the GPU before we force a sync for the KL value from the device. The alternative was to keep both the KL and adaptive learning rate schedule entirely on the GPU. I still force one sync for the KL value to the CPU at the end for logging purposes.

3. Compiling of various functions with simple PyTorch @torch.compile function decorators. I compiled the normalization updates, adaptive learning rate updates, the action selection, the critic value calculation, the GAE computation, and the mini-batch loss computation. One thing that I did not notice immediately was the benefit of minibatch indexing inside the loss function, instead of inside the arguments when calling the loss function. That work was done outside of the compiled function. I had also been generating the indecies into the rollout data for each minibatch with numpy, so every minibatch dataset was built on the GPU then copied to the GPU. I changed this to use PyTorch and perform this work on the GPU.

4. Through the use of CUDA graphs, I can prevent unnecessary delays by enqueueing the entire pipeline of kernels together in one call from the host rather than one by one sequentially from the host. 

5. I was starting all my episodes together at the start of a training run. This can result in a few quirks with my implementation. First, I log my data once based on the rollout that just finished. When a task has no terminal states, and only truncates, this results in a jagged staircase effect in the plotted results. Another quirk is that in this case each environment is in lockstep with each other, and each rollout only updates the network with data from a specific portion of an episode. To fix this, I first tried initializing every episode at a random step within an episode. This caused a drastic slowdown in my implementation, caused by the same issue of no terminal states. This meant that every step of every rollout a subset of environments needed to reset themselves, causing a slowdown on every step. To fix this, I instead start each environment at quantized intervals equal to the rollout length. This way, for a task with no terminal states the reset cost is only added to one cycle per rollout, while still providing smooth recording and data from various portions of the episode within each rollout.

6. Initially the parameters of my model were all full precision FP32. At one point I tried enabling TF32 activations, but then my training started to fail and initially I incorrectly attributed issues that surfaced as a result of the change to the change itself. The root cause of the issue was that my act() and minibatch_loss() functions were compiled by PyTorch in different modes, which chose different kernels. As a result, identical weights would still result in a non-zero KL causing issues with the adaptive learning rate scheduler. I am still using FP32/TF32 for my networks, but I have done some experimentation using BF16 for activations. I found that the policy itself was not the best, likely due to the reduced precision, which the robot needs to use specific encoder steps. I think this is a specific implementation issue with the position controlled SO-ARM101. 

7. Switching the optimizer to a fused implementation, and moving my own timing instrumentation onto CUDA events, so that measuring the code no longer serialized it. These are small changes, but when combined with my other changes, they add up.

8. I reduced the VRAM usage by implementing a number of shared memory buffers preventing needless copying of data between different functions and CUDA graphs. My library has a larger VRAM footprint than the others as a result of the CUDA graphs, which is the price for faster throughput. I wanted to reduce this cost as much a I could.

## Benchmarking against the bundled libraries

### Throughput

I benchmarked this implementation against the four RL libraries bundled with Isaac Lab — [rsl_rl](https://github.com/leggedrobotics/rsl_rl), [rl_games](https://github.com/Denys88/rl_games), [skrl](https://github.com/Toni-SM/skrl), and [sb3](https://github.com/DLR-RM/stable-baselines3) on three tasks, cartpole, ant, and my SO-ARM101 reach task. Every run used 8,192 parallel environments on an RTX 5080, and each library's training config matched. This is a throughput speed measurement, however my library has a slightly longer startup due to pytorch compilation of some functions. In a short run like cartpole, this could affect the total time more significantly than longer runs. I think with the speedup that they provide, especially for more complex tasks that require long runs, the benefits outweigh this penalty.

**Cartpole** — 16 steps/env

| Framework                | Throughput (steps/s) | Iteration (ms) | Update (ms) | ryan_ppo speedup |
| ------------------------ | -------------------: | -------------: | ----------: | ---------------: |
| **ryan_ppo (this repo)** |        **1,816,704** |         **72** |       **5** |                — |
| skrl                     |            1,183,419 |            111 |          34 |            1.54× |
| rl_games                 |            1,173,036 |            112 |          31 |            1.55× |
| rsl_rl                   |            1,136,048 |            115 |          35 |            1.60× |
| sb3                      |              694,571 |            189 |           — |            2.62× |

**Ant** — 32 steps/env

| Framework                | Throughput (steps/s) | Iteration (ms) | Update (ms) | ryan_ppo speedup |
| ------------------------ | -------------------: | -------------: | ----------: | ---------------: |
| **ryan_ppo (this repo)** |          **994,079** |        **264** |      **27** |                — |
| skrl                     |              893,478 |            293 |          54 |            1.11× |
| rl_games                 |              885,594 |            296 |          49 |            1.12× |
| rsl_rl                   |              832,319 |            315 |          56 |            1.19× |
| sb3                      |              508,995 |            515 |           — |            1.95× |

**Reach** — 24 steps/env

| Framework                | Throughput (steps/s) | Iteration (ms) | Update (ms) | ryan_ppo speedup |
| ------------------------ | -------------------: | -------------: | ----------: | ---------------: |
| **ryan_ppo (this repo)** |        **1,947,558** |        **101** |      **16** |                — |
| skrl                     |              946,503 |            208 |         119 |            2.06× |
| rl_games                 |              880,621 |            223 |         127 |            2.21× |
| rsl_rl                   |              657,378 |            299 |         139 |            2.96× |
| sb3                      |              546,775 |            360 |           — |            3.56× |

`ryan_ppo` has the highest throughput on every task: 1.11–2.06× the next-fastest library and 2.0–3.6× sb3. For every task, the execution time for the physics is largely unchanged library to library, and the library implementation controls the interface of the agent with the physics, and the update steps of the PPO algorithm. My library has a significantly faster update portion, and some of the surrounding framework for the rollouts is also optimized better.

### Faster environments

Each of the three tasks also has a direct workflow version running on Newton (MuJoCo Warp) physics with a CUDA-graph-captured physics step, instead of going through Isaac Lab's managers on PhysX. In these tasks, both the usage of a direct workflow instead of a manager based workflow, and the usage of the Newton physics backend instead of the PhysX backend improve the throughput of the task on the environment side. Same benchmark:

**Cartpole (direct, Newton)** — 16 steps/env

| Framework                | Throughput (steps/s) | Iteration (ms) | Update (ms) | ryan_ppo speedup |
| ------------------------ | -------------------: | -------------: | ----------: | ---------------: |
| **ryan_ppo (this repo)** |        **3,508,178** |         **37** |       **4** |                — |
| skrl                     |            1,739,236 |             75 |          34 |            2.02× |
| rl_games                 |            1,654,663 |             79 |          34 |            2.12× |
| rsl_rl                   |            1,636,259 |             80 |          36 |            2.14× |
| sb3                      |              845,351 |            155 |           — |            4.15× |

**Ant (direct, Newton)** — 32 steps/env

| Framework                | Throughput (steps/s) | Iteration (ms) | Update (ms) | ryan_ppo speedup |
| ------------------------ | -------------------: | -------------: | ----------: | ---------------: |
| **ryan_ppo (this repo)** |        **2,097,027** |        **125** |      **27** |                — |
| skrl                     |            1,522,399 |            172 |          54 |            1.38× |
| rl_games                 |            1,494,115 |            175 |          50 |            1.40× |
| rsl_rl                   |            1,325,754 |            198 |          57 |            1.58× |
| sb3                      |              648,647 |            404 |           — |            3.23× |

**Reach (direct, Newton)** — 24 steps/env

| Framework                | Throughput (steps/s) | Iteration (ms) | Update (ms) | ryan_ppo speedup |
| ------------------------ | -------------------: | -------------: | ----------: | ---------------: |
| **ryan_ppo (this repo)** |        **2,880,358** |         **68** |      **16** |                — |
| skrl                     |            1,106,186 |            178 |         118 |            2.60× |
| rl_games                 |            1,062,657 |            185 |         126 |            2.71× |
| sb3                      |              620,027 |            317 |           — |            4.65× |
| rsl_rl                   |              503,670 |            390 |         145 |            5.72× |

### Learning

The above sections focus on how the training throughput of my library compares to the other libraries, measured with short runs that don't learn a full policy. This is only relevant if my library also learns the tasks as well as them. To measure this, I also ran a number of runs to verify the learned behavior is on par or better than the other libraries. I did this with three tasks, Cartpole, Ant, and Franka Reach. Each library performs a training run on the tasks, one for each of three set seeds (42, 43, 44), using the config that is provided with Isaac Lab by that library for each task. These were all run on an RTX 5080, with 8192 parallel envs. Some adjustments had to be made for rl_games and sb3, to scale from the default 4096 envs to 8192 envs. The rewards are the median across the three seeds, and the "final" rewards value is the mean of the last 10% of the training run. Range is 25th-75th percentile.

**Cartpole** — 131M env steps (1,000 iterations)

| Framework                | Final reward |  Seed range | Throughput (steps/s) | Wall-clock (min) |
| ------------------------ | -----------: | ----------: | -------------------: | ---------------: |
| **ryan_ppo (this repo)** |    **4.954** | 4.951–4.954 |        **1,820,493** |          **1.5** |
| rsl_rl                   |        4.953 | 4.940–4.954 |            1,172,452 |              2.1 |
| sb3                      |        4.953 | 4.951–4.956 |              193,833 |             11.5 |
| skrl                     |        4.922 | 4.888–4.927 |              837,575 |              2.9 |
| rl_games                 |        4.797 | 4.782–4.801 |              784,086 |              3.0 |

<div align="center">
  <img src="{{ site.baseurl }}/assets/img/benchmark_cartpole_learning.png" width="100%" alt="Cartpole reward vs env steps">
</div>

**Ant** — 524M env steps (2,000 iterations at 32 steps/env; 4,000 for the libraries that ship 16 steps/env)

| Framework                | Final reward | Seed range | Throughput (steps/s) | Wall-clock (min) |
| ------------------------ | -----------: | ---------: | -------------------: | ---------------: |
| **ryan_ppo (this repo)** |     **99.0** |  97.1–99.2 |          **990,794** |          **9.3** |
| rsl_rl                   |         87.6 |  87.5–88.0 |              667,776 |             13.5 |
| skrl                     |         68.9 |  67.3–74.0 |              877,204 |             10.4 |
| rl_games                 |         64.7 |  62.5–68.9 |              861,793 |             10.6 |
| sb3                      |         59.3 |  58.0–61.1 |              561,684 |             16.0 |

<div align="center">
  <img src="{{ site.baseurl }}/assets/img/benchmark_ant_learning.png" width="100%" alt="Ant reward vs env steps">
</div>

**Franka Reach** — 197M env steps (1,000 iterations)

| Framework                | Final reward |  Seed range | Throughput (steps/s) | Wall-clock (min) |
| ------------------------ | -----------: | ----------: | -------------------: | ---------------: |
| **ryan_ppo (this repo)** |        0.240 | 0.239–0.240 |          **503,474** |          **7.2** |
| rl_games                 |    **0.243** | 0.243–0.243 |              452,301 |              7.9 |
| skrl                     |        0.242 | 0.242–0.243 |              449,226 |              7.9 |
| rsl_rl                   |        0.237 | 0.236–0.237 |              441,120 |              8.1 |

<div align="center">
  <img src="{{ site.baseurl }}/assets/img/benchmark_reach_learning.png" width="100%" alt="Franka Reach reward vs env steps">
</div>

In the Ant task, my library clearly outperforms the others in this small benchmark, in both learning and speed. For the other two tasks, all the libraries learn similar policies, however mine finishes training first. For these tasks, my library appears to have some sample inefficiency early on compared to the others, so it doesn't necessarily reach every reward threshold before the others. The takeaway I had from this is that my library is able to accurately learn tasks comparatively to the shipped libraries, while having a significantly higher training throughput due to the effort spent on optimizing portions of the loop.

## What else I have trained with it

The plots below show the training progress across the range of Isaac Lab environments (and one mjlab environment, the MicroDuck swizzle at the bottom) I have used this implementation on:

<p float="left">
  <img src='{{ site.baseurl }}/assets/img/training_plot_episodes_Isaac-Reach-SO-ARM101-v0.png' width='49%'> <video src='{{ site.baseurl }}/assets/img/isaac_reach_so101.mp4' width="49%" style="vertical-align: middle" autoplay loop muted playsinline></video>
</p>

<p float="left">
  <img src='{{ site.baseurl }}/assets/img/training_plot_episodes_Isaac-Fourbar-Pole-Swingup-v0.png' width='49%'> <video src='{{ site.baseurl }}/assets/img/isaac_fourbar.mp4' width="49%" style="vertical-align: middle" autoplay loop muted playsinline></video>
</p>

<p float="left">
  <img src='{{ site.baseurl }}/assets/img/training_plot_episodes_Isaac-Open-Drawer-Franka-v0.png' width='49%'> <video src='{{ site.baseurl }}/assets/img/isaac_drawer.mp4' width="49%" style="vertical-align: middle" autoplay loop muted playsinline></video>
</p>

<p float="left">
  <img src='{{ site.baseurl }}/assets/img/training_plot_episodes_Isaac-Velocity-Rough-UnitreeGo2-v0.png' width='49%'> <video src='{{ site.baseurl }}/assets/img/isaac_go2_rough.mp4' width="49%" style="vertical-align: middle" autoplay loop muted playsinline></video>
</p>

<p float="left">
  <img src='{{ site.baseurl }}/assets/img/training_plot_episodes_Isaac-Shadow-Handover-v0.png' width='49%'> <video src='{{ site.baseurl }}/assets/img/isaac_shadow_handover.mp4' width="49%" style="vertical-align: middle" autoplay loop muted playsinline></video>
</p>

<p float="left">
  <img src='{{ site.baseurl }}/assets/img/training_plot_episodes_Isaac-Lift-KukaAllegro.png' width='49%'> <video src='{{ site.baseurl }}/assets/img/isaac_lift_kuka_allegro.mp4' width="49%" style="vertical-align: middle" autoplay loop muted playsinline></video>
</p>

<p float="left">
  <img src='{{ site.baseurl }}/assets/img/training_plot_episodes_Ryan-Juggle-Franka-Realistic.png' width='49%'> <video src='{{ site.baseurl }}/assets/img/isaac_juggle_franka.mp4' width="49%" style="vertical-align: middle" autoplay loop muted playsinline></video>
</p>

<p float="left">
  <img src='{{ site.baseurl }}/assets/img/training_plot_episodes_Ryan-BallPlate-B601-DM.png' width='49%'> <video src='{{ site.baseurl }}/assets/img/isaac_ball_plate_b601.mp4' width="49%" style="vertical-align: middle" autoplay loop muted playsinline></video>
</p>

<p float="left">
  <img src='{{ site.baseurl }}/assets/img/training_plot_episodes_Mjlab-Velocity-Swizzle-MicroDuck.png' width='49%'> <video src='{{ site.baseurl }}/assets/img/mjlab_microduck_swizzle.mp4' width="49%" style="vertical-align: middle" autoplay loop muted playsinline></video>
</p>

The reach task above is the one I have done sim-to-real transfer with a real SO-ARM101, which turned out to be a much harder problem than training it was. That is [its own write-up]({{ site.baseurl }}/projects/01_ppo_sim2real/).
