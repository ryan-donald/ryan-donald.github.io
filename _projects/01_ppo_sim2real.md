---
layout: page
title: Sim-To-Real Transfer of a Learned Policy From Isaac Lab to a Real SO-ARM101
description: System identification for stronger sim-to-real transfer on an open-source SO-ARM101.
img: assets/img/thumb_so101_reach.jpg
importance: 1
category: Reinforcement Learning
---

I trained a policy in Isaac Lab with [my own PPO implementation]({{ site.baseurl }}/projects/02_ppo_isaaclab/), deployed it to a real SO-ARM101, and it performed poorly compared to in the Isaac Lab simulator. I went through a variety of stages with the mis-match of the performance in the sim and the performance in the real world. First, it was dramatically bad, it would overshoot the goal, oscillate around the goal, and occasionally impact the table. I had spent some time working to match the dynamics of the real robot to the sim, but it was clearly not enough.

I realized here that I would need to spend time collecting data on the real robot, and identifying parameter values for the simulator. Not only this, but also employing domain randomization for these parameters in the simulator, as I was experimentally determining these values and did not want the policy to overfit to the specific values. I finally got the policy into a good, but not great, state. An easy explanation would have been that the servos were cheap and a combination of gear backlash and other shortcomings was the reason it was not perfect. That explanation would have been wrong, and the process I went through to figure out why that is wrong is what I will detail here. The arm was fine, my simulation of it was not, and a policy trained against a simulator that does not match the real world learns a control strategy that only makes sense in the simulator.

The policy is trained to move the end-effector from any joint configuration to a specified position in the robot's coordinate frame. On the real robot, it simply iterates through a randomly sampled list of targets at specific intervals.

[![PPO SO-ARM101 sim2real](https://img.youtube.com/vi/MzxyW7mrM0s/maxresdefault.jpg)](https://www.youtube.com/watch?v=MzxyW7mrM0s)

My PPO implementation is [here](https://github.com/ryan-donald/ppo), and the deployment scripts for the real robot are [here](https://github.com/ryan-donald/so101_ppo).

## The method

The systematic process I followed to fix the gap.

1. **Determine a parameter to measure** After determining what parameter I wanted to measure, I would define a test to collect the necessary metrics.
2. **Fit the simulated system** Once the metrics were gathered, I analyzed them and attempted to fit the simulated model to the predicted value.
3. **Verify in simulation** I would first perform a training run in the sim, to verify that the policy still learns to solve the task.
4. **Verify on hardware** Next, I would verify that the performance of the new policy on the real robot is better than the performance from the old policy.
5. **Keep what transferred** If the resulting transfer improved, I would keep what I adjusted, if not, I would review my beliefs of why I wanted to target that parameter, and how the results differed from what I expected.

## The various mis-matches I had

**Action Space Representation** I command the real robot through LeRobot, uses a normalized joint space, [-100, 100], based on initial robot calibration. My Isaac Lab environment commanded joint angles in radians from the URDF. To fix this, I normalized the observations and actions within the simulator to match this [-100, 100] joint space that the real robot uses.

**Servo PD terms** The robot's joints are position controlled with each servo utilizing an internal PD loop, which maps onto a joint's stiffness and damping in the Isaac Lab simulation. My original values for these were far too stiff, and they trained perfectly well in simulation, which is the problem. This resulted in a policy which learned to control initially as a somewhat bang-bang controller for each joint. Recording step responses and fitting against them put the effective proportional gain closer to 16, with a fit of roughly kp 17.8, kd 1.5, and a friction term of 0.12.

Originally, I gather a quick step response for the motors, and tried to visually match the curve from the same motion in the sim to this. This ultimately resulted in myself choosing terms which did not make physical sense on the robot, as I was trying to quickly settle this. The curves looked similar, but I only had a single step response and I did not put much thought into this besides "make the curves match". This ended up biting me when I started the sim to real transfer signficantly. Even after I went back and re-did this with more reasonable gains, I still had a slight issue, in that the gains did not accurately match how the arm performed when fighting against gravity and moving with it.

**Joint Velocity Limits** I initially used a joint velocity limit I found online for these motors. This ended up resulting in a very fast and twitchy policy, as it was trying to utilize the maximum speed anywhere it could, to maximize the reward. To fix this, I decided to add a hard velocity limit within each motor, utilizing an internal velocity goal register. This results in much smoother and consistent motion, while also adding a level of safety in case the policy has some unintended behavior.

**Action delay** Initially I trained the robot arm in the simulator without any action delay. Looking back, this was clearly an optimistic approach, and I was betting on the policy being able to handle it. As I went through the process of measuring these other parameters, I decided that if I wanted it to truly be as accurate as I could make it for the best transfer, I should measure the delay from when the policy takes an action, to when it shows up in the observation. I measured this delay at about 38 ms, a little over two control steps at 60 Hz. It is an easy measurement to get wrong, so I confirmed it three separate ways before changing anything, and it came out to roughly half of the delay I had been training with by that point.

**Inertia and acceleration limits** After the above changes, I still noticed that the arm had an odd jitter when it reached the goal, for about half of the episodes. I determined that a partial cause for this were mismatched inertia and acceleration limits for each actuator, which caused a mismatch between the physical behavior in the sim and in the real world.

**Encoder quantization** The joint positions the policy reads arrive in discrete encoder steps rather than as continuous values, so I quantized the observations in the simulator to match. This quantization on its own is not a major issue, since a policy trained against continuous observations generalizes to seeing only a subset of them at deployment.

**Servo Deadzone** The servo will not move at all for small changes, around 0.3 to 1.0 normalized steps, depending on the specific joint and the current joint configuration. This, I believe, is due to static friction in the gearbox that prevents small movements. I implemented a deadzone band in the simulator and it degraded the performance of the policy. The arm would sag and sway due to gravity in the simulator more than it would on the real robot. To overcome this I need to model static friction within the simulator, but I have yet to determine a method for that.

Throughout all of this I kept domain randomization on. The goal of measuring these parameters was never to produce one perfect model of one specific robot, it was to center the range the policy trains across on something close to the real thing.

## How much this actually helped

Once I had all of these changes in, I ran the new policy and the previously deployed policy against the same set of goals on the real arm, and scored both on metrics I had picked before running anything. I wanted to avoid the temptation of looking at the results first and deciding afterwards what counted as an improvement.

The oscillation around the goal, which is the problem that started all of this, mostly went away. Scoring 40 goals per policy:

| | Original Policy | Intermediate Policy | Final Policy |
|---|---:|---:|---:|
| goals showing jitter | 15% | 0% | 0% |
| wobble while holding, p95 (encoder ticks) | 28 | 0.0 | 0.0 |
| settled error | 4.9 mm | 6.6 mm | 4.9 mm |

The buzzing at the goal is gone, and the arm now arrives and stops, which it did not do before.

What did not really change was the accuracy. The final distance to the goal stayed about where it was, within a few millimeters either way, which was not what I expected going in. All of this work bought smoothness and predictability rather than precision.

I do not have a confirmed answer for what sets the accuracy floor. My first assumption was the domain randomization, since the policy trains against a per-episode calibration offset of about 0.6° per joint, which works out to roughly 5mm at the end-effector and lines up almost exactly with the error I measure. I have found that letting the agent train longer allows it to achieve a policy that is much closer to the goal, and settles. I tried with and without a gated L2 action rate penalty, which applied a large penalty for changing actions near the goal, but I found that this did not really improve performance over the standard reward distribution, especially when performing longer training runs.

The plots below show training progress for the deployed policy alongside its behavior in simulation. The reward shown is the fine-grained end-effector position term, which pays out on every step the end-effector is near the goal, and pays more the closer it is.

<p float="left">
  <img src='{{ site.baseurl }}/assets/img/reach_training.png' width='49%'> <img src='{{ site.baseurl }}/assets/img/so101_reach.gif' width="49%">
</p>

## What is still unsolved

I would like to have a policy that learns to be both incredibly precise, and incredibly stable. This ideal policy moves the end-effector exactly to the goal, and stops it there. I am unsure if this is 100% possible with the randomization in the task, but my goal is to get it as close as possible.

Only the reach task has been deployed on the real robot. I have push and lift trained and evaluated in the simulator against the corrected model, but I do not have the confidence that they will safely deploy to the real robot. Currently, there is no method for detecting the live position of the cube, which is necessary for these two tasks.

## What I would take to the next robot

First, this experience reinforced my belief that it is almost always best to understand the whole picture, or as much of it as you can, before starting work on a project. On top of this, systematically working through a problem is the only way to truly track progress and ensure that you are not wasting time.

The second is to decide what counts as an improvement before running the test. I was repeatedly tempted to run a comparison, look at all of the numbers, and pick whichever one had improved. I find that the best practice is to determine beforehand what constitutes a success for a test, or at least what you think it should look like. The possible failure modes are also something to think about, and both of these can help you reason through the results of the test and what they mean, good or bad.

The third is that the metric I cared about most was not the one that predicted real-world behavior. The success rate in simulation turned out to be a weak predictor of how a policy would actually behave on the arm, while the smoothness metrics lined up well. The success rate in the simulation only matters for the real-world behavior if the simulator actually matches the real world, or at least very closely! At one point the policy with the better success rate in the simulator was clearly the worse one to put on the robot.
