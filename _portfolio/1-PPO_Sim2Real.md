---
title: "PPO Control with the Standard Open ARM101"
excerpt: "Implementation of the Proximal Policy Optimization (PPO) algorithm, tested with the Standard Open ARM101 in IsaacLab and deployed in the real world<br/><img src='https://img.youtube.com/vi/0l35dRZK9nA/0.jpg' width='50%'>"
collection: portfolio
---
Using my implementation of PPO, I wanted to train a model within the IsaacLab simulator, and deploy it on a real-world robot, the SO-ARM101. I have trained two policies for the robot within IsaacLab, and I am currently in the process of deploying the model to the real world. This process can be very complicated, as the simulator needs to be configured, and the task designed in a way that works on a real robot in the real world. Below are the main sim2real disconnects that I have had to address and re-train my model to account for:

LeRobot is used to command this robot and retrieve the current joint-states. These commands and observations are in normalized ranges [-100, 100] for each motor except for the gripper, which is in the range [0, 100]. Through the calibration of the robot, LeRobot abstracts the specific motor encoder values away from the user, and works in terms of a normalized workspace range, representing the limits of each joint. In my initial IsaacLab environment, the joints were controlled directly with radians based on the URDF. To address this gap, I implemented a similar normalization technique for both the observation term and the action term as inputs and outputs of the network.

Additionally, the initial values for the motor parameters did not match the real-world robot. The real-world robot uses a PD controller to move each motor to the commanded joint position. In IsaacLab, the proportional (P) term of the controller is represented by the stiffness value of a joint, and the derivative (D) term of the controller is represented by the damping value of a joint. Additionally, the velocity limit for the joints was initially set at a value around 30% of the real-world velocity limit, which meant that the model was trained on a much slower robot As a result of these mis-matches, the model performed as expected in the simulator, but poorly on the real-world robot. To fix this, I collected data on the motor's step responses, as well as the velocity that the motors moved at. Using this, I tuned the simulator values to closely match the values the real-world motors had, and once I was able to re-train the model, it worked as expected on the real-world robot. 

In the current state of this project, I have the reach task working as expected on the real-world robot. A video of the reach task on the real-world robot can be found [here](https://www.youtube.com/watch?v=0l35dRZK9nA). In this, the robot is controlled by a model which is trained to control and move the robot from any joint state to one where the end-effector is at a specific position in the robot's coordinate frame. Once the end-effector is within 4cm of the target position, a new target position is randomly sampled in the workspace and the robot then moves to that location. As you can see in the video, the motion is not perfectly smooth, as the commands for each joint are specifying joint positions, not velocities, and there is some backlash in the motors themselves as they are inexpensive hobby motors. Regardless, I was able to train a model within IsaacLab and deploy it onto a real-world robot.

My implementation of PPO that I am using can be found [here](https://github.com/ryan-donald/PPO_IsaacLab), and my deployment scripts can be found [here](https://github.com/ryan-donald/so101_ppo).


My trained models are shown below, both the final visual performance, and the average reward during training.

<p float="left">
    <img src='/images/training_plot_episodes_Isaac-Lift-Cube-SO-ARM101-v0.svg' onerror this.src='/images/training_plot_episodes_Isaac-Lift-Cube-SO-ARM101-v0.png' width='49%'> <img src='/images/so101_lift.gif' width="49%">  
</p>
<p float="left">
    <img src='/images/training_plot_episodes_Isaac-Reach-SO-ARM101-v0.svg' onerror this.src='/images/training_plot_episodes_Isaac-Reach-SO-ARM101-v0.png' width='49%'> <img src='/images/so101_reach.gif' width="49%"> 
</p>