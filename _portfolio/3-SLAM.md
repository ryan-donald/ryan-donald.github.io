---
title: "Visual SLAM with OpenCV"
excerpt: "Visual SLAM pipeline tested with the KITTI dataset <br><img src='/images/kitti_07_slam_2d.svg' onerror this.src='/images/kitti_07_slam_2d.png' width='50%'>"
collection: portfolio
---
I improved my visual odometry pipeline that I had previously implemented and tested with the KITTI dataset, to include keyframing, loop closures, and graph optimization. I use the same KITTI dataset as in my visual odometry pipeline, information on their dataset can be found [here](https://www.cvlibs.net/datasets/kitti/index.php).

Below I will describe the changes to my visual odometry pipeline, to improve the algorithm to be a full SLAM implementation. 
- First, I utilize my visual odometry pipeline which determines a transformation in 3D space between each timestep.
- Next, using the predicted transformation between timesteps, I create a keyframe at the initial timestep, and at any timestep that differs by more than a threshold in position or rotation from the previous keyframe. 
- I then compare information about the current keyframe to previous keyframes to find a match, signaling a loop-closure. To determine matches, a Bag-of-Words method of describing images is uses, alongside a KD-tree for efficiently searching previously found keyframes.
- Now, the algorithm now has a pose graph defined by the odometry predictions from the visual-odometry algorithm, and loop-closure constraints. This is then optimized using the g2o open-source graph optimization library. This optimization can be done during collection, or after. In my case, I do this at the end for time efficiency

As you can see below, I have run this algorithm and compared the resulting trajectories with my non-optimized, visual odometry only tratrajectories. As you can see in the results, whenever the car detects it is in a location it has been in before, it adds a constraint to the pose graph resulting in a trajectory closer to the ground truth. There is still a large amount of error as a result of drift from the visual odometry pipeline, but this algorithm allows for a reduction in this drift within trajectories with loops. 

One issue, however, is shown in the sequence '02'. Near the end of the trajectory the car travels through a location it has been previously, however this time it is traveling in the opposite direction, and fails to detect the loop.

My code can be found [here](https://github.com/ryan-donald/slam/)  
  
  
<img src='/images/kitti_02_slam_2d.svg' onerror this.src='/images/kitti_02_slam_2d.png'>
<img src='/images/kitti_05_slam_2d.svg' onerror this.src='/images/kitti_05_slam_2d.png'>
<img src='/images/kitti_07_slam_2d.svg' onerror this.src='/images/kitti_07_slam_2d.png'>