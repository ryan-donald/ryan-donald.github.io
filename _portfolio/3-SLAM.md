---
title: "Visual SLAM with OpenCV"
excerpt: "Visual SLAM pipeline tested with the KITTI dataset <br><img src='/images/kitti_02_slam_2d.svg' onerror this.src='/images/kitti_02_slam_2d.png' width='50%'>"
collection: portfolio
---
I improved my visual odometry pipeline that I had previously implemented and tested with the KITTI dataset, to include keyframing, loop closures, and graph optimization. I use the same KITTI dataset as in my visual odometry pipeline, information on their dataset can be found [here](https://www.cvlibs.net/datasets/kitti/index.php).

Below I will describe the changes to my visual odometry pipeline, to improve the algorithm to be a full SLAM implementation. 
- First, I utilize my visual odometry pipeline which determines a transformation in 3D space between each timestep.
- Next, using the predicted transformation between timesteps, I create a keyframe at the initial timestep, and at any timestep that differs by more than a threshold in position or rotation from the previous keyframe. 
- I then compare information about the current keyframe to previous keyframes to find a match, signaling a loop-closure. To determine matches, a Bag-of-Words method of describing images is uses, alongside a KD-tree for efficiently searching previously found keyframes.
- Now, the algorithm now has a pose graph defined by the odometry predictions from the visual-odometry algorithm, and loop-closure constraints. This is then optimized using the g2o open-source graph optimization library. This optimization can be done during collection, or after. In my case, I do this at the end for time efficiency


My code can be found [here](https://github.com/ryan-donald/slam/)  
  
  
<img src='/images/kitti_02_slam_2d.svg' onerror this.src='/images/kitti_02_slam_2d.png'>