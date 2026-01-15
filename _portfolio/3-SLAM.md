---
title: "Visual Odometry Pipeline and Visual Slam Pipeline with the KITTI dataset in Python"
excerpt: "Visual odometry pipeline, and a visual SLAM pipeline. Both tested with the KITTI dataset <br><img src='/images/kitti_05_ground_truth_vs_estimated_2d.svg' onerror this.src='/images/kitti_05_ground_truth_vs_estimated_2d.png' width='49%'><img src='/images/kitti_07_slam_2d.svg' onerror this.src='/images/kitti_07_slam_2d.png' width='49%'>"
collection: portfolio
---
I have implemented a visual odometry pipeline, tested with the KITTI dataset. This dataset was captured by the Karlsruhe Institute of Technology and Toyota Technological Institute and Chicago by driving their test vehicle, equipped with stereo cameras, a Velodyne laser scanner, and a combined GPS/IMU inertial navigation system. There are 10 sequences of data collected by driving this system on the street in Karlsruhe. More information on this project can be found [here](https://www.cvlibs.net/datasets/kitti/index.php).  
  
I utilized the data from this, as I find the self-driving car problem to be very interesting. Additionally, this data is useful as they provide the ground-truth information for each sequence, which helped me verify my implementation. This algorithm follows the outline below:  
- First, the left and right stereo images of each frame are loaded. These are pre-rectified in the dataset. Additionally, the left camera's image from the next frame is loaded.
- Next, utilizing the stereo images of the current frame, a disparity map is computed and transformed into a depth map.  
- Then, keypoints and descriptors are extracted from the left image of the current frame and the next frame.  
- Features are matched between the current frame's left image and the next frame's left image, and filtered with the ratio test to ensure strong matches.  
- Using these good matches, the sets of keypoints from each image are filtered.  
- Next, the points in the 3-dimensional space are reconstructed, and filtered for invalid depth values.
- Using the set of 3-dimensional points, and the corresponding points in the next image, the camera's new pose is estimated using Perspective-n-Point RANSAC. 

My code can be found [here](https://github.com/ryan-donald/Visual-Odometry)  
  
After this, I then improved my visual odometry pipeline to include keyframing, loop closures, and graph optimization. I use the same KITTI dataset as in my visual odometry pipeline.

The additions to my visual odometry pipeline to implement a SLAM algorithm are detailed below.  
- First, I utilize my visual odometry pipeline which determines a transformation in 3D space between each timestep.
- Next, using the predicted transformation between timesteps, I create a keyframe at the initial timestep, and at any timestep that differs by more than a threshold in position or rotation from the previous keyframe. 
- I then compare information about the current keyframe to previous keyframes to find a match, signaling a loop-closure. To determine matches, a Bag-of-Words method of describing images is uses, alongside a KD-tree for efficiently searching previously found keyframes.
- Now, the algorithm now has a pose graph defined by the odometry predictions from the visual-odometry algorithm, and loop-closure constraints. This is then optimized using the g2o open-source graph optimization library. This optimization can be done during collection, or after. In my case, I do this at the end for time efficiency

As you can see below, I have run this algorithm and compared the resulting trajectories with my non-optimized, visual odometry only tratrajectories. As you can see in the results, whenever the car detects it is in a location it has been in before, it adds a constraint to the pose graph resulting in a trajectory closer to the ground truth. There is still a large amount of error as a result of drift from the visual odometry pipeline, but this algorithm allows for a reduction in this drift within trajectories with loops. 

One issue, however, is shown in the sequence '02'. Near the end of the trajectory the car travels through a location it has been previously, however this time it is traveling in the opposite direction, and fails to detect the loop.

My code can be found [here](https://github.com/ryan-donald/slam/)  


Below, estimated trajectories for three of the sequences can be seen from the visual odometry only pipeline. The top-down image is shown, as this shows how well the implementation can track the path of the vehicle on a map.
  
<img src='/images/kitti_05_ground_truth_vs_estimated_2d.svg' onerror this.src='/images/kitti_05_ground_truth_vs_estimated_2d.png'>
<img src='/images/kitti_07_ground_truth_vs_estimated_2d.svg' onerror this.src='/images/kitti_07_ground_truth_vs_estimated_2d.png'>
<img src='/images/kitti_08_ground_truth_vs_estimated_2d.svg' onerror this.src='/images/kitti_08_ground_truth_vs_estimated_2d.png'>

Next, I showcase the estimated trajectories for three sequences from the SLAM pipeline, again with a top-down image shown.

<img src='/images/kitti_02_slam_2d.svg' onerror this.src='/images/kitti_02_slam_2d.png'>
<img src='/images/kitti_05_slam_2d.svg' onerror this.src='/images/kitti_05_slam_2d.png'>
<img src='/images/kitti_07_slam_2d.svg' onerror this.src='/images/kitti_07_slam_2d.png'>
