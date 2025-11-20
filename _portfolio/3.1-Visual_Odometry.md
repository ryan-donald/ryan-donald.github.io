---
title: "Visual Odometry with OpenCV"
excerpt: "Visual Odometry pipeline tested with the KITTI dataset <br><img src='/images/kitti_05_ground_truth_vs_estimated_2d.svg' onerror this.src='/images/kitti_05_ground_truth_vs_estimated_2d.png' width='50%'>"
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
  
Below, estimated trajectories for three of the sequences can be seen. The top-down image is shown, as this shows how well the implementation can track the path of the vehicle on a map.
  
  
<img src='/images/kitti_05_ground_truth_vs_estimated_2d.svg' onerror this.src='/images/kitti_05_ground_truth_vs_estimated_2d.png'>
<img src='/images/kitti_07_ground_truth_vs_estimated_2d.svg' onerror this.src='/images/kitti_07_ground_truth_vs_estimated_2d.png'>
<img src='/images/kitti_08_ground_truth_vs_estimated_2d.svg' onerror this.src='/images/kitti_08_ground_truth_vs_estimated_2d.png'>
