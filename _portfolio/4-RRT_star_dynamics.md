---
title: "RRT* with differential-drive dynamics implementation in Python"
excerpt: "Implementation of the RRT* algorithm with differential-drive dynamics within a jupyter notebook<br/><img src='/images/rrt_star_dynamics_website.gif'>"
collection: portfolio
---
This implementation of the RRT* algorithm, with the dynamics of a differential-drive robot is a continuation of the RRT* implementation I have made. This allows for dynamically possible motions for the robot, as opposed to straight line connections between nodes.


```python
%matplotlib widget
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
```

Below is the Node class from my previous implementation of RRT*. This was adjusted to store the pose (x, y, theta) of the node, instead of simply the position (x,y). Additionally, the trajectory from the parent to this node is stored.


```python
class Node:
    def __init__(self, pose, parent=None):
        self.pose = pose
        self.parent = parent
        self.cost = 0.0
        self.trajectory = []
```

Below is the RRTStarDynamics class. This class contains many similar functions as the previous RRTStar class I implemented. For this specific implementation, main difference is the changes to the *steer()* function, the *rewire()* function, the *choose_parent()* function, and the *check_collision()* function. The *simulate_trajectory()* function now utilizes the differential-drive robot dynamics to simulate a discretized trajectory.  
  
  
The *steer()* function now samples *num_samples* trajectories with *simulate_trajectory()*. This is done with a random control input *v, w* based upon the robot's control limits. The trajectory that ends closest to the sampled node is chosen as the location of the new node, alongside the sampled trajectory.  
The *simulate_trajectory()* function creates a trajectory with a given control input.  
The *rewire()* function now simulates *num_samples* trajectories for each nearby node, and if a trajectory ends within a *threshold* of the new node, if the cost is lower that node will be chosen as the new parent.  
The *choose_parent()* function now simulates a trajectory similar to *steer()* to determine a new nodes parent.  
The *check_collision()* function now checks if each pose in the trajectory is within an object, instead of just the mid-point of the trajectory.  



```python
class RRTStarDynamics:
    def __init__(self, start, goal, world_size, obstacles=[], threshold=0.05, step_size=0.5, max_iter=5000):
        self.start = Node(start)
        self.goal = Node(goal)
        self.world_size = world_size
        self.obstacles = obstacles
        self.threshold = threshold
        self.step_size = step_size
        self.max_iter = max_iter
        self.node_list = [self.start]
        self.reached_goal = False

    def generate_random_node(self):
        # generate a random node within the map with random orientation
        position = np.random.uniform(0, self.world_size, size=2)
        theta = np.random.uniform(-np.pi, np.pi)
        return Node(np.array([position[0], position[1], theta]))

    def steer(self, from_node, to_node, num_samples=10, delta_t=0.1, max_time=1.0, v_range=np.array([0.0, 1.0]), w_range=np.array([-np.pi/4, np.pi/4])):
        # calculate the best trajectory from from_node to to_node using sampled control inputs
        best_trajectory = None
        min_cost = float('inf')

        for _ in range(num_samples):
            # sample control inputs
            v = np.random.uniform(v_range[0], v_range[1])
            w = np.random.uniform(w_range[0], w_range[1])

            # simulate the motion for current control inputs
            trajectory = self.simulate_trajectory(from_node, v, w, delta_t, max_time)

            # if trajectory is not valid, skip this sample
            if not trajectory or len(trajectory) == 0:
                continue

            # compute the cost of this trajectory
            dist = np.linalg.norm(trajectory[-1][:2] - to_node.pose[:2])

            # check if this is the best trajectory, if so save it
            if dist < min_cost:
                min_cost = dist
                best_trajectory = trajectory

        return best_trajectory

    def simulate_trajectory(self, from_node, v, w, delta_t, max_time):
        # simulate the trajectory of the robot given the control inputs, using the differential drive model
        trajectory = []
        current_pose = from_node.pose

        # for each timestep, update the pose based on the control inputs and vehicle dynamics
        for _ in np.arange(0, max_time, delta_t):
            x = current_pose[0]
            y = current_pose[1]
            theta = current_pose[2]

            x += v * np.cos(theta) * delta_t
            y += v * np.sin(theta) * delta_t
            theta += w * delta_t

            if(x < 0 or x > self.world_size or y < 0 or y > self.world_size):
                break  # stop if trajectory goes out of bounds

            current_pose = np.array([x, y, theta])
            trajectory.append(current_pose)

        return trajectory

    def nearest_node(self, node):
        # find the nearest node in the map to a given node
        dlist = [np.linalg.norm(n.pose[:2] - node.pose[:2]) for n in self.node_list]
        min_index = np.argmin(dlist)
        return self.node_list[min_index]
    
    def find_nearby_nodes(self, new_node, radius):
        # find nearby nodes in the map to new_node within a specified radius
        nearby_nodes = []
        for node in self.node_list:
            dist = np.linalg.norm(node.pose[:2] - new_node.pose[:2])
            if dist <= radius:
                nearby_nodes.append(node)
        return nearby_nodes
    
    def choose_parent(self, new_node, nearby_nodes):
        # choose the best parent for new_node from nearby_nodes based on cost
        if not nearby_nodes:
            return new_node
        
        # check collision before considering this node as a parent
        min_cost = float('inf')
        best_parent = None
        best_traj = None
        
        # for each nearby node, simulate a trajectory to new_node and compute the cost
        for node in nearby_nodes:
            traj = self.steer(node, new_node)
            if traj and len(traj) > 0 and not self.check_collision(traj):

                endpoint = traj[-1][:2]
                target = new_node.pose[:2]

                # ensure the trajectory ends within self.threshold of new_node
                if np.linalg.norm(endpoint - target) > self.threshold:
                    continue

                # compute the cost of this trajectory, and if it's the best, save it
                cost = node.cost + self.trajectory_cost(traj)
                if cost < min_cost:
                    min_cost = cost
                    best_parent = node
                    best_traj = traj

        # update the nodes parent if a better parent was found
        if best_parent is not None:
            new_node.parent = best_parent
            new_node.cost = min_cost
            new_node.trajectory = best_traj

        return

    def rewire(self, new_node, nearby_nodes):

        # check all nearby_nodes to see if we can decrease their cost by rewiring through new_node
        for nearby_node in nearby_nodes:

            # skip if nearby_node is the parent of new_node or is the start node
            if nearby_node == new_node.parent or nearby_node == self.start:
                continue
            
            # steer from new_node to nearby_node, check if new_node is a better parent for any nearby node
            traj = self.steer(new_node, nearby_node)
            if traj and len(traj) > 0 and not self.check_collision(traj):

                # ensure that trajectory ends within self.threshold of nearby_node
                endpoint = traj[-1][:2]
                target = nearby_node.pose[:2]
                if np.linalg.norm(endpoint - target) > self.threshold:
                    continue
                
                # compute the cost of this trajectory, and if it's better, rewire nearby_node
                cost = nearby_node.cost + self.trajectory_cost(traj)
                if cost < nearby_node.cost:
                    nearby_node.parent = new_node
                    nearby_node.cost = cost
                    nearby_node.trajectory = traj
                    self.update_descendants_cost(nearby_node)

        return
    
    def update_descendants_cost(self, curr_node):
        # update costs of all descendants of the given node
        for node in self.node_list:
            if node.parent == curr_node:
                
                traj = node.trajectory
                if traj and len(traj) > 0:
                    node.cost = curr_node.cost + self.trajectory_cost(traj)
                    self.update_descendants_cost(node)

        return

    def trajectory_cost(self, trajectory):
        # compute the cost of a trajectory by summing the Euclidean distances between consecutive poses
        cost = 0.0
        for i in np.arange(1, len(trajectory)):
            cost += np.linalg.norm(trajectory[i][:2] - trajectory[i-1][:2])
        return cost
    
    def check_collision(self, trajectory):
        # check if any part of the trajectory collides with any obstacles
        for obs in self.obstacles:
            for pose in trajectory:
                if np.linalg.norm(pose[:2] - obs[0][:2]) <= obs[1]:
                    return True
        return False

    def get_path(self):
        # retrieve the path from start to goal by backtracking from goal to start
        if not self.reached_goal:
            return None
        path = []
        node = self.goal
        while node is not None:
            if node.trajectory and len(node.trajectory) > 0:
                path.extend(reversed(node.trajectory))
            else:
                path.append(node.pose)
            node = node.parent
        
        path.reverse()
        return path
```

The *animate()* function performs the same task as in the default RRT* implementation, calling each function to implement the RRT* with robot dynamics, as well as plotting the growth of the algorithm.


```python
def animate(i, rrt_star, ax, save_counter):
    # perform multiple iterations per animation frame for faster and smoother growth
    iters_per_frame = 10

    for _ in range(iters_per_frame):
        # check if max iterations reached
        if len(rrt_star.node_list) >= rrt_star.max_iter:
            break

        # biased sampling towards goal
        if(np.random.random() < 0.05):
            rand_node = Node(rrt_star.goal.pose.copy())
        else:
            rand_node = rrt_star.generate_random_node()

        # find nearest node and steer towards random node
        nearest_node = rrt_star.nearest_node(rand_node)
        new_trajectory = rrt_star.steer(nearest_node, rand_node)

        # if trajectory is invalid or collides, skip
        if not new_trajectory or len(new_trajectory) == 0 or rrt_star.check_collision(new_trajectory):
            continue

        # create a new node at the end of the steered trajectory
        new_pose = new_trajectory[-1]
        new_node = Node(new_pose)
        new_node.trajectory = new_trajectory
        new_node.parent = nearest_node
        new_node.cost = nearest_node.cost + rrt_star.trajectory_cost(new_trajectory)

        # choose the parent for the new node and rewire the tree
        nearby_nodes = rrt_star.find_nearby_nodes(new_node, radius=2.0)
        rrt_star.choose_parent(new_node, nearby_nodes)
        rrt_star.node_list.append(new_node)
        rrt_star.rewire(new_node, nearby_nodes)

        # check if the goal is reached within step_size
        if np.linalg.norm(new_node.pose[:2] - rrt_star.goal.pose[:2]) < rrt_star.step_size:
            new_cost = new_node.cost + np.linalg.norm(new_node.pose[:2] - rrt_star.goal.pose[:2])
            if not rrt_star.reached_goal or new_cost < rrt_star.goal.cost:
                rrt_star.reached_goal = True
                rrt_star.goal.parent = new_node
                rrt_star.goal.cost = new_cost

    # reset plot
    ax.clear()
    ax.set_xlim(0, rrt_star.world_size)
    ax.set_ylim(0, rrt_star.world_size)
    ax.set_title(f"RRT* With Dynamics")

    # draw obstacles
    for obs in rrt_star.obstacles:
        center, radius = obs[0], obs[1]
        circle = plt.Circle(center, radius, color='gray', alpha=0.5)
        ax.add_patch(circle)

    # draw trajectories between nodes
    for node in rrt_star.node_list:

        if node.parent and len(node.trajectory) > 0:
            traj = np.array(node.trajectory)
            ax.plot(traj[:, 0], traj[:, 1], color='blue', linewidth=0.8, alpha=0.7)

    # draw nodes
    pts = np.array([n.pose[:2] for n in rrt_star.node_list])
    if pts.size:
        ax.scatter(pts[:,0], pts[:,1], c='black', s=5, zorder=1)
    
    # draw start and goal
    ax.scatter(rrt_star.start.pose[0], rrt_star.start.pose[1], c='green', s=100, label='Start', zorder=2)
    ax.scatter(rrt_star.goal.pose[0], rrt_star.goal.pose[1], c='red', s=100, label='Goal', zorder=2)

    # if reached goal, draw path from start to goal
    if rrt_star.reached_goal:
        path = rrt_star.get_path()
        if path is not None and len(path) > 0:
            path = np.array(path)
            ax.plot(path[:,0], path[:,1], color='magenta', linewidth=2, label='Path')

    ax.legend(loc='lower right')
```


```python
# define start and goal poses [x, y, theta]
start = np.array([2.0, 2.0, 0.0])
goal = np.array([18.0, 18.0, 0.0])

# define obstacles [[x, y], radius]
obstacles = [
    (np.array([10, 10]), 2.0),
    (np.array([6, 14]), 1.5),
    (np.array([14, 6]), 1.5),
    (np.array([12, 16]), 2.0),
    (np.array([16, 12]), 2.0),
    (np.array([8, 8]), 1.5)
]

# Create and run RRT*
rrt_star = RRTStarDynamics(start, goal, world_size=20, 
                                obstacles=obstacles, 
                                step_size=0.5, 
                                max_iter=5000)

# Animate the planning process
fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlim(0, rrt_star.world_size)
ax.set_ylim(0, rrt_star.world_size)
ax.set_aspect('equal')

animated_rrt = animation.FuncAnimation(fig, animate, fargs=(rrt_star, ax, 0), frames=250, interval=60)
# animated_rrt.save('rrt_star_dynamics_website.gif', writer='pillow', fps=30) # Use 'pillow' writer for GIFs

plt.show()
```