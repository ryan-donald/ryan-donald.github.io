---
title: "RRT* implementation in Python"
excerpt: "Implementation of the RRT* algorithm within a jupyter notebook<br/><img src='/images/rrt_star_1.gif'>"
collection: portfolio
---

```python
%matplotlib widget
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
```

Below is the Node class, which is a simple data structure to represent each node in the RRT* tree, with the following member variables:  
- $position$, x,y position pair  
- $parent$, Node object represnting the parent Node in the RRT* tree
- $cost$, Distance along the tree from the start Node to this Node


```python
class Node:
    def __init__(self, position, parent=None):
        self.position = position
        self.parent = parent
        self.cost = 0.0
```

Below is the RRTStar class, which implements the RRT* algorithm. Within this class, there are various member variables and functions defined. This algorithm randomly samples points within a map, creating a tree for path planning in the map. If nodes are further away than a maximum distance, they will be moved closer to the node directly along a line connecting the sampled node and the nearest node. If this path is interrupted by an object, the node will not be added. The specific improvement of RRT* is that the parent of the new node is determined by checking the cost of each nearby node and choosing the parent which will result in the minimum cost. After this, it checks if any of the nearby nodes will benefit from having the new node as it's parent node. Additionally, RRT* will continue sampling and updating the tree after a solution is found, converging towards an optimal path, unlike RRT which stops once a path is found.



```python
class RRTStar:
    def __init__(self, start, goal, world_size, obstacles=[], step_size=0.5, max_iter=5000):
        self.start = Node(start)
        self.goal = Node(goal)
        self.world_size = world_size
        self.step_size = step_size
        self.max_iter = max_iter
        self.obstacles = obstacles
        self.node_list = [self.start]
        self.reached_goal = False

    def generate_random_node(self):
        # generate a random node within the map
        return Node(np.random.uniform(0, self.world_size, size=2))
    
    def steer(self, from_node, to_node):
        # calculate angle between from_node and to_node
        theta = np.arctan2(to_node.position[1] - from_node.position[1],
                           to_node.position[0] - from_node.position[0])
        
        # clip distance to step_size
        if np.linalg.norm(to_node.position - from_node.position) < self.step_size:
            new_position = to_node.position
        else:
            new_position = from_node.position + self.step_size * np.array([np.cos(theta), np.sin(theta)])

        # create new node and update its cost and parent
        new_node = Node(new_position, from_node)
        new_node.cost = from_node.cost + np.linalg.norm(new_position - from_node.position)
        new_node.parent = from_node
        return new_node
    
    def nearest_node(self, node):
        # find the nearest node in the map to a given node
        dlist = [np.linalg.norm(n.position - node.position) for n in self.node_list]
        min_index = dlist.index(min(dlist))
        return self.node_list[min_index]
    
    def find_nearby_nodes(self, new_node, radius):
        # find nearby nodes in the map to new_node within a specified radius
        nearby_nodes = []
        for node in self.node_list:
            if np.linalg.norm(node.position - new_node.position) <= radius:
                nearby_nodes.append(node)
        return nearby_nodes
    
    def choose_parent(self, new_node, nearby_nodes):
        # choose the best parent for new_node from nearby_nodes based on cost
        if not nearby_nodes:
            return new_node
        
        # check collision before considering this node as a parent
        costs = []
        valid_nodes = []
        for node in nearby_nodes:
            if not self.check_collision(node, new_node):
                cost = node.cost + np.linalg.norm(node.position - new_node.position)
                costs.append(cost)
                valid_nodes.append(node)
        
        if not valid_nodes:
            return new_node

        # select parent with minimum cost
        min_cost_index = costs.index(min(costs))
        best_parent = valid_nodes[min_cost_index]
        new_node.parent = best_parent
        new_node.cost = costs[min_cost_index]
        return new_node
    
    def rewire(self, new_node, nearby_nodes):

        # check all nearby_nodes to see if we can decrease their cost by rewiring through new_node
        for node in nearby_nodes:
            cost = new_node.cost + np.linalg.norm(node.position - new_node.position)
            if (cost < node.cost) and (self.check_collision(new_node, node) == False):
                node.parent = new_node
                node.cost = cost

                # update costs of all descendants
                self.update_descendants_cost(node)
    
    def update_descendants_cost(self, node):
        # update costs of all descendants of the given node
        for n in self.node_list:
            if n.parent == node:
                n.cost = node.cost + np.linalg.norm(n.position - node.position)
                self.update_descendants_cost(n)

    def check_collision(self, node1, node2):
        # check collision between line segment of node1 to node2, and each obstacle
        p1 = np.asarray(node1.position, dtype=float)
        p2 = np.asarray(node2.position, dtype=float)
        v = p2 - p1
        seg_len2 = np.dot(v, v)
        eps = 1e-8

        for center, radius in self.obstacles:
            c = np.asarray(center, dtype=float)
            if seg_len2 <= eps:
                if np.linalg.norm(p1 - c) <= radius:
                    return True
                continue

            t = np.dot(c - p1, v) / seg_len2
            t_clamped = np.clip(t, 0.0, 1.0)
            closest = p1 + t_clamped * v
            if np.linalg.norm(closest - c) <= radius:
                return True

        return False
```

Below is an animation function, which executes the RRT* algorithm, and plots the tree, obstacles, start, and goal within matplotlib.


```python
def animate(i, rrt_star, ax, save_counter):
    # perform multiple iterations per animation frame for faster and smoother growth
    iters_per_frame = 100
    for _ in range(iters_per_frame):
        if len(rrt_star.node_list) >= rrt_star.max_iter:
            break
        rand_node = rrt_star.generate_random_node()
        nearest_node = rrt_star.nearest_node(rand_node)
        new_node = rrt_star.steer(nearest_node, rand_node)
        
        if rrt_star.check_collision(nearest_node, new_node):
            continue 
        nearby_nodes = rrt_star.find_nearby_nodes(new_node, radius=2.0)
        new_node = rrt_star.choose_parent(new_node, nearby_nodes)
        rrt_star.rewire(new_node, nearby_nodes)
        rrt_star.node_list.append(new_node)

        # check if the goal is reached
        if np.linalg.norm(new_node.position - rrt_star.goal.position) < rrt_star.step_size:
            rrt_star.reached_goal = True
            rrt_star.goal.parent = new_node
            rrt_star.goal.cost = new_node.cost + np.linalg.norm(new_node.position - rrt_star.goal.position)
            rrt_star.node_list.append(rrt_star.goal)

    # reset plot
    ax.clear()
    ax.set_xlim(0, rrt_star.world_size)
    ax.set_ylim(0, rrt_star.world_size)
    ax.set_title(f"RRT* Growth - Nodes: {len(rrt_star.node_list)}")

    # draw obstacles
    for center, radius in rrt_star.obstacles:
        circle = plt.Circle(center, radius, color='gray', alpha=0.5)
        ax.add_patch(circle)

    # draw edges
    for node in rrt_star.node_list:
        if node.parent is not None:
            x = [node.position[0], node.parent.position[0]]
            y = [node.position[1], node.parent.position[1]]
            ax.plot(x, y, color='blue', linewidth=0.8)

    # draw nodes
    pts = np.array([n.position for n in rrt_star.node_list])
    if pts.size:
        ax.scatter(pts[:,0], pts[:,1], c='black', s=5)

    # draw start and goal
    ax.scatter(rrt_star.start.position[0], rrt_star.start.position[1], c='green', s=50, label='Start')
    ax.scatter(rrt_star.goal.position[0], rrt_star.goal.position[1], c='red', s=50, label='Goal')

    # if reached goal, draw path
    if rrt_star.reached_goal:
        path = []
        node = rrt_star.goal
        while node is not None:
            path.append(node.position)
            node = node.parent
        path = np.array(path[::-1])
        ax.plot(path[:,0], path[:,1], color='magenta', linewidth=2, label='Path')
        ax.legend()
    
    plt.savefig('rrt_star_progress' + str(i) + '.png')
```

Below is the initialization of the RRT* class, matplotlib plot, and the function call to begin and animate the RRT* algorithm.


```python
start = np.array([5,5])
goal = np.array([17,17])

obstacles = [
    (np.array([10, 10]), 2.0),
    (np.array([6, 14]), 1.5),
    (np.array([14, 6]), 1.5),
    (np.array([12, 16]), 2.0),
    (np.array([16, 12]), 2.0),
    (np.array([8, 8]), 1.5)
]
world_size = 20

rrt_star = RRTStar(start, goal, world_size, obstacles=obstacles, step_size=0.5, max_iter=5000)

fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlim(0, rrt_star.world_size)
ax.set_ylim(0, rrt_star.world_size)
ax.set_aspect('equal')

save_counter = 0
animated_rrt = animation.FuncAnimation(fig, animate, fargs=(rrt_star, ax, save_counter), frames=10000, interval=30)
plt.show()
```

