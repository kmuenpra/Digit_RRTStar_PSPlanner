from typing import List
from . import BasePlanner

import os
import sys
import math
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + "/pypolo/planners/RRTStar_package")

import env
import plotting
import utils
import queue


class Node:
    def __init__(self, n):
        self.x = n[0]
        self.y = n[1]
        self.parent = None


class GlobalRRTStar(BasePlanner):
        
    # Constructor Method
    def __init__(
        self,
        task_extent: List[float],
        rng: np.random.RandomState,
        x_start: tuple, 
        x_goal: tuple, 
        step_len: float, 
        goal_sample_rate: float, 
        search_radius: float, 
        iter_max: int, 
        ) -> None:
        
        #Initialize BasePlanner
        super().__init__(task_extent, rng)
        
        self.s_start = Node(x_start)
        self.s_goal = Node(x_goal)
        self.step_len = step_len
        self.goal_sample_rate = goal_sample_rate
        self.search_radius = search_radius
        self.iter_max = iter_max
        self.vertex = [self.s_start]
        self.path = []

        self.env = env.Env()
        self.plotting = plotting.Plotting(x_start, x_goal)
        self.utils = utils.Utils()
        self.utils.delta = 0.3

        self.x_range = (task_extent[0], task_extent[1])
        self.y_range = (task_extent[2], task_extent[3])
        self.obs_circle = self.env.obs_circle
        self.obs_rectangle = self.env.obs_rectangle
        self.obs_boundary = self.env.obs_boundary
        
        self.goal_radius = 1
        
        self.max_turn_ang = np.deg2rad(30)

    def plan(self, heading_c=0):
                
        for k in range(self.iter_max):
            
            node_rand = self.generate_random_node(self.goal_sample_rate)
            node_near = self.nearest_neighbor(self.vertex, node_rand)
            node_new = self.new_state(node_near, node_rand)

            if k % 500 == 0:
                print(k)
        
            if node_new and not self.utils.is_collision(node_near, node_new): 
                neighbor_index = self.find_near_neighbor(node_new)
                self.vertex.append(node_new)

                if neighbor_index:
                    
                    self.choose_parent(node_new, neighbor_index)
                    self.rewire(node_new, neighbor_index)
                        
                if math.hypot(node_new.x - self.s_goal.x, node_new.y - self.s_goal.y) <= self.goal_radius:
                    break
                    
        # index = self.search_goal_parent()
        # self.path = self.extract_path(self.vertex[index])
        if len(self.vertex) == 1:
            index = 0
        else:
            index = self.search_goal_parent()
        self.path, self.path_vertex = self.extract_path(self.vertex[index])
        self.k = k
        self.path_vertex.pop(-1)
        
        #Check to if any nodes in optimal path can be rewired together

        if len(self.path_vertex) > 3:
            # for i in np.flip(np.arange(0, len(self.path_vertex) - 1)):
            self.rewire_path(s_start_index=0)
            
    def is_goal_reachable(self):
        
        if len(self.path) == 0:
            print("There is currently no generated path to evaluate.")
            return False
        
        dist , _  = self.get_distance_and_angle(self.path_vertex[-1], self.s_goal)
        
        if dist <= self.goal_radius:
            print("Goal is reached!")
            return True
        else:
            print("Cannot reach the goal.")
            return False
        
        
        
    def reset_tree(self, start:tuple=None, goal:tuple=None):
        
        if start is not None:
            self.s_start = Node(start)
            
        if goal is not None:
            self.s_goal = Node(goal)
        
        self.path = []
        self.vertex = [self.s_start]
        self.path_vertex = []
        
        
    def rewire_path(self, s_start_index):   
        
        #Rewiring
        i = s_start_index
        
        if i > 0:
            _ , heading_c  = self.get_distance_and_angle(self.path_vertex[s_start_index - 1], self.path_vertex[s_start_index])
        else:
            heading_c = 0
        
        for path_node_index in np.arange(s_start_index + 1, len(self.path_vertex) - 1): 
            
            #Two step planning, make sure turning angles are valid for two forward steps
            dist , heading_n  = self.get_distance_and_angle(self.path_vertex[s_start_index], self.path_vertex[path_node_index])
            _ , heading_nn  = self.get_distance_and_angle(self.path_vertex[path_node_index], self.path_vertex[path_node_index + 1])

            
            if (abs(heading_n - heading_c) <= self.max_turn_ang) and (abs(heading_nn - heading_n) <= self.max_turn_ang) and \
                not self.utils.is_collision(self.path_vertex[s_start_index], self.path_vertex[path_node_index]):
                
                i = path_node_index
                
        if i > s_start_index:
            #if valid nodes found, set it as the child of new starting point
            if dist <= self.step_len:
                self.path_vertex[i].parent = self.path_vertex[s_start_index]
            else:
                self.add_interpolated_parents(self.path_vertex[s_start_index], self.path_vertex[i], step_length=self.step_len)
            
            #Save replanned path
            index = self.search_goal_parent()
            self.path, self.path_vertex = self.extract_path(self.vertex[index])
            self.path_vertex.pop(-1)
            
        if (i == s_start_index) or (i >= len(self.path_vertex) - 2):
            return
        else:
            self.rewire_path(s_start_index=i)
            
    def add_interpolated_parents(self, parent_node, child_node, step_length=None):
        
        if step_length is None:
            step_length = self.step_len
        
        parent_index = self.vertex.index(parent_node)
        last_child_index = self.vertex.index(child_node)
        
        # Calculate the distance between the two points
        dist, _ = self.get_distance_and_angle(parent_node, child_node)
        
        # Calculate the number of steps needed
        num_steps = int(dist / step_length)
        
        if num_steps > 1:
            # Calculate the step size in each dimension
            step_x = (child_node.x - parent_node.x) / num_steps
            step_y = (child_node.y - parent_node.y) / num_steps
            
            # Interpolated Nodes
            for i in range(num_steps):
                if i == 0:
                    continue
                
                new_x = parent_node.x + i * step_x
                new_y = parent_node.y + i * step_y
                
                interp_node = Node((new_x, new_y))
                interp_node.parent = self.vertex[parent_index]
                
                self.vertex.append(interp_node)
                parent_index = self.vertex.index(interp_node)
                
        self.vertex[last_child_index].parent = self.vertex[parent_index]
        
    # def branch_all_feasible_nodes(self, node_start, heading_default=0):
    #     '''
    #     TODO:
    #     Check if previous dheading was a positive or negative
    #         - if last dheading was positive, probably dont want to turn negative with large magnitude. 
    #         Otherwise, Digit would sway too much.
    #     '''
        
        
    #     #Define current position
    #     x = node_start.x
    #     y = node_start.y
        
    #     #Define current heading
    #     if node_start.parent is None:
    #         heading_c = heading_default
    #     else:
    #         _, heading_c = self.get_distance_and_angle(node_start.parent, node_start)
            
    #     angle = np.random.uniform(low=(heading_c - self.max_turn_ang), high=(heading_c - self.max_turn_ang), size=1)
    #     admiss_neighbor = Node((x + self.step_len * np.cos(angle), y + self.step_len * np.sin(angle)))
    #     admiss_neighbor.parent = node_start

    #     return admiss_neighbor
        
    def plot_map(self):
        
        if self.path:
            self.plotting.animation(self.vertex, self.path, "rrt*, total Iter. = " + str(self.k), )
        else:
            self.plotting.plot_grid("Plain Environment")

    def new_state(self, node_start, node_goal):
        dist, theta = self.get_distance_and_angle(node_start, node_goal)

        dist = min(self.step_len, dist)
        node_new = Node((node_start.x + dist * math.cos(theta),
                         node_start.y + dist * math.sin(theta)))

        node_new.parent = node_start

        return node_new

    def choose_parent(self, node_new, neighbor_index):
        
        cost = [self.get_new_cost(self.vertex[i], node_new) for i in neighbor_index]

        cost_min_index = neighbor_index[int(np.argmin(cost))]
        node_new.parent = self.vertex[cost_min_index]

    def rewire(self, node_new, neighbor_index):
        for i in neighbor_index:
            node_neighbor = self.vertex[i]
            
            if self.cost(node_neighbor) > self.get_new_cost(node_new, node_neighbor):
                node_neighbor.parent = node_new

    def search_goal_parent(self):
        dist_list = [math.hypot(n.x - self.s_goal.x, n.y - self.s_goal.y) for n in self.vertex]
        node_index = [i for i in range(len(dist_list)) if dist_list[i] <= self.goal_radius]

        # print("Searh Goal Parent - Global")
        # print(len(dist_list))
        # print(len(node_index))
        # print(len(self.vertex))
        
        if len(node_index) > 0:
            cost_list = [dist_list[i] + self.cost(self.vertex[i]) for i in node_index
                            if not self.utils.is_collision(self.vertex[i], self.s_goal)]
            if len(cost_list) > 0:
                return node_index[int(np.argmin(cost_list))]

        return len(self.vertex) - 1

    def get_new_cost(self, node_start, node_end):
        dist, _ = self.get_distance_and_angle(node_start, node_end)

        return self.cost(node_start) + dist

    def generate_random_node(self, goal_sample_rate):
        delta = self.utils.delta

        if np.random.random() > goal_sample_rate:
            return Node((np.random.uniform(self.x_range[0] + delta, self.x_range[1] - delta),
                         np.random.uniform(self.y_range[0] + delta, self.y_range[1] - delta)))

        return self.s_goal

    def find_near_neighbor(self, node_new):
        n = len(self.vertex) + 1
        r = min(self.search_radius * math.sqrt((math.log(n) / n)), self.step_len)

        dist_table = [math.hypot(nd.x - node_new.x, nd.y - node_new.y) for nd in self.vertex]
        dist_table_index = [ind for ind in range(len(dist_table)) if dist_table[ind] <= r and
                            not self.utils.is_collision(node_new, self.vertex[ind])]

        return dist_table_index

    @staticmethod
    def nearest_neighbor(node_list, n):
        return node_list[int(np.argmin([math.hypot(nd.x - n.x, nd.y - n.y)
                                        for nd in node_list]))]

    @staticmethod
    def cost(node_p):
        node = node_p
        cost = 0.0

        while node.parent:
            cost += math.hypot(node.x - node.parent.x, node.y - node.parent.y)
            node = node.parent
        
        return cost

    def update_cost(self, parent_node):
        OPEN = queue.QueueFIFO()
        OPEN.put(parent_node)

        while not OPEN.empty():
            node = OPEN.get()

            if len(node.child) == 0:
                continue

            for node_c in node.child:
                node_c.Cost = self.get_new_cost(node, node_c)
                OPEN.put(node_c)

    # def extract_path(self, node_end):
    #     path = [[self.s_goal.x, self.s_goal.y]]
    #     node = node_end

    #     while node.parent is not None:
    #         path.append([node.x, node.y])
    #         node = node.parent
    #     path.append([node.x, node.y])

    #     return path
    
    def extract_path(self, node_end):
        path = [[self.s_goal.x, self.s_goal.y]]
        path_vertex = [self.s_goal]
        
        node = node_end

        while node.parent is not None:
            path.append([node.x, node.y])
            path_vertex.append(node)
            
            node = node.parent
            
        path.append([node.x, node.y])
        path_vertex.append(node)
        

        return list(reversed(path)), list(reversed(path_vertex))

    @staticmethod
    def get_distance_and_angle(node_start, node_end):
        dx = node_end.x - node_start.x
        dy = node_end.y - node_start.y
        return math.hypot(dx, dy), math.atan2(dy, dx)
    
    def update_obs(self, obs_cir, obs_bound, obs_rec, obstacles):

            # #Convert Irregular Shape into multiple circles
            # n_clusters = 4
            # for path in obstacles:
            #     centers, radii = get_clusters(path.vertices, n_clusters)
            #     for i in range(len(centers)):
            #         circ = centers[i].tolist()
            #         circ.append(radii[i])
                    
            #         obs_cir.append(circ)
                    
            # obs_cir = remove_inner_circles(obs_cir)
            
        
            self.obs_circle = obs_cir
            self.obs_boundary = obs_bound
            self.obs_rectangle = obs_rec
            # self.obstacle = obstacles
            
            self.env.obs_circle = self.obs_circle
            self.env.obs_boundary = self.obs_boundary
            self.env.obs_rectangle = self.obs_rectangle
            # self.env.obstacle = obstacles
            
            self.plotting.obs_circle = self.obs_circle
            self.plotting.obs_boundary = self.obs_boundary
            self.plotting.obs_rectangle = self.obs_rectangle
            # self.plotting.obstacle = obstacles
            
            self.utils.obs_circle = self.obs_circle
            self.utils.obs_boundary = self.obs_boundary
            self.utils.obs_rectangle = self.obs_rectangle
            # self.utils.obstacle = obstacles
            
#------------------------------------------------------------

from sklearn.cluster import KMeans
from matplotlib.patches import Circle

def get_clusters(points, n_clusters):
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(points)
    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    # Plot the points and circles
    # plt.scatter(points[:, 0], points[:, 1], c=labels, cmap='viridis')
    
    centers = []
    radii = []
    
    for i in range(n_clusters):
        cluster_points = points[labels == i]
        cluster_center = cluster_centers[i]
        
        # Calculate the radius of the circle that covers the cluster points
        radius = np.max(np.linalg.norm(cluster_points - cluster_center, axis=1))
        
        # Plot the circle
        # circle = Circle(cluster_center, radius, color='red', alpha=0.2)
        # plt.gca().add_patch(circle)
        
        centers.append(cluster_center)
        radii.append(radius)

    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.title('Clustered Points with Circles')
    # plt.axis('equal')
    # plt.show()

    return centers, radii

def remove_inner_circles(circles):
    # Create a list to store circles that are not fully contained within any other circle
    valid_circles = []
    
    # Iterate over each circle
    for i in range(len(circles)):
        is_valid = True
        for j in range(len(circles)):
            if i != j:
                # Check if circle i is fully contained within circle j
                distance = np.linalg.norm(np.array(circles[i][:2]) - np.array(circles[j][:2]))
                if distance + circles[i][2] <= circles[j][2] + 0.6:
                    is_valid = False
                    break
        if is_valid:
            valid_circles.append(circles[i])

    return valid_circles

