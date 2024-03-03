from typing import List
from . import BasePlanner


# class GlobalRRTStar(BasePlanner):
#     def __init__(self, task_extent: List[float], rng: np.random.RandomState) -> None:
#         super().__init__(task_extent, rng)

#     def plan(self, num_points: int, *args, **kwargs) -> np.ndarray:
#         xmin, xmax, ymin, ymax = self.task_extent
#         xs = np.linspace(xmin, xmax, num_points // 2)
#         waypoints = []
#         for x in xs:
#             waypoints.append([x, ymin])
#             waypoints.append([x, ymax])
#         waypoints = np.array(waypoints)
#         return waypoints
    

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
    # def __init__(self, x_start, x_goal, step_len,
    #              goal_sample_rate, search_radius, iter_max):
        
        
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

        self.x_range = (task_extent[0], task_extent[1])
        self.y_range = (task_extent[2], task_extent[3])
        self.obs_circle = self.env.obs_circle
        self.obs_rectangle = self.env.obs_rectangle
        self.obs_boundary = self.env.obs_boundary
        
        self.goal_radius = 1

    def plan(self):
                
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
        index = self.search_goal_parent()
        self.path, self.path_vertex = self.extract_path(self.vertex[index])
        self.k = k
        self.path_vertex.pop(-1)
        
    def reset_tree(self, start:tuple=None, goal:tuple=None):
        
        if start is not None:
            self.s_start = Node(start)
            
        if goal is not None:
            self.s_goal = Node(goal)
        
        self.path = []
        self.vertex = [self.s_start]
        self.path_vertex = []
        
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

        if len(node_index) > 0:
            cost_list = [dist_list[i] + self.cost(self.vertex[i]) for i in node_index
                            if not self.utils.is_collision(self.vertex[i], self.s_goal)]
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

            #Merge overlapped obstacles
            
        
            self.obs_circle = obs_cir
            self.obs_boundary = obs_bound
            self.obs_rectangle = obs_rec
            self.obstacle = obstacles
            
            self.env.obs_circle = self.obs_circle
            self.env.obs_boundary = self.obs_boundary
            self.env.obs_rectangle = self.obs_rectangle
            self.env.obstacle = obstacles
            
            self.plotting.obs_circle = self.obs_circle
            self.plotting.obs_boundary = self.obs_boundary
            self.plotting.obs_rectangle = self.obs_rectangle
            self.plotting.obstacle = obstacles
            
            self.utils.obs_circle = self.obs_circle
            self.utils.obs_boundary = self.obs_boundary
            self.utils.obs_rectangle = self.obs_rectangle
            self.utils.obstacle = obstacles

