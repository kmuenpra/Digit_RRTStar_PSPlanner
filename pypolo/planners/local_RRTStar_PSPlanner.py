import numpy as np
from typing import List
from . import BasePlanner

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + "/pypolo/planners/RRTStar_package")
print(os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + "/pypolo/planners/RRTStar_package")


import math
import numpy as np
import env
import plotting
import utils
import queue


# class LocalRRTStar(BasePlanner):
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
    



class Node:
    def __init__(self, n):
        self.x = n[0]
        self.y = n[1]
        self.parent = None
        
    def print_node_chain(self, node=None, counter=0, show_parents=True):
        
        if node is None:
            node = self
                
        print("Node " + str(counter) + ": ")
        print("x: ", node.x)
        print("y: ", node.y)
        
        if node.parent is None:
            print("parent: None")  
            print("---------------------------------")
        else:
            print("parent: (" + str(node.parent.x) + ", " + str(node.parent.y) + ")")
        print("---------------------------------")
        
        if show_parents:
            counter += 1
            if node.parent is not None:
                node.print_node_chain(node=node.parent, counter=counter)


class LocalRRTStar(BasePlanner):
    
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
        max_turn_angle = np.deg2rad(14)
        ) -> None:
        
        #Initialize BasePlanner
        super().__init__(task_extent, rng)
        
        self.s_start = Node(x_start) #start
        self.s_goal = Node(x_goal) #goal
        self.goal_sample_rate = goal_sample_rate #chances (0~1) of sampling the goal during tree branching
        self.search_radius = search_radius #radius to find nearest neighnbor
        self.iter_max = iter_max
        self.vertex = [self.s_start]
        self.path = []
        self.path_vertex = []

        self.plotting = plotting.Plotting(x_start, x_goal)
        self.utils = utils.Utils()


        #------------------ Set up environment -----------------
        
        self.env = env.Env()
        
        self.margin = 1
        
        # Find minimum and maximum x coordinates with margin
        min_x = min( self.s_start.x, self.s_goal.x ) - self.margin
        max_x = max( self.s_start.x, self.s_goal.x ) + self.margin
        
        # Find minimum and maximum y coordinates with margin
        min_y = min( self.s_start.y, self.s_goal.y ) - self.margin
        max_y = max( self.s_start.y, self.s_goal.y ) + self.margin
        
        # Define bounding box ranges
        self.x_range = (min_x, max_x)
        self.y_range = (min_y, max_y)
        
        # self.x_range = self.env.x_range #(0, 50)
        # self.y_range = self.env.y_range #(0, 30)
        self.obs_circle = self.env.obs_circle
        self.obs_rectangle = self.env.obs_rectangle
        self.obs_boundary = self.env.obs_boundary
        
        #-----------------------------------------------------
        
        #For Digit Constraints:
        self.max_turn_ang = max_turn_angle
        self.step_len_max = 0.3
        self.step_len_min = 0.1
        self.step_len = min(step_len, self.step_len_max) #step 
        self.goal_radius = self.step_len_min
        
        #-----------------------------------------------------

    def plan(self, heading_c=0, obstacle_margin=None):
        '''
        TODO
        self.utils.is_collision()
            - implement a way to add new obstacles into the map
            - check for high uncertainty in the terrain / infeasible height change
        '''
        
        if obstacle_margin is None:
            obstacle_margin = self.utils.delta
            
        temp = self.max_turn_ang
        
        for k in range(self.iter_max):
            
            if k < 25:
                self.max_turn_ang = np.deg2rad(5)
            else:
                self.max_turn_ang = temp

            node_rand = self.generate_random_node(self.goal_sample_rate)
            node_near = self.nearest_neighbor(self.vertex, node_rand) #find the closest tree node to the sampled point
            # node_new = self.new_state(node_near, node_rand) #move the nearest tree node in the direction toward the sample point, but limited to the maximum step_len 
                                                            #(the nearest tree node will be the parent of the new node)
            admissible_node_list = self.branch_all_feasible_nodes(node_near, heading_default=heading_c)
            
            if k % 500 == 0:
                print(k)
            
            for node_new in admissible_node_list:
                
                #Check if path from nearet tree node to the new node is safe
                if node_new and not self.utils.is_collision(node_near, node_new, delta=obstacle_margin):
                    # neighbor_index = self.find_near_neighbor(node_new) #find other neighbor node near the new node
                    self.vertex.append(node_new) #append new node to the vertex

                    # if neighbor_index:
                    #     self.choose_parent(node_new, neighbor_index) #reassign the parent of the new node, if there are closer neighbor
                    #     self.rewire(node_new, neighbor_index) #if new node is closer to the start than neighbor, make the new node a parent of that neighbor
                        
                #if reach within 2m radius of the goal position, end the tree search
                if math.hypot(node_new.x - self.s_goal.x, node_new.y - self.s_goal.y) <= self.goal_radius:
                    break
                
            else:
                continue
            break #break outer for-loop when within goal radius
                    
        index = self.search_goal_parent()
        self.path, self.path_vertex = self.extract_path(self.vertex[index])
        self.k = k
        self.path_vertex.pop(-1)
        self.get_interpolant_param() #generate self.m_list and self.c_list
                
        #Check to if any nodes in optimal path can be rewired together
        for i in np.flip(np.arange(0, len(self.path_vertex) - 1)):
            self.rewire_path(s_start_index=i)
            
    
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
            self.get_interpolant_param() #generate self.m_list and self.c_list
            
        
        if i == s_start_index or i == len(self.path_vertex) - 2:
            return
        else:
            self.rewire_path(s_start_index=i)
        
    def replan(self, s_start_new, heading_c=0, recursive_count = 0, obstacle_margin=None):
        '''
        Search for closest nodes in the existing path to merge with the new starting point
        
        TODO: New strategy -> find the nearest node in the path, find the heading of the next node, plan from new start node to that node
        '''
        
        if recursive_count > 100:
            raise MemoryError ("Replanning Failed. Too much iterations.")
        recursive_count += 1
        
        if not isinstance(s_start_new, Node):
            if isinstance(s_start_new, tuple):
                s_start_new = Node(s_start_new) 
                self.vertex.append(s_start_new)  
            else:
                raise TypeError ("Value must be an instance of tuple or Node class")
        
        if obstacle_margin is None:
            obstacle_margin = self.utils.delta
        
        if self.utils.is_inside_obs(s_start_new, delta=obstacle_margin):
            if recursive_count < 2:
                raise ValueError ("Cannot start at a node where there is an obstacle.")
        
        #Get index of the starting node     
        s_start_index = self.vertex.index(s_start_new)
        self.vertex[s_start_index].print_node_chain(show_parents=False) #Print the node
        
        if math.hypot(self.vertex[s_start_index].x - self.s_goal.x, self.vertex[s_start_index].y - self.s_goal.y) <= self.goal_radius:
            #Save replanned path
            index = self.search_goal_parent()
            self.path, self.path_vertex = self.extract_path(self.vertex[index])
            self.path_vertex.pop(-1)
            return
        
        
        #--------------- Quick Check: drawing straight line to any node in the optimal path -----------------------
        
        # for path_node in self.path_vertex:
        #     dist, heading_n = self.get_distance_and_angle(self.vertex[s_start_index], path_node)
        #     num_steps = int(dist/self.step_len)
        #     num_turns = int((heading_n - heading_c)/self.max_turn_ang)
            
        #     if not self.utils.is_collision(self.vertex[s_start_index], path_node) and abs(heading_n - heading_c):
        #         if dist <= self.step_len:
        #             path_node.parent = self.vertex[s_start_index]
        #         else:
        #             self.add_interpolated_parents(self.vertex[s_start_index], path_node)
                
        #         #Save replanned path
        #         index_goal = self.search_goal_parent()
        #         self.path, _ = self.extract_path(self.vertex[index_goal])
        #         return
                
            
        # print("Quick Planning Failed, try to rebranch the start node.")
        
        #---------------- Branch out the start node and continue ----------------------------
        # lcv = 0
        # while lcv < 300:
            
        #     self.vertex[s_start_index].print_node_chain(show_parents=False) #Print to check
            
        #     #initialize cost
        #     minimum_cost = 100
                        
        #     #all feasible nodes branched out from the new starting point
        #     admissible_node_list = self.branch_all_feasible_nodes(self.vertex[s_start_index])      
            
        #     for node_new in admissible_node_list:
        #         #Check for collision
        #         if node_new and not self.utils.is_collision(self.vertex[s_start_index], node_new):
                    
        #             #First step replanning
        #             _ , heading_n  = self.get_distance_and_angle(self.vertex[s_start_index], node_new)
                    
        #             #find other neighbor node near the new node
        #             neighbor_index = self.find_near_neighbor(node_new) 
        #             self.vertex.append(node_new) #append new node to the vertex
                    
        #             for index in neighbor_index:
                        
        #                 #Second step replanning
        #                 dist , heading_nn  = self.get_distance_and_angle(node_new, self.vertex[index])
                        
        #                 #if valid nodes found, set it as the child of new starting point
        #                 if abs(heading_nn - heading_n) <= self.max_turn_ang:
                            
        #                     #is the neightbor of the new node a part of optimal trajectory?
        #                     if self.is_near_optimal_path(index, heading_nn):
                                
        #                         if dist <= self.step_len:
        #                             self.vertex[index].parent = node_new
        #                         else:
        #                             self.add_interpolated_parents(node_new, self.vertex[index])
                                
        #                         #Save replanned path
        #                         index_goal = self.search_goal_parent()
        #                         self.path, _ = self.extract_path(self.vertex[index_goal])
        #                         return
                                                        
        #             #check cost of the new node to see how close it is to the optimal path
        #             cost = [self.get_new_cost(node_new, path_node, heading_default=heading_n) for path_node in self.path_vertex]
        #             if min(cost) < minimum_cost:
        #                 #Set the lowest cost node as the new staring point
        #                 minimum_cost = min(cost)
        #                 s_start_index = len(self.vertex) - 1
                        
        #     lcv += 1
            
        # print("Replanning Failed.")
            
        # #Worst Case: replan the entire tree branch
        # self.path = []
        # self.path_vertex = []
        # self.vertex = [s_start_new]
        # self.plan(heading_c=heading_c)
        # return
                        
                    
                    
        
        #--------------------------------------------

        #Comparing the cost of getting from starting node to the current optimal path
        cost = [self.get_new_cost(self.vertex[s_start_index], path_node, heading_default=heading_c) for path_node in self.path_vertex]
        # cost.append(self.get_new_cost(self.vertex[s_start_index], self.s_goal, heading_default=heading_c))
        
        low_cost_count = sum(1 for c in cost if c < 50) #High cost means it violates maximum turning angle
        
        #Sort for lowest cost that only has admissable turning angle
        sorted_cost_indices = sorted(range(len(cost)), key=lambda i: cost[i])
        cost_min_indices = sorted_cost_indices[:low_cost_count]
        
        for cost_min_index in cost_min_indices: 
            
            #Two step planning, make sure turning angles are valid for two forward steps
            dist , heading_n  = self.get_distance_and_angle(self.vertex[s_start_index], self.path_vertex[cost_min_index])
            
            if cost_min_index < len(self.path_vertex) - 1:
                _ , heading_nn  = self.get_distance_and_angle(self.path_vertex[cost_min_index], self.path_vertex[cost_min_index + 1])
            else:
                heading_nn = heading_n
            
            if (abs(heading_n - heading_c) <= self.max_turn_ang) and (abs(heading_nn - heading_n) <= self.max_turn_ang) and \
                not self.utils.is_collision(self.vertex[s_start_index], self.path_vertex[cost_min_index], delta=obstacle_margin):
                
                #if valid nodes found, set it as the child of new starting point
                if dist <= self.step_len:
                    self.path_vertex[cost_min_index].parent = self.vertex[s_start_index]
                else:
                    self.add_interpolated_parents(self.vertex[s_start_index], self.path_vertex[cost_min_index])
                
                #Save replanned path
                index = self.search_goal_parent()
                self.path, self.path_vertex = self.extract_path(self.vertex[index])
                self.path_vertex.pop(-1)
                self.get_interpolant_param() #generate self.m_list and self.c_list
                return
        
        #--------------------------------------------
        
        print("Check if the start node overlaps the path")
        
        #Check for smallest distance to path, in order to determine the step length
        nearest_node_index = int(np.argmin([math.hypot(nd.x - self.vertex[s_start_index].x, nd.y - self.vertex[s_start_index].y) for nd in self.path_vertex]))
        
        step_l , angle = self.find_intercept_path(self.vertex[s_start_index], heading_c, nearest_node_index)
        
        if step_l ==  0 and angle == 0:
            self.path_vertex[nearest_node_index + 1].parent = self.vertex[s_start_index]
            
            #Save replanned path
            index = self.search_goal_parent()
            self.path, self.path_vertex = self.extract_path(self.vertex[index])
            self.path_vertex.pop(-1)
            self.get_interpolant_param() #generate self.m_list and self.c_list
            return
            
        elif step_l is not None:
            node_new = Node((self.vertex[s_start_index].x + step_l * np.cos(angle), self.vertex[s_start_index].y + step_l * np.sin(angle)))
            node_new.parent = self.vertex[s_start_index]
            self.vertex.append(node_new)
            self.path_vertex[nearest_node_index + 1].parent = node_new
            
            # print("node new chain")
            # node_new.print_node_chain()
                
            #Save replanned path
            index = self.search_goal_parent()
            self.path, self.path_vertex = self.extract_path(self.vertex[index])
            self.path_vertex.pop(-1)
            self.get_interpolant_param() #generate self.m_list and self.c_list
            return
            
        #--------------------------------------------
            
        print("Branching out the start node!")
                
        admissible_node_list = self.branch_all_feasible_nodes(self.vertex[s_start_index], heading_default=heading_c)
        minimum_cost = 100
        closest_node_index = s_start_index
        closest_node_heading = heading_c
        
        for node_new in admissible_node_list:
            if not self.utils.is_collision(self.vertex[s_start_index], node_new, delta=obstacle_margin):
                _ , heading_node_new  = self.get_distance_and_angle(self.vertex[s_start_index], node_new)
                self.vertex.append(node_new)
                
                #branch out the start node, and find the nearest one to the optimal path
                cost = [self.get_new_cost(node_new, path_node, heading_default=heading_node_new) for path_node in self.path_vertex]
                                
                if min(cost) < minimum_cost:
                    #Set the lowest cost node as the new staring point
                    minimum_cost = min(cost)
                    closest_node_index = len(self.vertex) - 1 #last index, since it is recently appended to the tree
                    closest_node_heading = heading_node_new
                
        #Keep replanning with new node that is nearest to the optimal path
        self.replan(self.vertex[closest_node_index], heading_c=closest_node_heading, recursive_count=recursive_count)
    
    
    
    def find_intercept_path(self, current_node, heading_c, nearest_node_index, eps = 0.015):
        
        if nearest_node_index  < len(self.path_vertex) - 1:
            m1 = self.m_list[nearest_node_index]
            c1 = self.c_list[nearest_node_index]
            
            
            _, heading_n = self.get_distance_and_angle(self.path_vertex[nearest_node_index], self.path_vertex[nearest_node_index + 1])
            if abs(m1*current_node.x + c1 - current_node.y) < eps and abs(heading_n - heading_c) <= self.max_turn_ang:
                print("new node is right on the path")
                return 0, 0
            
            for angle in np.linspace(heading_c - self.max_turn_ang, heading_c + self.max_turn_ang, 7):
                for step_l in np.linspace(self.step_len_min, self.step_len_max, 5):
                    
                    #branch current node
                    branch_node = (current_node.x + step_l * np.cos(angle), current_node.y + step_l * np.sin(angle))
                    
                    if abs(m1*branch_node[0] + c1 - branch_node[1]) < eps and abs(heading_n - angle) <= self.max_turn_ang:
                        print("new node can branch out and join the path")
                        return step_l, angle
                          
        return None, None
                    
        
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
        
    def get_interpolant_param(self, eps=0.00001):
    
        path = np.array(self.path)
        x = path[:len(path) - 1,0]
        y = path[:len(path) - 1,1]

        self.m_list = np.zeros(len(x)-1)
        self.c_list = np.zeros(len(x)-1)

        for i in range(len(x)-1):
            
            den = (x[i+1]-x[i])
            
            if den == 0:
                den = eps
            
            m = (y[i+1]-y[i])/den
            c1 = (y[i]-m*x[i])
            c2 = (y[i+1]-m*x[i+1])

            # print("For (%.1f, %.1f) and (%.1f, %.1f), m = %.2f and c1 = %.2f, c2 = %.2f."
            #          % (x[i], y[i], x[i+1], y[i+1], m, c1, c2))
            
            self.m_list[i] = m
            self.c_list[i] = c1
        

                        #node_near, node_rand
    def new_state(self, node_start, node_goal):
        dist, theta = self.get_distance_and_angle(node_start, node_goal)
        
        #Define current heading
        if self.is_start_node(node_start) or node_start.parent is None:
            heading_c = 0
        else:
            _, heading_c = self.get_distance_and_angle(node_start.parent, node_start)
            
        #Check if the heading chnage is feasible
        if abs(theta - heading_c) > self.max_turn_ang:
            return None

        #Else branch out the tree node that has feasible turning angle
        dist = min(self.step_len, dist)
        node_new = Node((node_start.x + dist * math.cos(theta),
                         node_start.y + dist * math.sin(theta)))

        node_new.parent = node_start

        return node_new
    
    def branch_all_feasible_nodes(self, node_start, heading_default=0):
        '''
        TODO:
        Check if previous dheading was a positive or negative
            - if last dheading was positive, probably dont want to turn negative with large magnitude. 
            Otherwise, Digit would sway too much.
        '''
        
        #Define current position
        x = node_start.x
        y = node_start.y
        
        #Define current heading
        if self.is_start_node(node_start) or node_start.parent is None:
            heading_c = heading_default
        else:
            _, heading_c = self.get_distance_and_angle(node_start.parent, node_start)
            
        #Branch the tree node_start to all possible feasible nodes        
        valid_neighbors = []

        for angle in np.linspace(heading_c - self.max_turn_ang, heading_c + self.max_turn_ang, 7):
            # for step_l in np.linspace(self.step_l_min, self.step_l_max, 3):
            admiss_neighbor = Node((x + self.step_len * np.cos(angle), y + self.step_len * np.sin(angle)))
            admiss_neighbor.parent = node_start
            valid_neighbors.append(admiss_neighbor)

        return valid_neighbors

    def choose_parent(self, node_new, neighbor_index):
        cost = [self.get_new_cost(self.vertex[i], node_new) for i in neighbor_index]
        
        cost_min_index = neighbor_index[int(np.argmin(cost))]
        node_new.parent = self.vertex[cost_min_index]

    def rewire(self, node_new, neighbor_index):
        for i in neighbor_index:
            node_neighbor = self.vertex[i]

            #Define current heading
            if self.is_start_node(node_new) or node_new.parent is None:
                heading_c = 0
            else:
                _, heading_c = self.get_distance_and_angle(node_new.parent, node_new)
            
            #Define new heading if node_new was the parent of the neighbor 
            _, new_heading = self.get_distance_and_angle(node_new, node_neighbor)
            
             #if the cost of the getting to the neighbor > the cost of getting to the new node then to the neighbor,
            #reassign the new node as the parent of the neighbor
            if self.cost(node_neighbor) > self.get_new_cost(node_new, node_neighbor) and (abs(new_heading - heading_c) <= self.max_turn_ang):
                node_neighbor.parent = node_new

    def search_goal_parent(self):
        dist_list = [math.hypot(n.x - self.s_goal.x, n.y - self.s_goal.y) for n in self.vertex]
        node_index = [i for i in range(len(dist_list)) if dist_list[i] <= self.goal_radius] #<= self.step_len

        if len(node_index) > 0:
            cost_list = [dist_list[i] + self.cost(self.vertex[i]) for i in node_index
                            if not self.utils.is_collision(self.vertex[i], self.s_goal)]
            return node_index[int(np.argmin(cost_list))]

        return len(self.vertex) - 1

    def get_new_cost(self, node_start, node_end, heading_default = 0):
        
        #original cost
        dist, heading_n = self.get_distance_and_angle(node_start, node_end)
        
        #Define current heading
        if self.is_start_node(node_start) or node_start.parent is None:
            heading_c = heading_default
        else:
            _, heading_c = self.get_distance_and_angle(node_start.parent, node_start)
        
        #Increase cost for violating the turning angle
        if abs(heading_n - heading_c) > self.max_turn_ang:
            turn_penalty = 50
        else:
            turn_penalty = 0

        return self.cost(node_start) + dist + turn_penalty

    def generate_random_node(self, goal_sample_rate):
        delta = self.utils.delta #0.5 as of now

        if np.random.random() > goal_sample_rate: #10% chance of not generating any new nodes
            return Node((np.random.uniform(self.x_range[0] + delta, self.x_range[1] - delta),
                         np.random.uniform(self.y_range[0] + delta, self.y_range[1] - delta)))

        return self.s_goal #return goal instead

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
    
    def is_start_node(self, node):
        #Check if the node is the start
        return (node.x == self.s_start.x) and (node.y == self.s_start.x)
    
    
    def plot_map(self):
        
        if self.path:
            self.plotting.animation(self.vertex, self.path, "rrt*, total Iter. = " + str(self.k), )
        else:
            self.plotting.plot_grid("Plain Environment")
            

