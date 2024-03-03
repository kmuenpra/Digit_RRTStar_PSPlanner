from typing import List

import numpy as np

from . import BasePlanner
from queue import PriorityQueue
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt

class AStarPlanner(BasePlanner):
    
    # Constructor Method
    def __init__(
        self,
        task_extent: List[float],
        rng: np.random.RandomState,
        start: np.ndarray,
        goal: np.ndarray,
        obstacles: np.ndarray,
        step_l_max = 0.3,
        step_l_min = 0.1
        ) -> None:
        
        super().__init__(task_extent, rng)

        # if not isinstance(start, np.ndarray) or not isinstance(goal, np.ndarray) or not isinstance(obstacles, np.ndarray) or not isinstance(grid_bound, tuple):
        #     raise TypeError("start, goal, obstacles must be of type numpy.ndarray. grid_bound must be of type tuple")
        
        #Initialize Objective
        self.start = start
        self.goal = goal
        self.current_heading = 0
        self.obstacles = obstacles
        self.grid_bound = task_extent
        
        #Step Size
        self.dx = 0.25
        self.dy = 0.25
        
        self.step_l_max = step_l_max
        self.step_l_min = step_l_min
        self.max_turn_ang = np.deg2rad(18)
        self.obstacle_radius = 0.3
        
        self.within_goal = 0.1
        
        self.path = []
        
    def get_next_goal(self, robot, num_steps=1, *args, **kwargs) -> np.ndarray:
        
        if not self.path: #If list is empty
            print("AStar.planner.get_next_goal() : Couldn't not find existing trajectory, now generating new one.")
            self.plan(robot)
        
        # print("AStarPlanner.py >> plan() >> self.path =", self.path)
        
        goal = []
        wpx, wpy = robot.wp_c_x, robot.wp_c_y
        
        for _ in range(num_steps):
            wp_n_x, wp_n_y = self.get_next_waypoint((wpx, wpy))
            goal.append([wp_n_x, wp_n_y])
        
        return np.array(goal)
    
    def get_current_waypoint(self, robot, *args, **kwargs) -> np.ndarray:
        
        if not self.path: #If list is empty
            print("AStar.planner.get_current_waypoint() : Couldn't not find existing trajectory, now generating new one.")
            self.plan(robot)
                
        current_goal = self.path[0]
        
        return np.atleast_2d(current_goal)
        
    
    def plan(self, robot=None, *args, **kwargs) -> None:
        
        #NOTE plan the entire trajectory, instead of just one waypoint at a time

        if robot is not None:
            self.start = robot.state[:2]
            self.current_heading = robot.state[2]
                    
        #Clear current plan
        self.path = []
        print("Planning path...")
        
        key_start = tuple(map(np.float64, self.start)) #turn numpy array into hashable tuple 
        key_goal = tuple(map(np.float64, self.goal))
        
        self.open_list = PriorityQueue()
        self.open_list.put((0, key_start))  # Priorityqueue contains tuples (priority, cell)
        self.came_from = {}
        self.cost_so_far = {}
        
        self.came_from[key_start] = None
        self.cost_so_far[key_start] = 0
        
        
        while not self.open_list.empty():
            
            priority_cost, current = self.open_list.get()    
            key_current = tuple(map(np.float64, current))
                
            if self.distance(np.array(current) , self.goal) < self.within_goal:
                print("Reach the goal!")
                # key_current = key_goal
                break
                     
            for next in self.neighbors(current):
                            
                new_cost = self.cost_so_far[key_current] + self.cost(current, next)
                key_next = tuple(map(np.float64, next))
                
                if (key_next not in self.cost_so_far) or (new_cost < self.cost_so_far[key_next]):
                    
                    self.cost_so_far[key_next] = new_cost
                    priority = new_cost + self.heuristic(self.goal, next)
                    self.open_list.put((priority, key_next))
                    self.came_from[key_next] = current
                  
        #Create the path with best heuristic cost
        current = key_current    
        while current[0] - self.start[0] + current[1] - self.start[1] > 0:
            self.path.append(current)
            
            key_current = tuple(map(np.float64, current))
            current = self.came_from[key_current]
            
        self.path.append(key_start)
        self.path.reverse()
        # self.path.append(key_goal)


    def heuristic(self, a, b):
        # Euclidean distance heuristic
        return self.distance(a, b)

    def distance(self, a, b):
        # Euclidean distance
        return np.linalg.norm(a - b)    

    def neighbors(self, cell):
        x, y = cell[0], cell[1]
        
        key_current = (x,y)
        
        if self.came_from[key_current] is None:
            heading_neighbor = self.current_heading
        else:
            previous_point = self.came_from[key_current]
            heading_neighbor = np.arctan2(y - previous_point[1], x - previous_point[0])        
        
        all_neighbors = np.empty((1,2))
        
        #Ensure that the turn is admissable
        for angle in np.linspace(heading_neighbor - np.pi/9, heading_neighbor + np.pi/9, 7):
            
            for step_l in np.linspace(self.step_l_min, self.step_l_max, 3):
                admiss_neighbor = np.array([x + step_l * np.cos(angle), y + step_l * np.sin(angle)])
                all_neighbors = np.vstack((all_neighbors, admiss_neighbor))

        valid_neighbors = []
        
        for neighbor in all_neighbors:
            
            bounded = self.grid_bound[0] <= neighbor[0] < self.grid_bound[1] and self.grid_bound[2] <= neighbor[1] < self.grid_bound[3]
            has_admissable_turn = abs(np.arctan2(neighbor[1] - y, neighbor[0] - x) - heading_neighbor) <= self.max_turn_ang
            
            if self.obstacles.shape[0] == 0:
                has_collision = np.array([False])
            else:
                distances = np.linalg.norm(neighbor - self.obstacles, axis=1)
                has_collision = distances <= self.obstacle_radius 
            
            # future_distances = np.linalg.norm(np.array([neighbor[0] + self.dx, neighbor[1] + self.dy]) - self.obstacles, axis=1)
            # obstacles_ahead = (future_distances <= self.obstacle_radius)
            
            if bounded and has_admissable_turn and (not has_collision.any()):
                valid_neighbors.append(neighbor)
                    
        return valid_neighbors

    def cost(self, a, b):
        # Cost function for moving from cell a to cell b
        # set it to 1 for adjacent cells and sqrt(2) for diagonal
        # return np.linalg.norm(np.array([a[1] - b[1] , a[0] - b[0]]))   
        
        return self.distance(a, b) * 0.9

    def add_obstacle(self, new_obstacles):
        
        if not isinstance(new_obstacles, np.ndarray) or new_obstacles.shape[1] != 2:
            raise("obstacles position must be of type numpy.ndarray and size of (n,22)")
        
        if self.obstacles.shape[0] == 0:
            self.obstacles = new_obstacles
        else:
            self.obstacles = np.vstack([self.obstacles, new_obstacles])
        
    def plot_map(self):
        # Plot obstacles
        for obstacle in self.obstacles:
            plt.scatter(obstacle[0], obstacle[1], color='red', marker='s')

        # Plot start and goal positions
        plt.scatter(self.start[0], self.start[1], color='green', marker='o', label='Start')
        plt.scatter(self.goal[0], self.goal[1], color='blue', marker='o', label='Goal')

        # Plot path
        if self.path:
            path_x = [cell[0] for cell in self.path]
            path_y = [cell[1] for cell in self.path]
            plt.plot(path_x, path_y, color='cyan', linewidth=2, label='Waypoints')
            plt.scatter(path_x, path_y, color='green', marker='o')

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('AStar Trajectory Map')
        plt.legend()
        plt.grid(True)
        plt.show()
        
    #Dont use this
    def get_action(self):
        
        self.action_list = []
        
        if self.path:
            for i in np.arange(0, len(self.path) - 1, 1):
                step_length = np.linalg.norm(np.array(self.path[i+1]) - np.array(self.path[i]))
                dheading = np.arctan2(self.path[i+1][1] - self.path[i][1], self.path[i+1][0] - self.path[i][0])
                self.action_list.append((step_length, dheading))
                
    def get_next_waypoint(self, pose):
        
        if not isinstance(pose, np.ndarray) and not isinstance(pose, tuple):
            raise("Current waypoint must be of type numpy.ndarray or tuple")
        
        if not isinstance(pose, tuple):
            pose = tuple(pose)
            
        if len(pose) != 2:
            raise("Current waypoint must be of size 2")
        
        if pose not in self.path:
            raise("Could not find current waypoint in the trajectory.") 
        
        return self.path[self.path.index(pose) + 1]
