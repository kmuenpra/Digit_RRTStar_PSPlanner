import numpy as np
import random

from . import BaseSensor


class LidarSensor(BaseSensor):

    # def sense(self, states: np.ndarray) -> np.ndarray:
    #     if states.ndim == 1:
    #         states = states.reshape(1, -1)
    #     noise_free = self.get(states[:, 0], states[:, 1])
    #     observations = self.rng.normal(loc=noise_free, scale=self.noise_scale)
    #     return observations.reshape(-1, 1)
    
    def __init__(self, matrix, env_extent, rate, noise_scale, rng, max_distance, perception_angle):
        super().__init__(matrix=matrix, env_extent=env_extent, rate=rate, noise_scale=noise_scale, rng=rng)
        self.max_distance = max_distance
        self.perception_angle = perception_angle
        
    
    def sense(self, states: np.ndarray, global_heading, ray_tracing = False, num_targets = 20) -> np.ndarray:
        # noise_free = set()  # Using a set to avoid repeated samples
        
        states = np.atleast_2d(states)
        
        if ray_tracing: #measure point forward
            sensor_angle = np.random.uniform(-self.perception_angle/2.0, self.perception_angle/2.0, num_targets)
            sensor_range = np.random.uniform(0 ,self.max_distance, num_targets)
            point_x_body = sensor_range * np.cos(np.radians(sensor_angle))
            point_y_body = sensor_range * np.sin(np.radians(sensor_angle))    
            #convert body to global frame
            point_x = np.cos(global_heading) * point_x_body - np.sin(global_heading) * point_y_body + states[:,0]
            point_y = np.sin(global_heading) * point_x_body + np.cos(global_heading) * point_y_body + states[:,1]
    
        else: #measure all around the robot 
            point_x = np.random.uniform(states[:,0]-self.max_distance , states[:,0]+self.max_distance, num_targets)
            point_y = np.random.uniform(states[:,1]-self.max_distance , states[:,1]+self.max_distance, num_targets)
             
        
        #Make sure it is within bound
        index_to_delete = []
        for i in range(num_targets):
            x, y = point_x[i], point_y[i]
            if (x < self.env_extent[0] or x > self.env_extent[1]) or (y < self.env_extent[2] or y > self.env_extent[3]):
                index_to_delete.append(i)
                
        point_x = np.delete(point_x, index_to_delete)
        point_y = np.delete(point_y, index_to_delete)
                
        locations = np.vstack([point_x[:], point_y[:]]).T
            
        noise_free = self.get(locations[:,0], locations[:,1])
        observations = self.rng.normal(loc=np.array(list(noise_free)), scale=self.noise_scale)
        
        return locations, observations.reshape(-1, 1)
