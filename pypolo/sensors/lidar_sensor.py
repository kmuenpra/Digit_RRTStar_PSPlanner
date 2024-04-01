import numpy as np
import random
from scipy.stats import norm

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
        
    
    def sense(self, model, states: np.ndarray, global_heading, ray_tracing = False, num_targets = 20, candidate_point=None) -> np.ndarray:
        # noise_free = set()  # Using a set to avoid repeated samples
        
        replanning_local = False
        
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
        observations = self.rng.normal(loc=np.array(list(noise_free)), scale=self.noise_scale).reshape(-1, 1)
        
        # z_pred, z_std = model.predict(locations)
        # z_quantile = self.find_quantile(value=observations, mean_pred=z_pred, std_pred=z_std)
        
        # if len(np.where(z_quantile < 0.05)[0]) > 0:
        #         replanning_local = True
        
        # # Check for prediction error for the next two point
        # if candidate_point is not None:
        #     candidate_point = np.atleast_2d(candidate_point)
        #     z_measure = self.get(candidate_point[:, 0], candidate_point[:, 1]).reshape(-1, 1)
        #     z_pred, z_std = model.predict(candidate_point)
        #     z_quantile = self.find_quantile(value=z_measure, mean_pred=z_pred, std_pred=z_std)    
            
        #     print("LIDAR sensor")
        #     print("measure", z_measure)
        #     print("pred", z_pred)
        #     print("quantile", z_quantile)
            
        #     if len(np.where(z_quantile < 0.1)[0]) > 0:
                
        #         replanning_local = True
                
            # print(locations)
            # print(observations)

            # locations = np.vstack([locations, candidate_point])
            # observations = np.vstack([observations, z_measure])
                
        
        return locations, observations, replanning_local


    def find_quantile(self, value, mean_pred, std_pred):
        
        
        # Calculate the standardized value
        z_score = (value - mean_pred) / std_pred
        
        # Calculate the cumulative probability
        cumulative_prob = norm.cdf(z_score)
        
        # Convert cumulative probability to quantile
        quantile = 1 - cumulative_prob  # 1 - CDF gives the tail probability
        
        return quantile