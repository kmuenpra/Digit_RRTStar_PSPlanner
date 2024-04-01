import csv
import numpy as np
import pickle
# https://docs.jaxgaussianprocesses.com/examples/collapsed_vi/
# Enable Float64 for more stable matrix inversions.
from jax import config
import os
import math
#os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
#os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".1"
#os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"
config.update("jax_enable_x64", True)

from dataclasses import dataclass
from typing import Dict

from jax import jit
import jax.numpy as jnp
import jax.random as jr
import jax.numpy as jnp
from jaxtyping import (
    Array,
    Float,
    install_import_hook,
)
import matplotlib.pyplot as plt
import numpy as np
from simple_pytree import static_field
import tensorflow_probability.substrates.jax as tfp
import optax as ox

with install_import_hook("gpjax", "beartype.beartype"):
    import gpjax as gpx
    from gpjax.base.param import param_field

#from docs.examples.utils import clean_legend

key = jr.PRNGKey(123)
# tfb = tfp.bijectors
# plt.style.use(
#     "https://raw.githubusercontent.com/JaxGaussianProcesses/GPJax/main/docs/examples/gpjax.mplstyle"
# )
# cols = plt.rcParams["axes.prop_cycle"].by_key()["color"]


class PathEvaluator:
    def __init__(self, filepath):
        
        print("initialize Path Evaluator.")
        
        # XLeft  = filepath_modelErrorX[0]
        # XRight = filepath_modelErrorX[1]
        # YLeft  = filepath_modelErrorY[0]
        # YRight = filepath_modelErrorY[1]
        
        # # Load data from pickle files
        with open(filepath, 'rb') as f1:
            D, opt_posterior = pickle.load(f1)
            
        # with open(XRight, 'rb') as f2:
        #     D_x_right, opt_posterior_x_right = pickle.load(f2)
                        
        # with open(YLeft, 'rb') as f3:
        #     D_y_left, opt_posterior_y_left = pickle.load(f3)
            
        # with open(YRight, 'rb') as f4:
        #     D_y_right, opt_posterior_y_right = pickle.load(f4)
            
        # #--------------------------------------------------

        self.opt_posterior = opt_posterior
        self.D = D
        
        # self.opt_posterior_x_right = opt_posterior_x_right
        # self.D_x_right = D_x_right
        
        # self.opt_posterior_y_left = opt_posterior_y_left
        # self.D_y_left = D_y_left
        
        # self.opt_posterior_y_right = opt_posterior_y_right
        # self.D_y_right = D_y_right

    def get_psp_actions(self, model, path,  heading_c=0, current_stance=0):
        
        actions = []
        # stances = []
        
        prev_x, prev_y = path[0]  # Initial (x, y) point
        prev_heading =  heading_c  # Initial heading angle
        stance = current_stance
        
        #Predict height at current point
        z_prev, _ = model.predict([[prev_x, prev_y]]) 
        z_prev = z_prev.ravel()
        
        for point in path[1:]:
            x, y = point
            
            #Predict height at next waypoint
            z, _ = model.predict([[x, y]]) 
            z = z.ravel()

            # Calculate distance between two adjacent points (x and y only)
            step_l = math.sqrt((x - prev_x)**2 + (y - prev_y)**2)

            # Calculate change in heading angle between two adjacent points (x-y only)
            heading = math.atan2(y - prev_y, x - prev_x)
            dheading = heading - prev_heading
            dheading = (dheading + math.pi) % (2 * math.pi) - math.pi # Ensure dheading is between -pi and pi

            # Change in z
            dz = z - z_prev  # Change in z relative to the previous point

            actions.append([step_l, np.rad2deg(dheading), float(dz), stance])
            # stances.append(stance)

            # Update previous values for next iteration
            prev_x, prev_y, prev_heading, z_prev = x, y, heading, z
            stance = 0 if stance else 1

        return actions
    
    def calculate_entropy(self, model, path, *args, **kwargs) -> float:
        assert model is not None, "Model must be provided."
        # entropies = []
        
        path = np.atleast_2d(path)
        
        _, std = model.predict(path)
        
        entropies = 0.5 * np.log(2 * np.pi * np.square(std)) + 0.5
        
        # for point in path:
        #     x, y = point
        #     _, std = model.predict([[x, y]])
        #     std = std.ravel()
        #     entropy = 0.5 * np.log(2 * np.pi * np.square(std)) + 0.5
        #     entropies.append(entropy)
        return sum(entropies)
    
    def predict_model_uncertainty(self, model, path,  heading_c=0, current_stance=0):
        
        actions = jnp.array(self.get_psp_actions(model, path,  heading_c=heading_c, current_stance=current_stance))
        
        latent_dist          =  self.opt_posterior(actions, train_data=self.D)
        predictive_dist      =  self.opt_posterior.posterior.likelihood(latent_dist)
        predicted_error      =  predictive_dist.mean()
        
        dx_global = -np.sin(heading_c) * predicted_error
        dy_global = np.cos(heading_c) * predicted_error
        
        change_in_deviation = np.hstack([dx_global.reshape(-1,1), dy_global.reshape(-1,1)])
        print("change dev", change_in_deviation)
        print("Sum", np.sum(change_in_deviation, axis=0))
        
        return predicted_error
        
        # A_left, A_right = split_array_based_on_binary(actions, stances)
        # A_left = jnp.array(A_left)
        # A_right = jnp.array(A_right)
        
        # print("Actions (Left): ", A_left.shape)
        # print(A_left)
        
        # print("Actions (Right): ", A_right.shape)
        # print(A_right)
    
        # if len(A_left) > 0:
        #     latent_dist_x          =  self.opt_posterior_x_left(A_left, train_data=self.D_x_left)
        #     predictive_dist_x      =  self.opt_posterior_x_left.posterior.likelihood(latent_dist_x)
        #     predicted_error_x_left =  predictive_dist_x.mean()
            
        #     latent_dist_y          = self.opt_posterior_y_left(A_left, train_data=self.D_y_left)
        #     predictive_dist_y      = self.opt_posterior_y_left.posterior.likelihood(latent_dist_y)
        #     predicted_error_y_left =  predictive_dist_y.mean()
            
        #     predicted_error_x_all = predicted_error_x_left
        #     predicted_error_y_all = predicted_error_y_left
            
        #      #--------------------------------------------------------------
        
        # if len(A_right) > 0:
        #     latent_dist_x          =  self.opt_posterior_x_right(A_right, train_data=self.D_x_right)
        #     predictive_dist_x      =  self.opt_posterior_x_right.posterior.likelihood(latent_dist_x)
        #     predicted_error_x_right=  predictive_dist_x.mean()
            
        #     latent_dist_y          =    self.opt_posterior_y_right(A_right, train_data=self.D_y_right)
        #     predictive_dist_y      = self.opt_posterior_y_right.posterior.likelihood(latent_dist_y)
        #     predicted_error_y_right=  predictive_dist_y.mean()
            
        #     predicted_error_x_all = predicted_error_x_right
        #     predicted_error_y_all = predicted_error_y_right
        
        # # print("Predicted shape")
        # # print(predicted_error_x_left.shape)
        # # print(predicted_error_x_right.shape)
            
        
        # if len(A_left) > 0 and len(A_right) >  0:
        #     predicted_error_x_all = np.append(predicted_error_x_left,predicted_error_x_right)
        #     predicted_error_y_all = np.append(predicted_error_y_left,predicted_error_y_right)
            
        # if len(A_left) > 0 or len(A_right) >  0:
        #     predicted_error_norm = np.linalg.norm(np.vstack([ [predicted_error_x_all], [predicted_error_y_all]]).T, axis=1)
        #     return float(np.sum(predicted_error_norm))
        
        # return 0

            # psp_error = []
            # for point in points:
            #     x, y, _ = point
            #     # TODO: make sure model return std rather than var
            #     _, std = model.predict([[x, y]])
            #     std = std.ravel()
            #     uncertainties.append(std)
            # return uncertainties

    def evaluate_path(self, trajectory, model, heading_c = 0, current_stance = 0, *args, **kwargs) -> float:
        
        # Call plan method to calculate entropy of each point in the trajectory
        entropies = self.calculate_entropy(model, trajectory)

        # Call predict_model_uncertainty method to get uncertainties of each point in the trajectory
        uncertainties = self.predict_model_uncertainty(model, trajectory,  heading_c=heading_c, current_stance=current_stance)

            
        return entropies, uncertainties
    
def generate_stance(size):
    '''
    Generate 0,1,0,... in order
    '''
    alternating_array = []
    for i in range(size):
        alternating_array.append(i % 2)
    return np.array(alternating_array).reshape(-1, 1)

def split_array_based_on_binary(data_array, binary_array):
    # Initialize two lists to store elements based on binary values
    array_with_1 = []
    array_with_0 = []

    # Iterate through both arrays simultaneously
    for data, binary_value in zip(data_array, binary_array):
        if binary_value == 1:
            array_with_1.append(data)
        else:
            array_with_0.append(data)

    return array_with_0, array_with_1
