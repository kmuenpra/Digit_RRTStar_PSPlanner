from time import time
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
import pypolo
import gpytorch
from pypolo.scalers import MinMaxScaler, StandardScaler
from PIL import Image
import os
import datetime
import math
import random
import traceback


def set_random_seed(seed):
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    print(f"Set random seed to {seed} in numpy and torch.")
    return rng


def get_environment(cfg, filepath="./data/n44w111.npz"):
    with np.load(filepath) as data:
        env = data["arr_0"]
    print(f"Loaded environment of shape {env.shape}.")
    return env


def get_environment_from_image(cfg, png_path, resize=None, max_height=None):
    
    image = Image.open(png_path)
    image = image.transpose(Image.FLIP_TOP_BOTTOM)

    if resize is not None:
        if not isinstance(resize, tuple):
            raise TypeError
        else:
            resized_image = image.resize(resize, Image.BOX)
            image = resized_image.convert('L') 
    
    env = np.array(image)
    
    if max_height is not None:
        
        if not isinstance(max_height, float):
            raise TypeError
        else:
            min_value = np.min(env)
            max_value = np.max(env)
            normalized_matrix = (env - min_value) / (max_value - min_value)
            env = normalized_matrix * max_height
            
    print(f"Loaded environment of shape {env.shape}.")
    return env


def get_sensor(cfg, env, rng, scan_radius=4):
    
    sensor = pypolo.sensors.LidarSensor(
        matrix=env,
        env_extent=cfg.env_extent,
        rate=cfg.sensing_rate,
        noise_scale=cfg.noise_scale,
        rng=rng,
        max_distance = scan_radius, #https://community.robotshop.com/blog/show/lidar-light-amp-laser-based-distance-sensors
        perception_angle=90,
    )
    
    print(
        f"Initialized sensor with rate {cfg.sensing_rate} and noise scale {cfg.noise_scale}."
    )
    return sensor


def get_robot(cfg, sensor, start, heading_start=0):
    
    robot = pypolo.robots.DigitRobot(
        sensor=sensor,
        start= np.append(start, heading_start),
        control_rate=cfg.control_rate,
        max_lin_vel=cfg.max_lin_vel,
        max_ang_vel=cfg.max_ang_vel,
        goal_radius=cfg.goal_radius,
    )
    
    print(f"Initialized Digit with control rate {cfg.control_rate}.")
    return robot


def get_visualizer(cfg, env):
    visualizer = pypolo.utils.Visualizer(
        cfg.env_extent, cfg.task_extent, cfg.plot_robot_interval
    )
    vmin, vmax = np.min(env), np.max(env)
    visualizer.vmins[1], visualizer.vmaxs[1] = vmin, vmax
    visualizer.plot_image(
        index=0, matrix=env, title="Ground Truth", vmin=vmin, vmax=vmax
    )
    print(f"Initialized visualizer.")
    return visualizer


def pilot_survey(cfg, sensor, rng):
    
    '''
    Randomly sample 500 points from the evaluation points
    '''

    #Generate all possible evaluation points in the environment   
    x_min, x_max, y_min, y_max = cfg.task_extent
    num_x, num_y = cfg.eval_grid
    x_grid = np.linspace(x_min, x_max, num_x)
    y_grid = np.linspace(y_min, y_max, num_y)
    xx, yy = np.meshgrid(x_grid, y_grid)
    eval_inputs = np.column_stack((xx.flatten(), yy.flatten()))
    
    np.random.shuffle(eval_inputs)

    # Split the shuffled list into three subsets
    x_init = eval_inputs[:500]
    y_init = sensor.get(x_init[:, 0], x_init[:, 1]).reshape(-1, 1)
    
    print(f"Collected {len(x_init)} samples in pilot survey.")

    return x_init, y_init

    
    #------------------------------------------------------------------
    
    
    # '''
    # Sampled by Diff-Drive Robot
    # '''
    
    # point_sensor = pypolo.sensors.PointSensor(
    #     matrix=sensor.matrix,
    #     env_extent=cfg.env_extent,
    #     rate=cfg.sensing_rate,
    #     noise_scale=cfg.noise_scale,
    #     rng=rng,
    # )
    
    # robot = pypolo.robots.DiffDriveRobot(
    #     sensor=sensor,
    #     state=np.array([cfg.task_extent[1], cfg.task_extent[2], -np.pi]),
    #     control_rate=cfg.control_rate,
    #     max_lin_vel=cfg.max_lin_vel,
    #     max_ang_vel=cfg.max_ang_vel,
    #     goal_radius=cfg.goal_radius,
    # )
    # print(f"Initialized robot with control rate {cfg.control_rate}.")
    
    # bezier_planner = pypolo.planners.BezierPlanner(cfg.task_extent, rng)
    # goals = bezier_planner.plan(num_points=cfg.num_bezier_points)
    # robot.goals = goals
    # while len(robot.goals) > 0:
    #     robot.step(current_heading=0, num_targets=10)
    # x_init, y_init = robot.commit_samples()
    # print(f"Collected {len(x_init)} samples in pilot survey.")
    
    # return x_init, y_init
    
    #-------------------------------------------------------------------
    
    # '''
    # Grid-wise Sample
    # '''
        
    # x_min, x_max, y_min, y_max = cfg.task_extent
    # x_grid = np.linspace(x_min, x_max, 10)
    # y_grid = np.linspace(y_min, y_max, 10)
    # xx, yy = np.meshgrid(x_grid, y_grid)
    # x_init = np.column_stack((xx.flatten(), yy.flatten()))
    # y_init = sensor.get(x_init[:, 0], x_init[:, 1]).reshape(-1, 1)
    
    
    # print(f"Collected {len(x_init)} samples in pilot survey.")

    # return x_init, y_init

    #-------------------------------------------------------------------


def get_model(cfg, x_init, y_init, x_scaler, y_scaler):
    
    if cfg.kernel.name == "RBF":
        kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        kernel.base_kernel.lengthscale = cfg.kernel.lengthscale
        kernel.outputscale = cfg.kernel.outputscale

    elif cfg.kernel.name == "AK":
        kernel = gpytorch.kernels.ScaleKernel(
            pypolo.models.gp.kernels.AttentiveKernel(
                dim_input=x_init.shape[1],
                dim_hidden=cfg.kernel.dim_hidden,
                dim_output=cfg.kernel.dim_output,
                min_lengthscale=cfg.kernel.min_lengthscale,
                max_lengthscale=cfg.kernel.max_lengthscale,
            )
        )
        
    elif cfg.kernel.name == "NN":    
            
        model = pypolo.models.gp.GPRModel_NNKernel(
            cfg=cfg,
            x_train = x_init,
            y_train = y_init)
        
        print(f"Initialized model {cfg.model.name} with kernel {cfg.kernel.name}.")
        return model
        
    else:
        
        raise ValueError(f"Unknown kernel: {cfg.kernel}")
    
    if cfg.model.name == "GPR":
        model = pypolo.models.gp.GPRModel(
            cfg=cfg,
            x_train=x_init,
            y_train=y_init,
            x_scalar=x_scaler,
            y_scalar=y_scaler,
            kernel=kernel,
            noise_var=cfg.model.noise_var,
            num_sparsification=cfg.model.num_sparsification,
        )
    else:
        raise ValueError(f"Unknown model: {cfg.model.name}")
    print(f"Initialized model {cfg.model.name} with kernel {cfg.kernel.name}.")
    return model


def get_planner(cfg, rng, start, goal, obstacles=None, step_len=0.1, goal_sample_rate=0.6, search_radius=0.6, iter_max=10000, max_turn_angle=np.deg2rad(10)):
    
    if cfg.planner.name == "MaxEntropy":
        planner = pypolo.planners.MaxEntropyPlanner(
            cfg.task_extent, rng, cfg.planner.num_candidates
        )        
    elif cfg.planner.name == "LocalRRTStar":
        start = tuple(start)
        goal = tuple(goal)
        planner = pypolo.planners.LocalRRTStar(
            cfg.task_extent, rng, start, goal, step_len=step_len, goal_sample_rate=goal_sample_rate, search_radius=search_radius, iter_max=iter_max, max_turn_angle=abs(max_turn_angle)
        )
        
    else:
        raise ValueError(f"Unknown planner: {cfg.planner.name}")
    print(f"Initialized planner {cfg.planner.name}.")
    return planner


def model_update(cfg, num_steps, model, evaluator):
    start = time()
    losses = model.optimize(num_steps=num_steps) #calculate lossses
    end = time()
    evaluator.training_times.append(end - start)
    evaluator.losses.extend(losses)


def evaluation(model, evaluator, robot):
    # print(f"Prediction...")
    start = time()
    mean, std = model.predict(evaluator.eval_inputs)
    end = time()
    print("prediction time", end - start)
    evaluator.prediction_times.append(end - start)
    evaluator.compute_metrics(mean, std)


def visualization(visualizer, evaluator, final_goal, x_inducing=None):
    # print(f"Visualization...")
    visualizer.plot_prediction(evaluator.mean, evaluator.std, evaluator.abs_error)
    visualizer.plot_data(evaluator.x_train, final_goal)
    if x_inducing is not None:
        print("Plotting inducing inputs...")
        visualizer.plot_inducing_inputs(x_inducing)
    visualizer.plot_metrics(evaluator)
    
    
def step_local_planner(model, evaluator, robot, planner, global_goal,  num_step=1, samples_per_dt=10, visualizer=None, \
    log_data=np.empty((0,22)), log_data_PSP=np.empty((0,15)), log_data_GP=np.empty((0,19)), log_data_flag=False, verbose=True,\
        time_elapsed=0 , decision_epoch=0):
    
    if verbose:
        print("Run PSP")
    
    #Final local goal
    final_goal = np.array([planner.s_goal.x , planner.s_goal.y])    
    iters = 0
    replanning_local = False
    
    current_waypoint_index = planner.path.index([robot.wp_c_x, robot.wp_c_y])

    robot.goals = robot.path[current_waypoint_index:]
    
    while iters < num_step and robot.has_goals:

        if np.linalg.norm([robot.wp_n_x - robot.wp_c_x, robot.wp_n_y - robot.wp_c_y]) < 0.00001:
            break
    
    # while iters < num_step and (np.linalg.norm([final_goal[0] - robot.wp_c_x, final_goal[1] - robot.wp_c_y]) > planner.goal_radius):
        # robot.goals = np.atleast_2d([robot.wp_n_x, robot.wp_n_y]) #next waypoint
        # print("current local goal:", robot.goals)
        if verbose:
            print("distance to subgoal",np.linalg.norm([final_goal[0] - robot.wp_c_x, final_goal[1] - robot.wp_c_y]))
            print("current state: " ,(robot.state[:2], np.rad2deg(robot.state[2])))
        
        visualizer.plot_goal(final_goal, global_goal)        
        visualizer.pause()
        plot_counter = 0
        
        plot_counter += 1
        
        if log_data_flag:
            
            replanning_local, PSP_output, PSP_hyperparams, GP_output = robot.step(model, num_targets=samples_per_dt, log_data_flag=log_data_flag, verbose=verbose)
            log_data = np.vstack([log_data, PSP_output])
            log_data_PSP = np.vstack([log_data_PSP, PSP_hyperparams])
            log_data_GP = np.vstack([log_data_GP, np.append(np.array([decision_epoch, time_elapsed, np.sum(evaluator.abs_error)]), GP_output)])
            
        else:
            
            replanning_local = robot.step(model, num_targets=samples_per_dt, log_data_flag=log_data_flag, verbose=verbose)
                    
        # if visualizer.interval > 0 and plot_counter % visualizer.interval == 0:
            # visualizer.plot_robot(robot.state)
            # visualizer.pause()
            
        if replanning_local:
            
            print("High Uncertainty within the trajectory, replan local is triggered")
            break
                
        iters += 1
        
    if len(robot.sampled_observations) > 0:

        x_new, y_new = robot.commit_samples()
        print("End of " + str(iters) + "-steps PSP")
        print(" . \n . \n .")
        
        if log_data_flag:
            return x_new, y_new, log_data, log_data_PSP, log_data_GP, replanning_local
        else:
            return x_new, y_new, replanning_local
        
    else:
        
        if log_data_flag:
            return [], [], log_data, log_data_PSP, log_data_GP, replanning_local
        else:
            return [], [], replanning_local
        
def generate_constraint_points(angle_radians, num_points, spacing, perturb_x):
    points = []

    # Calculate the distance between points along the line
    dx = spacing * math.cos(angle_radians)
    dy = spacing * math.sin(angle_radians)

    # Generate points along the line at the given angle
    for i in range(0, num_points + 1):
        x = i * dx + perturb_x
        y = i * dy
        points.append([x, y])

    return points

def normalize_value(value, str):
    """
    Normalize a value within a given range.

    Args:
    - value (float): The value to normalize.
    - min_val (float): The minimum bound of the range.
    - max_val (float): The maximum bound of the range.

    Returns:
    - float: The normalized value.

    """
    
    if str == 'entropy':
        min_val = -20.80388728
        max_val = -0.67507272
        
    elif str == 'error':
        min_val =  0.00083169
        max_val = 0.49974317
    # Check if the range is valid
    if min_val >= max_val:
        raise ValueError("Minimum value must be less than maximum value")

    # Normalize the value
    normalized_value = (value - min_val) / (max_val - min_val)
    
    return normalized_value
    

# ----- Functions for allocating sub-goals -----


def local_grid_init(cfg, global_start, global_goal, digit_step_len, num_step_local):
    localGrid_info = {"center":[],
                "bound_x":[],
                "bound_y":[],
             "local_goal": [],
             "local_data":[],
             "all_centers":[],
             "k_grids":[],
             "(num_grid_x, num_grid_y)":[],
             "global_goal":[],
             "global_start":[]
             }

    # NOTE number of sub_environment = num_grid_x * num_grid_y

    num_grid_x = int((cfg.env_extent[1] - cfg.env_extent[0])/(num_step_local*digit_step_len))
    num_grid_y = int((cfg.env_extent[3] - cfg.env_extent[2])/(num_step_local*digit_step_len))

    k_grids = num_grid_x * num_grid_y #initilize number of clusters
    all_center_point = []

    xi = np.linspace(cfg.env_extent[0] , cfg.env_extent[1], num_grid_x + 1)
    yi = np.flip(np.linspace(cfg.env_extent[2] , cfg.env_extent[3], num_grid_y + 1))

    #Overlap-ness of the subgrids (defined in percentage)
    overlap = 0

    for i in  range(len(xi) - 1):
        for j in range(len(yi) - 1):
            
            # Set the bounds for each dimension
            bounds_x = (  xi[i] * (1-overlap)   ,  min(xi[i + 1] * (1+overlap), cfg.env_extent[1] ) )
            bounds_y = (  min(yi[j] * (1+overlap), cfg.env_extent[3]) , yi[j + 1] * (1-overlap) )
            
            bounds_y = tuple(reversed(bounds_y))
            # print("x", bounds_x)
            # print("y", bounds_y)     
            
            #Find center point of the grid
            midpoint_x = (bounds_x[0] + bounds_x[1]) / 2
            midpoint_y = (bounds_y[0] + bounds_y[1]) / 2
                    
            # print(midpoint_x)
            # print(midpoint_y)
            localGrid_info["bound_x"].append(bounds_x)
            localGrid_info["bound_y"].append(bounds_y)
            localGrid_info["center"].append(np.atleast_2d([midpoint_x, midpoint_y]))
            all_center_point.append([midpoint_x, midpoint_y])
            
    localGrid_info["all_centers"] = all_center_point
    localGrid_info["k_grids"] = k_grids
    localGrid_info["(num_grid_x, num_grid_y)"] = (num_grid_x, num_grid_y)
    localGrid_info["global_goal"] = global_goal
    localGrid_info["global_start"] = global_start
    
    return localGrid_info

def locate_state(localGrid_info ,state):
    
    #Get param
    k_grids = localGrid_info["k_grids"]
    
    state = np.atleast_2d(state)
    for index in range(k_grids):
        lb_x = localGrid_info["bound_x"][index][0]
        ub_x = localGrid_info["bound_x"][index][1]
        lb_y = localGrid_info["bound_y"][index][0]
        ub_y = localGrid_info["bound_y"][index][1]
        
        if (state[:,0] >= lb_x and state[:,0] <= ub_x) and (state[:,1] >= lb_y and state[:,1] <= ub_y):
            # print("state is bounded by: ", (lb_x, ub_x, lb_y, ub_y))
            return index
    
def assign_subgoals(localGrid_info, global_path, grid_size):
    
    #Get param
    # global_start = localGrid_info["global_start"]
    global_goal = localGrid_info["global_goal"]
    k_grids = localGrid_info["k_grids"]
    

    global_path = np.array(global_path)
    manhattan_distance = np.sum(np.abs(np.diff(global_path, axis=0)))
    
    num_indices = int(manhattan_distance/grid_size) 
    # print("global path len: ", len(global_path))
    # print("num_index", num_indices)
    
    # Calculate the indices
    indices = np.linspace(0, len(global_path)-1, num=num_indices, dtype=int)

    # Get the values at those indices
    subgoals = global_path[indices]
    # print(subgoals)
    
    #Clear old data
    # localGrid_info["local_data"] = []
    localGrid_info["local_goal"] = []
    
    for _ in range(k_grids):
        # localGrid_info["local_data"].append(np.array([]))
        localGrid_info["local_goal"].append(np.array([]))
        
    
    #Assgin global waypoint to the different subgrid
    for subgoal in subgoals:
        
        #Skip starting position
        # if np.linalg.norm([subgoal[0] - global_start[0], subgoal[1] - global_start[1]]) < 0.0001: 
        #     continue
        
        index = locate_state(localGrid_info, subgoal)
        
        localGrid_info["local_goal"][index] = np.atleast_2d(subgoal)
        # if localGrid_info["local_goal"][index].shape[0] == 0:
        #     localGrid_info["local_goal"][index] = np.atleast_2d(np.append(localGrid_info["local_goal"][index] , subgoal))
        # else:
        #     localGrid_info["local_goal"][index] = np.atleast_2d(np.vstack([localGrid_info["local_goal"][index] , subgoal]))
            
        #     #Only keep the subgoal closest to the global goal
        #     distances = np.linalg.norm(localGrid_info["local_goal"][index] - global_goal, axis=1)
        #     min_distance_index = np.argmin(distances)
        #     localGrid_info["local_goal"][index] = localGrid_info["local_goal"][i][min_distance_index]
                    
                    
    #Only keep the subgoal closest to the global goal
    # for i in range(k_grids):
    #     if localGrid_info["local_goal"][i].shape[0] < 2:
    #         if localGrid_info["local_goal"][i].shape[0] == 1 and np.linalg.norm(localGrid_info["local_goal"][i] - global_goal, axis=1) > 0.00001:
    #             localGrid_info["local_goal"][i] = []
    #         continue
        
    #     # index_subgoal = [global_path.index(subgoal) for subgoal in list(localGrid_info["local_goal"][i])]
    #     # localGrid_info["local_goal"][i] = localGrid_info["local_goal"][i][-1]
        
            
    #     # print(i)
        
    #     distances = np.linalg.norm(localGrid_info["local_goal"][i] - global_goal, axis=1)
    #     min_distance_index = np.argmin(distances)
        
    #     localGrid_info["local_goal"][i] = localGrid_info["local_goal"][i][min_distance_index]
    
    
        # randomly_selected_subgol = random.choice(localGrid_info["local_goal"][i])
        # localGrid_info["local_goal"][i] = randomly_selected_subgol
        
        
        # print("Assign Goal: ", localGrid_info["local_goal"][i][min_distance_index])
        # print("To Grid Index: ", i)
        
       
            
def find_next_goal(localGrid_info, current_state, heading_c):
 
    num_grid_x, num_grid_y = localGrid_info["(num_grid_x, num_grid_y)"]
    global_goal = localGrid_info["global_goal"]
    
    #Convert to matrix indices
    current_index = locate_state(localGrid_info, current_state)
    i, j = (current_index % (num_grid_y)), int(current_index / (num_grid_x)) 
    print("current state:", current_state)
    print("current index: ", current_index)
    print("current grid: ", (i,j))
    
    #Clear current goal
    localGrid_info["local_goal"][current_index] = []
    
    
    for k in [1,2,3]:
        
        adjacent_indices = []
        goal_candidate = np.array([])
        index_candidate = []
        
        if i > 0:
            adjacent_indices.append((i- k,j))
            if j > 0:
                adjacent_indices.append((i- k,j- k))
                
        if i+ k < (num_grid_y -  k):
            adjacent_indices.append((i+ k,j))
            if j+ k < (num_grid_x -  k):
                adjacent_indices.append((i+ k,j+ k))
            
        if j > 0:
            adjacent_indices.append((i,j- k))
            if i+ k < (num_grid_y -  k):
                adjacent_indices.append((i+ k,j- k))
            
        if j+ k < (num_grid_x -  k):
            adjacent_indices.append((i,j+ k))
            if i > 0:
                adjacent_indices.append((i- k,j+ k))
            
        # print("adjacent grid: ", adjacent_indices)
        
        #Iterate through adjacent grid
        for i_next, j_next in adjacent_indices:
            
            #Convert back to vectorized list index
            next_index = j_next * num_grid_x + i_next
            # print("convert grid: ", (i_next, j_next))
            # print("to index: ", next_index)
            # print("grid[index]: ", localGrid_info["local_goal"][next_index])
            
            
            if len(localGrid_info["local_goal"][next_index]) > 0:
                
                print("Return next sub-goal")
                
                goal = np.atleast_2d(localGrid_info["local_goal"][next_index]) 
                return goal, next_index
                

        #         if len(goal_candidate) == 0:
        #             goal_candidate = goal
        #         else:
        #             goal_candidate = np.vstack([goal_candidate, goal])
                    
        #         for _ in range(len(goal_candidate)):
        #             index_candidate.append(next_index)
                
                
        
        # print("goal_candidate",goal_candidate)
        # print("index_candidate", index_candidate)
        # if len(goal_candidate) == 0:
        #     continue
        
        # elif len(goal_candidate) == 1:
        #     return goal_candidate[0], index_candidate[0]
        # else:
    
        #     goal_candidate = np.array(goal_candidate)
        #     print("goal_candidate2",goal_candidate)
        #     distances = np.linalg.norm(goal_candidate - global_goal, axis=1)
        #     min_distance_index = np.argmin(distances)        
    
        #     return goal_candidate[min_distance_index, :], index_candidate[min_distance_index]
        #     # current_state = np.atleast_2d(current_state)
                   
    return [], 0


# def heading_quadrant(heading):

#     # Calculate sine and cosine of the angle
#     sin_angle = math.sin(heading)
#     cos_angle = math.cos(heading)

#     # Determine quadrant based on signs of sine and cosine
#     if sin_angle >= 0 and cos_angle >= 0:
#         quadrant = 1
#     elif sin_angle >= 0 and cos_angle < 0:
#         quadrant = 2
#     elif sin_angle < 0 and cos_angle < 0:
#         quadrant = 3
#     else:
#         quadrant = 4

#     return quadrant


# ----- Functions for allocating sub-goals -----

# from scipy.stats import norm

# def find_quantile(value, mean_pred, std_pred):
#     # Calculate the standardized value
#     z_score = (value - mean_pred) / std_pred
#     # Calculate the cumulative probability
#     cumulative_prob = norm.cdf(z_score)
#     # Convert cumulative probability to quantile
#     quantile = 1 - cumulative_prob  # 1 - CDF gives the tail probability
#     return quantile

# def pinball_loss(y, y_pred, quantile=0.9):
#     # Calculate the quantile of y_pred
#     y_pred_q = np.quantile(y_pred, quantile)
    
#     # Calculate the pinball loss
#     pinball_score = np.mean(np.maximum(quantile * (y - y_pred_q), (quantile - 1) * (y_pred_q - y)))
    
#     return pinball_score




'''
-------------
MAIN FUNCTION
-------------
'''

@hydra.main(version_base=None, config_path="configs", config_name="main")
def main(cfg: DictConfig) -> None:
    
    # LOGGER
    
    #Configure File Name
    home_path = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__name__)) , os.pardir))
    log_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_dir = os.path.join(home_path, 'logs/digit-state-' + log_time + ".csv")
    log_PSP_dir = os.path.join(home_path, 'logs_psp_belief/psp-hyperparam-' + log_time + ".txt")
    log_GP_dir = os.path.join(home_path, 'logs_GP_prediction/GP-data-' + log_time + ".csv")
    log_eval_dir = os.path.join(home_path, 'logs_GP_prediction/eval-data-' + log_time + ".csv")
    
    log_data = np.empty((0,22))
    log_data_PSP = np.empty((0,15))
    log_data_flag = False
    
    log_data_GP = np.empty((0,22))
    log_data__GP_flag = False
    
    log_eval = np.empty((0,5))
    
    
    #---------------------------------

    # ALL PARAMS
    
    #Environment Properties
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)
    # env_path = home_path + '/raw_data/terrains/Kinskuch_River.png'
    # env_path = home_path + '/raw_data/terrains/n30_w106_1arc_v3.jpg'
    env_path = home_path + '/raw_data/terrains/Powell_Lake.png'
    # env_path = home_path + '/raw_data/terrains/n40_w116_1arc_v3.jpg'
    # env_path = home_path + '/raw_data/terrains/n24w102.jpg'
    # env_path = home_path + '/raw_data/terrains/n16e102.jpg'
    env_path = home_path + '/raw_data/terrains/n17e102.png'
    # env_path = '/home/kmuenpra/Desktop/terrains/n16e102_edited.png' (0,0) -> (8,-2)
    # env_path = home_path + '/raw_data/terrains/N42W114.png'
    # env_path = home_path + '/raw_data/terrains/terrain6.png'
    
    # env_path = home_path +'/raw_data/terrains/N16E102.png'
    
    
   


    max_height = 0.5
    
    #Starting Position    n17e102                N16E102         Powell             n24w102                  Kinskuch        
    global_start =        (-5.0,-6.0)        #  (-8,0)     #     (-7.5,6.5)     # (-5,-9.0) #       (0,0)     #(-5.5,5.4)   (-11, 10.7) # (-8,6.5) #(-9, -9) #
    global_goal =          (6.0,6.0)         #  (8,4)      #      (8,-7)        #  (6,3)     #         (4.5,-7)    #(-6, -8)    (-12, -16)  #(7,-3)  #(-1.5 2)
    heading_start =   np.deg2rad(45)
    
    #Sensor config    
    scan_radius = 6 #How far can LIDAR sensor see [meter]
    samples_per_dt = 9 # How many samples is collcted at each PSP step
    
    #---------- Mujoco setting ------------
    
    # # Digit's constraint
    # digit_step_len = 0.118 #0.1 #This Step length will remain throughout the planning
    # digit_max_turn_angle = np.deg2rad(5)
    # tread_max_height = 0.15 #0.15#Digit's Maximum treadable Terrain Height (should be less than environment's max_height)
    
    # #Local Planner config
    # goal_sample_rate_local = 0.65
    # iter_max_local = 2000
    # obstacles_margin_local = 0.4#0.8 #0.1
    # goal_radius_local = 0.5#0.4
    # n_path_gen = 1 #Number of paths generated to be evaluated in each grid
    
    # #Global Planner config
    # goal_sample_rate_global = 0.1
    # iter_max_global = 5000
    # obstacles_margin_global = 0.4#0.8 #0.2
    # goal_radius_global = 0.7  #0.5
    # step_len_global  = digit_step_len
    
    #---------- RRT setting ------------
    
    # Digit's constraint
    digit_step_len = 0.1 #This Step length will remain throughout the planning
    digit_max_turn_angle = np.deg2rad(10)
    tread_max_height = 0.5 #Digit's Maximum treadable Terrain Height (should be less than environment's max_height)
    
    #Local Planner config
    goal_sample_rate_local = 0.65
    iter_max_local = 2000
    obstacles_margin_local = 0.4
    goal_radius_local = 0.5#0.4
    n_path_gen = 1 #Number of paths generated to be evaluated in each grid
    
    #Global Planner config
    goal_sample_rate_global = 0.2
    iter_max_global = 12000
    obstacles_margin_global = 0.4#0.13
    goal_radius_global = 1 #0.5
    step_len_global  = 0.2
    
    #----------------------------------
    
    #Environment Grid config
    num_step_local = 13#15 #Expected Number of Step in each local grid
    
    #Update other config
    PSP_num_steps = 20 #How many step of PSPlanner before updating the visual plots
    cfg.env_extent = [-10.0, 10.0, -10.0, 10.0]
    cfg.noise_scale = 0.03 * max_height
    cfg.planner.name = "LocalRRTStar"
    cfg.task_extent = [cfg.env_extent[0] + 0.1, cfg.env_extent[1] - 0.1, cfg.env_extent[2] + 0.1, cfg.env_extent[3] - 0.1 ]# cfg.env_extent #
    cfg.eval_grid = [50,50]
    cfg.kernel.name = "NN"
    cfg.num_train_steps = 25
    '''
    TODO:
    - find a better way to quantify risk_radius to define the radius of the obstacles
    - NN kernel model.predict(): return a correct std, mean + faster way each local points to predict without K-mean clustering
    - Implement KD-Tree for all kernel >> Try to make it work for NN kernle first:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.KDTree.query.html
    '''

    #---------------------------------
    
    #Set up environment, visualizer, evaluator, and sensors

    rng = set_random_seed(cfg.seed)
    env = get_environment_from_image(cfg, png_path=env_path, resize=(50,50), max_height=max_height)
    visualizer = get_visualizer(cfg, env)
    
    sensor = get_sensor(cfg, env, rng, scan_radius=scan_radius)
    
    evaluator = pypolo.utils.Evaluator(sensor, cfg.task_extent, cfg.eval_grid)
    
    #---------------------------------
    
    # GLOBAL PLANNER INIT.
    
    #Define Global Planner and Local grid tracker

    #Initialize Global Planner and generate global trajectory
    print("Initialized planner GlobalRRTStar")
    global_planner = pypolo.planners.GlobalRRTStar(
                    cfg.task_extent, rng, global_start, global_goal, step_len=step_len_global, goal_sample_rate=1, search_radius=0.6, iter_max=iter_max_global)
    global_planner.plan(heading_c=heading_start)
    
    print("Start", global_start)
    print("Global Goal", global_goal)
    
    #Define subgoal in each grid
    localGrid_info = local_grid_init(cfg, global_start=global_start ,global_goal=global_goal, digit_step_len=digit_step_len, num_step_local=num_step_local)
    assign_subgoals(localGrid_info, global_planner.path, grid_size=(num_step_local*digit_step_len))

    
    #---------------------------------
    
    # ROBOT INIT.
    
    #Define Digit robot
    start = np.array(global_start)
    robot =  get_robot(cfg, sensor, start, heading_start=heading_start) 
    
    #---------------------------------
    
    # TERRAIN GP MODEL INIT.
    
    #Collect training data from pilot survey
    x_init, y_init =  pilot_survey(cfg, sensor, rng)

    x_scaler = MinMaxScaler()
    x_scaler.fit(x_init) #find min/max of (x1,x2) coordinates
    y_scaler = StandardScaler()
    y_scaler.fit(y_init) #find mean and std of y_init
    evaluator.add_data(x_init, y_init) #set training data
    
    #Collect training data from the terrain at starting point
    # x_init, y_init = robot.sensor.sense( robot.state, robot.heading_c, ray_tracing = True, num_targets = 100)
    # evaluator.add_data(x_init, y_init) #set training data
    
    #Define GP Model
    model = get_model(cfg, x_init, y_init, x_scaler, y_scaler)
    '''
    uncomment thes 2 lines when done debugging
    |
    V
    '''
    # model_update(cfg, cfg.num_train_steps, model, evaluator)
    # evaluation(model, evaluator, robot)

    # #Collect training data from the terrain at starting point
    x_at_start, y_at_start, _ = robot.sensor.sense( model, robot.state, robot.heading_c, ray_tracing = True, num_targets = 200)
    
    # # Clean the data collected
    # col = model.getMatEntries.xs_to_cols(x_at_start[:, 0])
    # row = model.getMatEntries.ys_to_rows(x_at_start[:, 1])
    # clean_indices = row * cfg.eval_grid[1] + col
    
    # filtered_x_train, filtered_indices = np.unique(evaluator.eval_inputs[clean_indices], axis=0, return_index=True)
    # filtered_y_train = y_at_start[filtered_indices]

    # model.add_data(filtered_x_train, filtered_y_train)
    # evaluator.add_data(filtered_x_train, filtered_y_train) #set training data
    # model_update(cfg, cfg.num_train_steps, model, evaluator)
    # evaluation(model, evaluator, robot)
        
    #Old
    
    if cfg.kernel.name == "NN":
        num_train_steps = cfg.num_train_steps
    else:
        num_train_steps = 450
    
    model.add_data(x_at_start, y_at_start)
    evaluator.add_data(x_at_start, y_at_start) #set training data
    model_update(cfg, num_train_steps, model, evaluator) #
    evaluation(model, evaluator, robot)
    
    #---------------------------------
    
    #Define the obstacles in terrain, if found at the starting point
    
    #Replan Global Trajectory
    mean, std = model.predict(evaluator.eval_inputs)
    
    #Define risk region
    obs_cir = []
    # risk_radius = 0.15

    std = np.atleast_2d(std)
    risk_index = np.where(mean > tread_max_height)[0].tolist()
    # risk_radius = np.max(std[risk_index], axis=0) * 2
    # print(std[risk_index])
    for x,y,risk_radius in np.hstack([evaluator.eval_inputs[risk_index].tolist(), std[risk_index].tolist()]):
    # for x,y in evaluator.eval_inputs[risk_index].tolist():
        obs_cir.append([x,y, 0.15])

        # obs_cir.append([x,y, max(0.15,2*risk_radius)])
        
    #----------------------------------------------
    # entropies = 0.5 * np.log(2 * np.pi * np.square(std)) + 0.5
    # print("max Entropy", np.max(entropies, axis=0))
    # print("min Entropy", np.min(entropies, axis=0) )
    #---------------------------------------------
        
    #Define Heading constraint for the global planner
    COS = np.cos(robot.heading_c)
    SIN = np.sin(robot.heading_c)
    
    jitter = (3*10**(-1))
    num_pts = 30 #30
    n_turns = int((np.pi/2)/digit_max_turn_angle)
    
    horz_dist = 0
    vert_dist = 0
    for i in range(n_turns + 2):
        horz_dist += digit_step_len*np.cos(i*digit_max_turn_angle)
        vert_dist += digit_step_len*np.sin(i*digit_max_turn_angle)
        
    line_constr = 3*digit_step_len #np.sqrt(((jitter + horz_dist)**2 + vert_dist**2))
    angle_constr = np.arctan2(vert_dist, (jitter + horz_dist))
    
    # print("Expected Local Goal Radius for Digit", line_constr)
    # print("Highest Uncertainty", np.max(std, axis=0))
    
    
    # angle_constr = np.deg2rad(65)
    # bound1 = generate_constraint_points(angle_constr,  num_pts, abs(3*digit_step_len/(num_pts*np.cos(angle_constr))), perturb_x)
    # bound2 = generate_constraint_points(-angle_constr, num_pts, abs(3*digit_step_len/(num_pts*np.cos(angle_constr))), perturb_x)
    perturb_x = -jitter
    bound1 = generate_constraint_points(angle_constr,  num_pts, line_constr/num_pts, perturb_x)
    bound2 = generate_constraint_points(-angle_constr, num_pts, line_constr/num_pts, perturb_x)
    bound = bound1 + bound2
    
    global_obs_bound = []
    for point in bound:
        c_x = point[0]*COS - point[1]*SIN + global_start[0]
        c_y = point[0]*SIN + point[1]*COS + global_start[1]
        global_obs_bound.append([c_x, c_y, (horz_dist)/num_pts, vert_dist/num_pts])
        
    # print("bound", global_obs_bound)

    #Update global planner configuration
    global_planner.update_obs(obs_cir=obs_cir, obs_bound=global_obs_bound, obs_rec=[], obstacles=[])
    global_planner.goal_sample_rate = goal_sample_rate_global
    global_planner.utils.delta = obstacles_margin_global
    global_planner.goal_radius = goal_radius_global
    
    
    #---------------------------------
    
    # REPLANNING INITIAL GLOBAL PATH (if necessary)
    
    start_pose = global_start
    
    
    # Check if global path has risky Terrain elevation
    z, _ = model.predict(global_planner.path)
    index = np.where(z > tread_max_height)
    index = index[0]
    
    replanning = False
    
    if len(index) > 0:
        replanning = True
    else:
        #Check if current trajetory crosses any obstacles
        for i in range(len((global_planner.path_vertex)) - 1):
            if global_planner.utils.is_collision(global_planner.path_vertex[i], global_planner.path_vertex[i + 1]):
                replanning = True
                break
            
            
    # while replanning:
    #     print("Replanning Global Trajectory!")
    #     global_planner.reset_tree(start=global_start)
    #     global_planner.plan()
    #     assign_subgoals(localGrid_info=localGrid_info, global_path=global_planner.path, grid_size=(num_step_local*digit_step_len))
    #     goal, localGridIndex = find_next_goal(localGrid_info=localGrid_info, current_state=global_start, heading_c=robot.heading_c)
    #     goal = np.atleast_2d(goal)
    #     global_planner.utils.delta = max(0.1, global_planner.utils.delta - 0.05)
    #     replanning_count += 1
    
        
    #     if replanning_count > 5:
    #         raise TimeoutError
            
    #     if len(goal[0]) == 0:
    #         continue
        
    #     print("Next subgoal: ", goal)
        
    #_------------------ Old condition ---------------------
    
    while True:
        
        #Replan global planner again for any new obstacles
        global_planner.reset_tree()
        global_planner.plan(heading_c=robot.heading_c)
        
        
        # if not global_planner.is_goal_reachable():
            
        #     print("No goal-reachable path found, collecting more data and replanning...")
            
        #     #Collect training data from the terrain at starting point
        #     x_at_start, y_at_start, _ = robot.sensor.sense( model, robot.state, robot.heading_c, ray_tracing = True, num_targets = 20)

        #     model.add_data(x_at_start, y_at_start)
        #     evaluator.add_data(x_at_start, y_at_start)
        #     model_update(cfg, cfg.num_train_steps, model, evaluator)
        #     evaluation(model, evaluator, robot)
            
        #     mean, std = model.predict(evaluator.eval_inputs)
            
        #     #Define risk region
        #     obs_cir = []

        #     std = np.atleast_2d(std)
        #     risk_index = np.where(mean > tread_max_height)[0].tolist()
            
        #     for x,y,risk_radius in np.hstack([evaluator.eval_inputs[risk_index].tolist(), std[risk_index].tolist()]):
        #     # for x,y in evaluator.eval_inputs[risk_index].tolist():
        #         obs_cir.append([x,y, 0.15])
                
        #     global_planner.update_obs(obs_cir=obs_cir, obs_bound=global_obs_bound, obs_rec=[], obstacles=[])
            
        #     continue
                
        
        print("Initialize Global Trajectory: ")
        assign_subgoals(localGrid_info=localGrid_info, global_path=global_planner.path, grid_size=(num_step_local*digit_step_len))
        # print("Global Plan", global_planner.path)
        # print(global_planner.path)
        # for i in range(len((global_planner.path_vertex)) - 1):
        #     if global_planner.utils.is_collision(global_planner.path_vertex[i], global_planner.path_vertex[i + 1]):
        #         print(global_planner.path_vertex[i].x)
        #         print(global_planner.path_vertex[i].y)


        #Local Planner that robot tracks        
        goal, localGridIndex = find_next_goal(localGrid_info=localGrid_info, current_state=start_pose, heading_c=robot.heading_c)
        goal = np.atleast_2d(goal)
        print("Local Goal", goal)
        
        #if nearby subgoal is not found
        if goal[0].shape[0] == 0:
            global_planner.goal_sample_rate = min(0.35, global_planner.goal_sample_rate - 0.1)
            # global_planner.utils.delta = min(0.1, global_planner.utils.delta - 0.05)
            print("*---------------------------------*")
            print("     Replanning Even Triggered     ")
            print("   > Couldn't find nearby goal based on current position")
            print("*---------------------------------*")
            continue
            
        # # Choose the next-next goal if Digit is too close to the next one
        # if np.linalg.norm([goal[:,0] - global_start[0], goal[:,1] - global_start[1]]) < goal_radius_local:
            
        #     global_planner.goal_sample_rate = min(0.35, global_planner.goal_sample_rate - 0.1)
        #     global_planner.utils.delta = min(0.3, global_planner.utils.delta - 0.05)
        #     print("*---------------------------------*")
        #     print("     Replanning Even Triggered     ")
        #     print("   > Next Local Goal is too close to the start. (Less than Digit's minimum step length)")
        #     print("*---------------------------------*")
        #     continue
        
        break
    
    while np.linalg.norm([goal[:,0] - global_start[0], goal[:,1] - global_start[1]]) <= goal_radius_local + digit_step_len \
        and np.linalg.norm([global_goal[0] - robot.apex_x, global_goal[1] - robot.apex_y]) > goal_radius_global:
        
        print("Current local goal is too close, move onto the next one.")
        
        #locally plan toward next subgoal
        goal, localGridIndex = find_next_goal(localGrid_info=localGrid_info, current_state=goal, heading_c=robot.heading_c)
        goal = np.atleast_2d(goal)
    
    #Reset goal sample rate
    global_planner.goal_sample_rate = goal_sample_rate_global
    global_planner.utils.delta = obstacles_margin_global
    wp_x_start_global = goal[0,0]
    wp_y_start_global = goal[0,1]
    
    #---------------------------------
    
    #PATH EVALUATOR
    
    '''
    TODO
    '''
    # GP_ModelErrorX_Left = home_path  + '/raw_data/GP_ModelError/Train_with_Separate_Stance/GPModelError_posterior_errorX_Left.pkl'
    # GP_ModelErrorX_Right = home_path + '/raw_data/GP_ModelError/Train_with_Separate_Stance/GPModelError_posterior_errorX_Right.pkl'
    # GP_ModelErrorY_Left = home_path  + '/raw_data/GP_ModelError/Train_with_Separate_Stance/GPModelError_posterior_errorY_Left.pkl'
    # GP_ModelErrorY_Right = home_path + '/raw_data/GP_ModelError/Train_with_Separate_Stance/GPModelError_posterior_errorY_Right.pkl'
    
    GP_ModelError = home_path + '/raw_data/GP_ModelError/GPModelError_posterior_change_in_dy_1.pkl'
    # path_evaluator = pypolo.utils.PathEvaluator(filepath_modelErrorX=[GP_ModelErrorX_Left, GP_ModelErrorX_Right] , filepath_modelErrorY=[GP_ModelErrorY_Left, GP_ModelErrorY_Right])

    path_evaluator = pypolo.utils.PathEvaluator(filepath=GP_ModelError)
    #---------------------------------
        
    # LOCAL PLANNER INIT.
    
    planner =  get_planner(cfg, rng, start, goal=np.array(goal).flatten(), \
                        step_len=digit_step_len, goal_sample_rate=goal_sample_rate_local, search_radius=0.6, iter_max=iter_max_local, max_turn_angle=digit_max_turn_angle)
    
    #update local planner configuration
    global_planner.utils.delta = obstacles_margin_global
    planner.goal_radius = goal_radius_local
    planner.utils.delta = obstacles_margin_local
    planner.update_obs(obs_cir=obs_cir, obs_bound=[], obs_rec=[], obstacles=[])
    
    #---------
    
    # GENERATE MULTIPLE PATHS for EVALUATION
        
    all_path = []
    all_vertex = []
    all_path_vertex = []
    all_errors = []
    cost = 500
    
    i = 0
    
    for _ in range(n_path_gen):
        
        planner.plan(heading_c = robot.state[2], obstacle_margin=obstacles_margin_local)        
        all_path.append(planner.path)
        all_vertex.append(planner.vertex)
        all_path_vertex.append(planner.path_vertex)
        # print(planner.path)
        
        '''
        TODO
        '''
        
        entropies, uncertainties = path_evaluator.evaluate_path(model=model, trajectory=planner.path, heading_c=robot.heading_c, current_stance=robot.stance)
        print("Path " + str(i) + " :")
        print("Information Entropy: ", entropies)
        print("PSP square error: ", uncertainties)
        print("----------------------------------")
        
        all_errors.append(uncertainties)
        
        # new_cost = 1000 #call Path Evaluator Here
        # if new_cost < cost:
        #     low_cost_path = []
        #     low_cost_vertex = []
        #     low_cost_path_vertex = []
        
        # planner.utils.delta = min(0.1, planner.utils.delta - 0.01)
        planner.reset_tree()
        
        i += 1

    visualizer.clear()
    # visualizer.plot_path_candidate(all_path, localGrid_info, localGridIndex)
    visualizer.pause()
        
    #Randomly Choose for now
    random_index = random.randint(0, len(all_path) - 1)
    print("Choose path[" + str(random_index) + "] out of " + str(len(all_path)))
    low_cost_path = all_path[random_index] #DELETE THIS LINE LATER
    low_cost_vertex = all_vertex[random_index] #DELETE THIS LINE LATER
    low_cost_path_vertex = all_path_vertex[random_index] #DELETE THIS LINE LATER
    uncertainties = all_errors[random_index]
    
    
    
    
    #---------  
    
    #TRACK PATHS

    planner.track_path(low_cost_path, low_cost_vertex, low_cost_path_vertex)
    robot.update_new_path(model, low_cost_path) #make sure robot has the same path, generated by the local planner
    
    #Reset Goal Sample Rate
    planner.goal_sample_rate = goal_sample_rate_local
    planner.utils.delta = obstacles_margin_local
        
    
    #---------------------------------

    #NOTE Currently not plotting any metrics for evaluation
    visualization(visualizer, evaluator, final_goal=global_goal)
    visualizer.pause()

    decision_epoch = 0
    start_time = time()
    
    x_collect = None
    y_collect = None
    # end_flag = True

    try:
        while (np.linalg.norm([global_goal[0] - robot.state[0], global_goal[1] - robot.state[1]]) > global_planner.goal_radius):
            time_elapsed = time() - start_time
            decision_epoch += 1
            visualizer.plot_title(decision_epoch, time_elapsed)
            
            verbose = False
            replanning_local = False
            
            if log_data_flag:
                x_new, y_new, log_data, log_data_PSP, log_data_GP, replanning_local = step_local_planner(model, evaluator, robot, planner, global_goal, num_step=PSP_num_steps, samples_per_dt=samples_per_dt, \
                                                            visualizer=visualizer, log_data=log_data, log_data_PSP=log_data_PSP, log_data_GP=log_data_GP, log_data_flag=log_data_flag, verbose=verbose, \
                                                                time_elapsed=time_elapsed, decision_epoch=decision_epoch )
                
            else:
                
                x_new, y_new, replanning_local = step_local_planner(model, evaluator, robot, planner, global_goal, num_step=PSP_num_steps, samples_per_dt=samples_per_dt, \
                                                                visualizer=visualizer, log_data=log_data, log_data_PSP=log_data_PSP, log_data_GP=log_data_GP ,log_data_flag=log_data_flag, verbose=verbose)
            
            if len(x_new) > 0 and len(y_new) > 0:
                
                
                # print("Before Cleaning")
                # print("x_new", np.sort(x_new, axis=0))
                # print("y_new", y_new)
                
                # Clean the data collected
                # col = model.getMatEntries.xs_to_cols(x_new[:, 0])
                # row = model.getMatEntries.ys_to_rows(x_new[:, 1])
                # clean_indices = row * cfg.eval_grid[1] + col
                
                # x_new, filtered_indices = np.unique(evaluator.eval_inputs[clean_indices], axis=0, return_index=True)
                # y_new = y_new[filtered_indices]
                
                # print("After Cleaning")
                # print("x_new", x_new)
                # print("y_new", y_new)
                
                evaluator.add_data(x_new, y_new)
                model.add_data(x_new, y_new)
                model_update(cfg, cfg.num_train_steps, model, evaluator)
                evaluation(model, evaluator, robot)
                
                
                log_eval = np.vstack([log_eval, np.array([decision_epoch, np.average(evaluator.std), np.sum(evaluator.std),  np.average(evaluator.abs_error), np.sum(evaluator.abs_error)])])
                
            # for param_name, param in model.named_parameters():
            #     print(f'Parameter name: {param_name:42} value = {param.item()}')
                
                
            # if decision_epoch % 5 == 0:
            # visualizer.save_plot_svg(decision_epoch)
            
            #--------------------------------
            #Check after Stepping
            
            if (np.linalg.norm([global_goal[0] - robot.state[0], global_goal[1] - robot.state[1]]) <= global_planner.goal_radius):
                break
        
            # print(evaluator.abs_error.shape)
            # print(np.sum(evaluator.abs_error))
            
            #---------------------------------
            
            eps = 0.0005
            if np.linalg.norm([robot.wp_n_x - robot.wp_c_x, robot.wp_n_y - robot.wp_c_y]) < eps or replanning_local:
                
                if not replanning_local:
                    print("Local Subgoal reached!")
                    
                #Update the waypoint list of path planner
                # print("goal", goal)
                # print("Shape", goal.shape)
                # try:
                #     current_goal_index = global_planner.path.index(list(goal.flatten()))
                    
                # except ValueError:
                    
                #     print(global_planner.path)
                #     print("END")
                #     return
                
                # print("Current Goal:", [goal, current_goal_index])
                # global_planner.path = global_planner.path[(current_goal_index + 1):]
                # assign_subgoals(localGrid_info=localGrid_info, global_path=global_planner.path)
                localGrid_info["local_goal"][localGridIndex] = []
                
                # if current_goal_index != len(global_planner.path_vertex) - 1:
                #     global_planner.path_vertex = global_planner.path_vertex[current_goal_index:]
            
                #-----------------------------------
            
                # if local goal is reached,
                # --> update the global planner and assign new subgoal to local planner
                
                
                #Define new starting waypoint
                if np.linalg.norm([robot.wp_c_x - robot.apex_x, robot.wp_c_y - robot.apex_y]) > robot.state_deviate:
                    if robot.stance:
                        wp_x_start = robot.apex_x + robot.state_deviate*np.sin(robot.heading_c)
                        wp_y_start = robot.apex_y - robot.state_deviate*np.cos(robot.heading_c)
                    else:
                        wp_x_start = robot.apex_x - robot.state_deviate*np.sin(robot.heading_c)
                        wp_y_start = robot.apex_y + robot.state_deviate*np.cos(robot.heading_c)
                else: 
                    wp_x_start = robot.wp_c_x
                    wp_y_start = robot.wp_c_y
                    
                print("current waypoint", (wp_x_start, wp_y_start))
                
                #-----------------------------------
                
                #Update obstacles in terrain based on new information
                mean, std = model.predict(evaluator.eval_inputs)
                
                #Define risk region
                obs_cir = []
                risk_index = np.where(mean > tread_max_height)[0].tolist()
                
                std = np.atleast_2d(std)
                # print("std[risk]", std[risk_index])
                # risk_radius = np.max(std[risk_index], axis=0) * 2
                for x,y,risk_radius in np.hstack([evaluator.eval_inputs[risk_index].tolist(), std[risk_index].tolist()]):
                # for x,y in evaluator.eval_inputs[risk_index].tolist():
                    obs_cir.append([x,y,0.15])
                    
                
                    
                #Define Heading constraint for the global planner
                COS = np.cos(robot.heading_c)
                SIN = np.sin(robot.heading_c)
                
                
                global_obs_bound = []
                for point in bound:
                    c_x = point[0]*COS - point[1]*SIN + wp_x_start
                    c_y = point[0]*SIN + point[1]*COS + wp_y_start
                    global_obs_bound.append([c_x, c_y, (horz_dist)/num_pts, vert_dist/num_pts])

                #Update obstacles
                global_planner.update_obs(obs_cir=obs_cir, obs_bound=global_obs_bound, obs_rec=[], obstacles=[])
                planner.update_obs(obs_cir=obs_cir, obs_bound=[], obs_rec=[], obstacles=[])
                
                #-----------------------------------------------------
                
                #Get next Goal
                
                goal, localGridIndex = find_next_goal(localGrid_info=localGrid_info, current_state=(wp_x_start, wp_y_start), heading_c=robot.heading_c)
                
                #-----------------------------------------------------
                
                #Check for global replanning
                
                z, _ = model.predict(global_planner.path)
                index = np.where(z > tread_max_height)
                index = index[0]
                print("global Trajectory nodes that intersect the obstacles", index)
                
                replanning = False
                if len(index) > 0:
                    replanning = True
                    replanning_local = True
                else:
                    #Check if current trajetory crosses any obstacles
                    for i in range(len((global_planner.path_vertex)) - 1):
                        if global_planner.utils.is_collision(global_planner.path_vertex[i], global_planner.path_vertex[i + 1]):
                            print("global Trajectory intersect the obstacles inbetween nodes")
                            replanning = True
                            replanning_local = True
                            break
                
                if len(goal) >0:
                    goal = np.atleast_2d(goal)
                    if np.linalg.norm([goal[:,0] - robot.wp_c_x, goal[:,1] - robot.wp_c_y]) > goal_radius_local:
                        replanning = True
                        replanning_local = True


                replanning_count = 0
                while replanning:
                    print("Replanning Global Trajectory!")
                    global_planner.reset_tree(start=(wp_x_start, wp_y_start))
                    global_planner.plan()
                    assign_subgoals(localGrid_info=localGrid_info, global_path=global_planner.path, grid_size=(num_step_local*digit_step_len))
                    goal, localGridIndex = find_next_goal(localGrid_info=localGrid_info, current_state=(wp_x_start, wp_y_start), heading_c=robot.heading_c)
                    goal = np.atleast_2d(goal)
                    # global_planner.utils.delta = max(0.1, global_planner.utils.delta - 0.05)
                    replanning_count += 1
                    
                    if (np.linalg.norm([global_goal[0] - robot.state[0], global_goal[1] - robot.state[1]]) <= global_planner.goal_radius + 3*num_step_local*digit_step_len):
                        goal = global_goal
                        print("Replanning Next Local Trajectory!")
                        
                        planner.reset_tree(start=(wp_x_start, wp_y_start), goal=tuple(np.array(goal).flatten()))
                        planner.plan(heading_c = robot.state[2])
                        robot.update_new_path(model, planner.path)
                        break
                        
                    
                    if replanning_count > 5:
                        raise TimeoutError
                        
                    if len(goal[0]) == 0:
                        continue
                    
                    print("Next subgoal: ", goal)
                    
                    # #locally plan toward next subgoal
                    
                    
                    print("Replanning Next Local Trajectory!")
                    
                    planner.reset_tree(start=(wp_x_start, wp_y_start), goal=tuple(np.array(goal).flatten()))
                    planner.plan(heading_c = robot.state[2])
                    robot.update_new_path(model, planner.path)
                    
                    #--------------------
                    
                    # print("Replanning Local Trajectory!")
                    # planner.reset_tree(start=(wp_x_start, wp_y_start), goal=tuple(np.array(goal).flatten()))
            
                    # all_path = []
                    # all_vertex = []
                    # all_path_vertex = []
                    # all_errors = []
                    # cost = 500
                    
                    # i = 0
                    # for _ in range(n_path_gen):
                    #     planner.plan(heading_c = robot.state[2], obstacle_margin=obstacles_margin_local)        
                    #     all_path.append(planner.path)
                    #     all_vertex.append(planner.vertex)
                    #     all_path_vertex.append(planner.path_vertex)
                    #     # print(planner.path)
                        
                    #     entropies, uncertainties = path_evaluator.evaluate_path(model=model, trajectory=planner.path, heading_c=robot.heading_c, current_stance=robot.stance)
                    #     print("Path " + str(i) + " :")
                    #     print("Information Entropy: ", entropies)
                    #     print("PSP square error: ", uncertainties)
                    #     print("----------------------------------")
                        
                    #     # all_errors.append(uncertainties)
                        
                    #     new_cost =  1.2 * np.sum(normalize_value(uncertainties, 'error')) +  0.5* np.sum(normalize_value(entropies, 'entropy'))
                    #     new_cost = 1000 #call Path Evaluator Here
                    #     if new_cost < cost:
                    #         low_cost_path = []
                    #         low_cost_vertex = []
                    #         low_cost_path_vertex = []
                        
                    #     planner.goal_sample_rate = min(0.35, planner.goal_sample_rate - 0.1)
                    #     planner.reset_tree() 
                        
                    #     i += 1
                        
                    # #Randomly Choose for now
                    # # visualizer.plot_path_candidate(all_path, localGrid_info, localGridIndex)
                    # # visualizer.pause()
                    
                    # random_index = random.randint(0, len(all_path) - 1)
                    # print("Choose path[" + str(random_index) + "] out of " + str(len(all_path)))
                    # low_cost_path = all_path[random_index] #DELETE THIS LINE LATER
                    # low_cost_vertex = all_vertex[random_index] #DELETE THIS LINE LATER
                    # low_cost_path_vertex = all_path_vertex[random_index] #DELETE THIS LINE LATER
                    # uncertainties = all_errors[random_index]
                                    
                    #---------  
                    
                    #TRACK PATHS

                    # planner.track_path(low_cost_path, low_cost_vertex, low_cost_path_vertex)
                    # robot.update_new_path(model, low_cost_path) #make sure robot has the same path, generated by the local planner
                        
                    
                    #-------------------
                    
                    
                    if len(planner.path) < 3:
                        continue
                    
                    break
                
                global_planner.utils.delta = obstacles_margin_global

            #---------------------------------

            visualizer.clear()
            visualization(visualizer, evaluator, final_goal=global_goal)
            visualizer.plot_PSP_traj(cfg, robot, planner, global_planner, localGridIndex, localGrid_info)
            visualizer.pause()
        

        # END OF WHILE LOOP (REACH GOAL)

        #---------------------------------
        
        print("Global Goal Reached!")
        print("Done!")
        
    except TimeoutError:
        print("Digit may need to stop and turn around. Smooth and Continuous Phase Space Planning might not be feasible at current location.")
    
    except UnboundLocalError:
        print(traceback.format_exc())
        print("PSP fails to converge!")
        
    except Exception as e:
        print(traceback.format_exc())
        print("Failed!")
        print("Current State", (robot.apex_x, robot.apex_y, np.rad2deg(robot.heading_c)))
        
        print("Current Waypoint",  (robot.wp_c_x, robot.wp_c_x))
        z, std = model.predict([robot.wp_c_x, robot.wp_c_x])
        print("Predicted mean at current waypoint", z)
        print("Predicted std at current waypoint", std)
        print("Next Waypoint",  (robot.wp_n_x, robot.wp_n_x))
        z, std = model.predict([robot.wp_n_x, robot.wp_n_x])
        print("Predicted mean at next waypoint", z)
        print("Predicted  std at next waypoint", std)
        
    
    #Record Data
    if log_data_flag:
        column_names = ["apex_x", "apex_y", "apex_z", "dl_switch", "ds_switch", "step_length", "step_width", "step_time", "dheading", "heading_n", \
                "t1", "t2", "opt_vn", "foot_x", "foot_y", "wp_x_n", "wp_y_n", "wp_z_n", "foot_z", "dz", "s_switch_g", "l_switch_g"]
        np.savetxt(log_dir, np.column_stack((log_data[:,0], 
                                            log_data[:,1], 
                                            log_data[:,2],
                                            log_data[:,3],
                                            log_data[:,4],
                                            log_data[:,5],
                                            log_data[:,6],
                                            log_data[:,7],
                                            log_data[:,8],
                                            log_data[:,9],
                                            log_data[:,10],
                                            log_data[:,11],
                                            log_data[:,12],
                                            log_data[:,13],
                                            log_data[:,14],
                                            log_data[:,15],
                                            log_data[:,16],
                                            log_data[:,17],
                                            log_data[:,18],
                                            log_data[:,19],
                                            log_data[:,20],
                                            log_data[:,21])), delimiter=',', fmt='%s', header=','.join(column_names), comments='')
        
        print(f'Data has been stored in {log_dir}')

    # column_names = ["apex_x", "apex_y", "apex_z", "dl_switch", "ds_switch", "step_length", "step_width", "step_time", "dheading", "heading_n", \
    #         "t1", "t2", "opt_vn", "foot_x", "foot_y", "wp_x_n", "wp_y_n", "wp_z_n", "foot_z", "dz", "s_switch_g", "l_switch_g"]
        np.savetxt(log_PSP_dir, np.column_stack((log_data_PSP[:,0], 
                                            log_data_PSP[:,1], 
                                            log_data_PSP[:,2],
                                            log_data_PSP[:,3],
                                            log_data_PSP[:,4],
                                            log_data_PSP[:,5],
                                            log_data_PSP[:,6],
                                            log_data_PSP[:,7],
                                            log_data_PSP[:,8],
                                            log_data_PSP[:,9],
                                            log_data_PSP[:,10],
                                            log_data_PSP[:,11],
                                            log_data_PSP[:,12],
                                            log_data_PSP[:,13],
                                            log_data_PSP[:,14])), delimiter=' ', fmt='%s' ,comments='')

        print(f'Data has been stored in {log_PSP_dir}')
        
        
    
        column_names = ["decision_epoch", "time_elapsed", "total_abs_value","apex_x", "apex_y", "apex_z", "foot_x", "foot_y", "foot_z_psp", "foot_z_pred", "foot_z_var", "foot_z_actual", "wp_x_n", "wp_y_n", \
            "wp_z_psp", "wp_z_pred", "wp_z_var","wp_z_actual", "heading_n", "step_l", "dheading", "dz"]
        np.savetxt(log_GP_dir, np.column_stack((log_data_GP[:,0], 
                                                log_data_GP[:,1], 
                                                log_data_GP[:,2],
                                                log_data_GP[:,3],
                                                log_data_GP[:,4],
                                                log_data_GP[:,5],
                                                log_data_GP[:,6],
                                                log_data_GP[:,7],
                                                log_data_GP[:,8],
                                                log_data_GP[:,9],
                                                log_data_GP[:,10],
                                                log_data_GP[:,11],
                                                log_data_GP[:,12],
                                                log_data_GP[:,13],
                                                log_data_GP[:,14],
                                                log_data_GP[:,15],
                                                log_data_GP[:,16],
                                                log_data_GP[:,17],
                                                log_data_GP[:,18],
                                                log_data_GP[:,19],
                                                log_data_GP[:,20],
                                                log_data_GP[:,21])), delimiter=',', fmt='%s', header=','.join(column_names), comments='')

        print(f'Data has been stored in {log_GP_dir}')
        
        column_names = ["apex_x", "apex_y", "apex_z", "dl_switch", "ds_switch", "step_length", "step_width", "step_time", "dheading", "heading_n", \
                "t1", "t2", "opt_vn", "foot_x", "foot_y", "wp_x_n", "wp_y_n", "wp_z_n", "foot_z", "dz", "s_switch_g", "l_switch_g"]
        np.savetxt(log_dir, np.column_stack((log_data[:,0], 
                                            log_data[:,1], 
                                            log_data[:,2],
                                            log_data[:,3],
                                            log_data[:,4],
                                            log_data[:,5],
                                            log_data[:,6],
                                            log_data[:,7],
                                            log_data[:,8],
                                            log_data[:,9],
                                            log_data[:,10],
                                            log_data[:,11],
                                            log_data[:,12],
                                            log_data[:,13],
                                            log_data[:,14],
                                            log_data[:,15],
                                            log_data[:,16],
                                            log_data[:,17],
                                            log_data[:,18],
                                            log_data[:,19],
                                            log_data[:,20],
                                            log_data[:,21])), delimiter=',', fmt='%s', header=','.join(column_names), comments='')
        
        print(f'Data has been stored in {log_dir}')

    # column_names = ["apex_x", "apex_y", "apex_z", "dl_switch", "ds_switch", "step_length", "step_width", "step_time", "dheading", "heading_n", \
    #         "t1", "t2", "opt_vn", "foot_x", "foot_y", "wp_x_n", "wp_y_n", "wp_z_n", "foot_z", "dz", "s_switch_g", "l_switch_g"]
        np.savetxt(log_PSP_dir, np.column_stack((log_data_PSP[:,0], 
                                            log_data_PSP[:,1], 
                                            log_data_PSP[:,2],
                                            log_data_PSP[:,3],
                                            log_data_PSP[:,4],
                                            log_data_PSP[:,5],
                                            log_data_PSP[:,6],
                                            log_data_PSP[:,7],
                                            log_data_PSP[:,8],
                                            log_data_PSP[:,9],
                                            log_data_PSP[:,10],
                                            log_data_PSP[:,11],
                                            log_data_PSP[:,12],
                                            log_data_PSP[:,13],
                                            log_data_PSP[:,14])), delimiter=' ', fmt='%s' ,comments='')

        print(f'Data has been stored in {log_PSP_dir}')
        
        
    
        column_names = ["decision_epoch", "avg_std", "sum_std", "avg_error", "sum_error"]
        np.savetxt(log_eval_dir, np.column_stack((log_eval[:,0], 
                                                log_eval[:,1], 
                                                log_eval[:,2],
                                                log_eval[:,3],
                                                log_eval[:,4],)), delimiter=',', fmt='%s', header=','.join(column_names), comments='')

        print(f'Data has been stored in {log_eval_dir}')
        
        
    

    
    visualizer.clear()
    visualization(visualizer, evaluator, final_goal=global_goal)
    visualizer.plot_PSP_traj(cfg, robot, planner, global_planner, localGridIndex, localGrid_info)
    visualizer.pause()
    visualizer.show()


if __name__ == "__main__":
    main()
