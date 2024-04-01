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
    
    #-------------------------------------------------------------------
    
    #Grid-wise Sample
    x_min, x_max, y_min, y_max = cfg.task_extent
    x_grid = np.linspace(x_min, x_max, 10)
    y_grid = np.linspace(y_min, y_max, 10)
    xx, yy = np.meshgrid(x_grid, y_grid)
    x_init = np.column_stack((xx.flatten(), yy.flatten()))
    y_init = sensor.get(x_init[:, 0], x_init[:, 1]).reshape(-1, 1)
    
    #-------------------------------------------------------------------
    
    print(f"Collected {len(x_init)} samples in pilot survey.")

    return x_init, y_init


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
    else:
        raise ValueError(f"Unknown kernel: {cfg.kernel}")
    if cfg.model.name == "GPR":
        model = pypolo.models.gp.GPRModel(
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


def model_update(num_steps, model, evaluator):
    # print("Optimization...")
    start = time()
    losses = model.optimize(num_steps=num_steps) #calculate lossses
    end = time()
    evaluator.training_times.append(end - start)
    evaluator.losses.extend(losses)


def evaluation(model, evaluator):
    # print(f"Prediction...")
    start = time()
    mean, std = model.predict(evaluator.eval_inputs)
    end = time()
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
    
    
def information_gathering(model, robot, planner, global_goal,  num_step=1, samples_per_dt=10, visualizer=None, log_data=np.empty((0,22)), log_data_flag=False, verbose=True):
    
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
        
        # while robot.has_goals:
            
        
        plot_counter += 1
        
        if log_data_flag:
            
            replanning_local, PSP_output = robot.step(model, num_targets=samples_per_dt, log_data_flag=log_data_flag, verbose=verbose)
            log_data = np.vstack([log_data, PSP_output])
            
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
            return x_new, y_new, log_data, replanning_local
        else:
            return x_new, y_new, replanning_local
        
    else:
        
        if log_data_flag:
            return [], [], log_data, replanning_local
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
    
def assign_subgoals(localGrid_info, global_path):
    
    #Get param
    # global_start = localGrid_info["global_start"]
    global_goal = localGrid_info["global_goal"]
    k_grids = localGrid_info["k_grids"]
    
    #Clear old data
    # localGrid_info["local_data"] = []
    localGrid_info["local_goal"] = []
    
    for _ in range(k_grids):
        # localGrid_info["local_data"].append(np.array([]))
        localGrid_info["local_goal"].append(np.array([]))
        
    
    #Assgin global waypoint to the different subgrid
    for subgoal in global_path:
        
        #Skip starting position
        # if np.linalg.norm([subgoal[0] - global_start[0], subgoal[1] - global_start[1]]) < 0.0001: 
        #     continue
        
        index = locate_state(localGrid_info, subgoal)
                
        if localGrid_info["local_goal"][index].shape[0] == 0:
            localGrid_info["local_goal"][index] = np.atleast_2d(np.append(localGrid_info["local_goal"][index] , subgoal))
        else:
            localGrid_info["local_goal"][index] = np.atleast_2d(np.vstack([localGrid_info["local_goal"][index] , subgoal]))
                    
                    
    #Only keep the subgoal closest to the global goal
    for i in range(k_grids):
        if localGrid_info["local_goal"][i].shape[0] < 2:
            if localGrid_info["local_goal"][i].shape[0] == 1 and np.linalg.norm(localGrid_info["local_goal"][i] - global_goal, axis=1) > 0.00001:
                localGrid_info["local_goal"][i] = []
            continue
        
        # index_subgoal = [global_path.index(subgoal) for subgoal in list(localGrid_info["local_goal"][i])]
        # localGrid_info["local_goal"][i] = localGrid_info["local_goal"][i][-1]
        
            
        # print(i)
        
        distances = np.linalg.norm(localGrid_info["local_goal"][i] - global_goal, axis=1)
        min_distance_index = np.argmin(distances)
        
        localGrid_info["local_goal"][i] = localGrid_info["local_goal"][i][min_distance_index]
    
    
        # randomly_selected_subgol = random.choice(localGrid_info["local_goal"][i])
        # localGrid_info["local_goal"][i] = randomly_selected_subgol
        
        
        # print("Assign Goal: ", localGrid_info["local_goal"][i][min_distance_index])
        # print("To Grid Index: ", i)
        
       
            
def find_next_goal(localGrid_info, current_state, heading_c):
 
    num_grid_x, num_grid_y = localGrid_info["(num_grid_x, num_grid_y)"]
    
    #Convert to matrix indices
    current_index = locate_state(localGrid_info, current_state)
    i, j = (current_index % (num_grid_y)), int(current_index / (num_grid_x)) 
    print("current state:", current_state)
    print("current index: ", current_index)
    print("current grid: ", (i,j))
    
    #Clear current goal
    localGrid_info["local_goal"][current_index] = []
    
    adjacent_indices = []
    if i > 0:
        adjacent_indices.append((i-1,j))
        if j > 0:
            adjacent_indices.append((i-1,j-1))
            
    if i+1 < (num_grid_y - 1):
        adjacent_indices.append((i+1,j))
        if j+1 < (num_grid_x - 1):
            adjacent_indices.append((i+1,j+1))
        
    if j > 0:
        adjacent_indices.append((i,j-1))
        if i+1 < (num_grid_y - 1):
            adjacent_indices.append((i+1,j-1))
        
    if j+1 < (num_grid_x - 1):
        adjacent_indices.append((i,j+1))
        if i > 0:
            adjacent_indices.append((i-1,j+1))
            
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
            # current_state = np.atleast_2d(current_state)
            
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
    log_dir = os.path.join(home_path, 'logs/psp-hyperparam-' + log_time + ".csv")
    
    log_data = np.empty((0,22))
    log_data_flag = True
    
    #---------------------------------

    # ALL PARAMS
    
    #Environment Properties
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)
    # env_path = home_path + '/raw_data/terrains/Kinskuch_River.png'
    # env_path = home_path + '/raw_data/terrains/n30_w106_1arc_v3.jpg'
    # env_path = home_path + '/raw_data/terrains/Powell_Lake.png'
    # env_path = home_path + '/raw_data/terrains/n40_w116_1arc_v3.jpg'
    # env_path = home_path + '/raw_data/terrains/n24w102.jpg'
    # env_path = home_path + '/raw_data/terrains/n16e102.jpg'
    env_path = home_path + '/raw_data/terrains/n17e102.png'

    max_height = 0.5
    
    #Starting Position                     Kinskuch               Powell
    global_start = (-6,-6)#(-5.5,5.4) #(9,-9)  # (-9,-9)     #(-11, 10.7) # (-8,6.5) #(-9, -9) #
    global_goal =  (6,6)#(-6, -8)    #(-3,6)  # (8,7)     # (-12, -16)  #(7,-3)  #(-1.5 2)
    heading_start = np.deg2rad(0)
    
    #Sensor config    
    scan_radius = 4 #How far can LIDAR sensor see [meter]
    samples_per_dt = 3 # How many samples is collcted at each PSP step
    
    # Digit's constrain
    digit_step_len = 0.1 #0.1 #This Step length will remain throughout the planning
    digit_max_turn_angle = np.deg2rad(5)
    tread_max_height = 0.15 #Digit's Maximum treadable Terrain Height (should be less than environment's max_height)
    
    #Local Planner config
    goal_sample_rate_local = 0.65
    iter_max_local = 2000
    obstacles_margin_local = 0.1
    goal_radius_local = 0.2
    n_path_gen = 1 #Number of paths generated to be evaluated in each grid
    
    #Global Planner config
    goal_sample_rate_global = 0.65
    iter_max_global = 20000
    obstacles_margin_global = 0.1
    goal_radius_global = 0.5  #0.5
    step_len_global  = 0.3 #0.3
    
    
    #Environment Grid config
    num_step_local = 15#15 #Expected Number of Step in each local grid
    
    #Update other config
    PSP_num_steps = 25 #How many step of PSPlanner before updating the visual plots
    cfg.env_extent = [-10.0, 10.0, -10.0, 10.0]
    cfg.noise_scale = 0.03 * max_height
    cfg.planner.name = "LocalRRTStar"
    cfg.task_extent = [cfg.env_extent[0] + 0.1, cfg.env_extent[1] - 0.1, cfg.env_extent[2] + 0.1, cfg.env_extent[3] - 0.1 ]
    cfg.eval_grid = [50,50]


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
    assign_subgoals(localGrid_info, global_planner.path)

    
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
    evaluator.add_data(x_init, y_init) #set training data
    
    #Define GP Model
    model = get_model(cfg, x_init, y_init, x_scaler, y_scaler)
    model_update(cfg.num_train_steps, model, evaluator)
    evaluation(model, evaluator)

    # #Collect training data from the terrain at starting point
    x_at_start, y_at_start, _ = robot.sensor.sense( model, robot.state, robot.heading_c, ray_tracing = False, num_targets = 200)

    model.add_data(x_at_start, y_at_start)
    evaluator.add_data(x_at_start, y_at_start) #set training data
    model_update(cfg.num_train_steps, model, evaluator)
    evaluation(model, evaluator)
    
    #---------------------------------
    
    #Define the obstacles in terrain, if found at the starting point
    
    #Replan Global Trajectory
    mean, std = model.predict(evaluator.eval_inputs)
    
    #Define risk region
    obs_cir = []
    # risk_radius = 0.15

    risk_index = np.where(mean > tread_max_height)[0].tolist()
    # risk_radius = np.max(std[risk_index], axis=0) * 2
    # print(std[risk_index])
    for x,y,risk_radius in np.hstack([evaluator.eval_inputs[risk_index].tolist(), std[risk_index]]):
    # for x,y in evaluator.eval_inputs[risk_index].tolist():
        obs_cir.append([x,y, max(0.15,2*risk_radius)])
                
        
    #Define Heading constraint for the global planner
    COS = np.cos(robot.heading_c)
    SIN = np.sin(robot.heading_c)
    
    jitter = (1*10**(-1))
    num_pts = 30
    n_turns = int((np.pi/2)/digit_max_turn_angle)
    
    horz_dist = 0
    vert_dist = 0
    for i in range(n_turns + 2):
        horz_dist += digit_step_len*np.cos(i*digit_max_turn_angle)
        vert_dist += digit_step_len*np.sin(i*digit_max_turn_angle)
        
    line_constr = np.sqrt(((jitter + horz_dist)**2 + vert_dist**2))
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
        global_obs_bound.append([c_x, c_y, 0.06, 0.06])
        
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
    
    while True:
        
        #Replan global planner again for any new obstacles
        global_planner.reset_tree()
        global_planner.plan(heading_c=robot.heading_c)
        print("Initialize Global Trajectory: ")
        assign_subgoals(localGrid_info=localGrid_info, global_path=global_planner.path)
        # print("Global Plan", global_planner.path)
        print(global_planner.path)
        for i in range(len((global_planner.path_vertex)) - 1):
            if global_planner.utils.is_collision(global_planner.path_vertex[i], global_planner.path_vertex[i + 1]):
                print("Yup")
                print(global_planner.path_vertex[i].x)
                print(global_planner.path_vertex[i].y)


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
    visualizer.plot_path_candidate(all_path, localGrid_info, localGridIndex)
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
            
            verbose =False
            replanning_local = False
            
            if log_data_flag:
                x_new, y_new, log_data, replanning_local = information_gathering(model, robot, planner, global_goal, num_step=PSP_num_steps, samples_per_dt=samples_per_dt, \
                                                            visualizer=visualizer, log_data=log_data, log_data_flag=log_data_flag, verbose=verbose)
                
            else:
                
                x_new, y_new, replanning_local = information_gathering(model, robot, planner, global_goal, num_step=PSP_num_steps, samples_per_dt=samples_per_dt, \
                                                            visualizer=visualizer, log_data=log_data, log_data_flag=log_data_flag, verbose=verbose)
            
            
            if len(robot.path) < 3:
                x_new, y_new, _ = robot.sensor.sense( model, robot.state, robot.heading_c, ray_tracing = True, num_targets = 5)
                

            # if len(x_new) > 0:
            #     print("--- For all collected data ---")
            #     z_new, std_new = model.predict(x_new)
            #     print("Newdata : highest std", np.max(std_new))
            #     index = np.where(std_new ==  np.max(std_new))[0].flatten()
            #     print("@point : ", x_new[index])
            #     print("")
                
            #     print("Newdata : lowest std", np.min(std_new))
            #     index = np.where(std_new ==  np.min(std_new))[0].flatten()
            #     print("@point : ", x_new[index])
            #     print("")
                
            #     pred_error = abs(y_new - z_new) 
            #     index = np.where(pred_error ==  np.max(pred_error))[0].flatten()
            #     print("Newdata : max error", np.max(pred_error, axis=0))
            #     print("@point : ", x_new[index])
            #     print("")
                
            #     index = np.where(pred_error ==  np.min(pred_error))[0].flatten()
            #     print("Newdata : min error", np.min(pred_error, axis=0))
            #     print("@point : ", x_new[index])
            #     print("")
            
            # if decision_epoch > 8:
            #     PSP_num_steps = 6
            #     samples_per_dt = 10
            
            if (len(x_new) > 0 and len(y_new) > 0):
                
                if x_collect is None and y_collect is None:
                    x_collect = x_new
                    y_collect = y_new
                else:
                    x_collect = np.vstack([x_collect, x_new])
                    y_collect = np.vstack([y_collect, y_new])
                    
            #Update terrain model if: these conditions met
            if (decision_epoch <= 15 and  x_collect is not None) or replanning_local:
                
                print("Updating the terrain GP model with new data. \n")
                
                evaluator.add_data(x_collect, y_collect)
                model.add_data(x_collect, y_collect)
                model_update(cfg.num_train_steps, model, evaluator)
                evaluation(model, evaluator)
                x_collect = None
                y_collect = None
                    
            elif (decision_epoch > 15  and  (x_collect is not None) and len(x_collect) > 100) or replanning_local:
                
                print("Updating the terrain GP model with new data. \n")
                samples_per_dt = 3
                evaluator.add_data(x_collect, y_collect)
                model.add_data(x_collect, y_collect)
                model_update(cfg.num_train_steps, model, evaluator)
                evaluation(model, evaluator)
                x_collect = None
                y_collect = None                

            # GLOBAL GOAL CHECK (AFTER STEPPING)
            
            # print("distance from global goal: ", np.linalg.norm([global_goal[0] - robot.wp_c_x, global_goal[1] - robot.wp_c_y]))
            if np.linalg.norm([global_goal[0] - robot.wp_c_x, global_goal[1] - robot.wp_c_y]) <= global_planner.goal_radius:
                break
            
            # if np.linalg.norm([global_goal[0] - robot.wp_c_x, global_goal[1] - robot.wp_c_y]) <= 1.5*global_planner.goal_radius:
            #     planner.goal_radius = 0.15
                
             #---------------------------------
                
            # visualizer.clear()
            # visualization(visualizer, evaluator, final_goal=global_goal)
            # visualizer.plot_PSP_traj(cfg, robot, planner, global_planner, localGridIndex, localGrid_info)
            # visualizer.pause()
        
            
            #---------------------------------
            
            # REPLANNING CHECK
            
            eps = 0.0005
            if np.linalg.norm([robot.wp_n_x - robot.wp_c_x, robot.wp_n_y - robot.wp_c_y]) < eps:
                
                
                print("Local Subgoal reached!")
                replanning_local = True
            
                #Update the waypoint list of path planner
                print("goal", goal)
                print("Shape", goal.shape)
                try:
                    current_goal_index = global_planner.path.index(list(goal.flatten()))
                    
                except ValueError:
                    
                    print(global_planner.path)
                    print("END")
                    return
                
                print("Current Goal:", [goal, current_goal_index])
                global_planner.path = global_planner.path[(current_goal_index + 1):]
                assign_subgoals(localGrid_info=localGrid_info, global_path=global_planner.path)
                localGrid_info["local_goal"][localGridIndex] = []
                
                if current_goal_index != len(global_planner.path_vertex) - 1:
                    global_planner.path_vertex = global_planner.path_vertex[current_goal_index:]
                    
            
            
                
                
                    #PRINT THE MODEL ERROR
                    # index_diff = len(planner.path) - len(robot.history["apex_state"][(-len(planner.path)):])
                    # p = planner.path[:len(planner.path) - index_diff]
                    # q = robot.history["apex_state"][(-len(planner.path)):]
                    # theta = robot.history["heading"][(-len(planner.path)):]
                    
                    # #Deviation in body frame (dx_b should almost be zero)
                    # dev_global = np.array(p) - np.array(q)
                    
                    # dev_local = []
                    # for i in range(len(theta)):
                    #     dx_b =  np.cos(theta[i])*dev_global[i,0] + np.sin(theta[i])*dev_global[i,1]
                    #     dy_b = -np.sin(theta[i])*dev_global[i,0] + np.cos(theta[i])*dev_global[i,1]
                    #     dev_local.append(dy_b)

                    # dev_local = np.array(dev_local)
                    
                    # change_in_dev = []
                    # for i in range(len(dev_local) - 1):
                    #     change_in_dev.append(dev_local[i+1] - dev_local[i])
                        
                    # change_in_dev = np.array(change_in_dev)

                    # print("Bigger than what", np.max(abs(dx_b),axis=0))
                    # print(len(dy_b))
                    
                    # dev_global = np.array(p) - np.array(q)
                    # print("dev_global",dev_global)
                    # change_in_dev = []
                    # for i in range(len(dev_global) - 1):
                    #     change_in_dev.append([dev_global[i+1, :] - dev_global[i, :]])
                    
                    # change_in_dev = np.array(change_in_dev)     
                    # print("change_in_dev",change_in_dev)     
                    # print("Actual Change in Deviation", change_in_dev.reshape(-1,1))  
                    # print("Predicted Change in Deviation", uncertainties.reshape(-1,1))    
                    # print("Actual Sum Erorr: ", np.sum(change_in_dev, axis=0))
                    # print("Predicted Sum Error: ", np.sum(uncertainties, axis=0))
                    # print("Actual Norm Error", np.linalg.norm(np.sum(change_in_dev, axis=0)))
                    # print("Predicted Norm Error: ", np.linalg.norm(np.sum(uncertainties, axis=0)))
                    
                    
                    
            #-----------------------------------
            
            # if local goal is reached,
            # --> update the global planner and assign new subgoal to local planner
            
            # Define new starting waypoint + track apex state
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
            
            # wp_x_start = robot.apex_x
            # wp_y_start = robot.apex_y
                
            print("current waypoint", (wp_x_start, wp_y_start))
            
            #-----------------------------------
            
            #Update obstacles in terrain based on new information
            mean, std = model.predict(evaluator.eval_inputs)
            
            #Define risk region
            obs_cir = []
            risk_index = np.where(mean > tread_max_height)[0].tolist()
            # print("std[risk]", std[risk_index])
            # risk_radius = np.max(std[risk_index], axis=0) * 2
            for x,y,risk_radius in np.hstack([evaluator.eval_inputs[risk_index].tolist(), std[risk_index]]):
            # for x,y in evaluator.eval_inputs[risk_index].tolist():
                obs_cir.append([x,y,max(0.15,2*risk_radius)])
                
            
                
            #Define Heading constraint for the global planner
            COS = np.cos(robot.heading_c)
            SIN = np.sin(robot.heading_c)
            
            
            # angle_constr = np.deg2rad(65)
            # bound1 = generate_constraint_points(angle_constr,  num_pts, abs(3*digit_step_len/(num_pts*np.cos(angle_constr))), perturb_x)
            # bound2 = generate_constraint_points(-angle_constr, num_pts, abs(3*digit_step_len/(num_pts*np.cos(angle_constr))), perturb_x)
            
            # perturb_x = -jitter
            # bound1 = generate_constraint_points(angle_constr,  num_pts, line_constr/num_pts, perturb_x)
            # bound2 = generate_constraint_points(-angle_constr, num_pts, line_constr/num_pts, perturb_x)
            # bound = bound1 + bound2
            
            global_obs_bound = []
            for point in bound:
                c_x = point[0]*COS - point[1]*SIN + wp_x_start
                c_y = point[0]*SIN + point[1]*COS + wp_y_start
                global_obs_bound.append([c_x, c_y, 0.06, 0.06])

            #Update obstacles
            global_planner.update_obs(obs_cir=obs_cir, obs_bound=global_obs_bound, obs_rec=[], obstacles=[])
            planner.update_obs(obs_cir=obs_cir, obs_bound=[], obs_rec=[], obstacles=[])
            
            #-----------------------------------------------------
            
            
            #Get Next Goal:
            goal, localGridIndex = find_next_goal(localGrid_info=localGrid_info, current_state=(wp_x_start, wp_y_start), heading_c=robot.heading_c)
            goal = np.atleast_2d(goal)
            print("Next subgoal before replanning: ", goal) 
                    
            #Check if global path has risky Terrain elevation
            # print("Global Path:" ,global_planner.path)
            z, _ = model.predict(global_planner.path)
            index = np.where(z > tread_max_height)
            index = index[0]
            print("global Trajectory node that intersect the obstacles", index)
            
            replanning = False
            if len(index) > 0 or len(goal[0]) == 0:
                replanning = True
                replanning_local = True
            else:
                #Check if current trajetory crosses any obstacles
                for i in range(len((global_planner.path_vertex)) - 1):
                    if global_planner.utils.is_collision(global_planner.path_vertex[i], global_planner.path_vertex[i + 1]):
                        print("global Trajectory that intersect the obstacles inbetween node")
                        replanning = True
                        replanning_local = True
                        break
                    
            replan_count = 0  
            while replanning:
            
                replan_count += 1
                
                if replan_count > 5:
                    visualizer.clear()
                    visualization(visualizer, evaluator, final_goal=global_goal)
                    visualizer.plot_PSP_traj(cfg, robot, planner, global_planner, localGridIndex, localGrid_info)
                    visualizer.pause()
                    raise TimeoutError
                
                print("Replanning Global Trajectory!")
                global_planner.reset_tree(start=(wp_x_start, wp_y_start))
                global_planner.plan(heading_c=robot.heading_c)
                assign_subgoals(localGrid_info=localGrid_info, global_path=global_planner.path)
                
                #locally plan toward next subgoal
                goal, localGridIndex = find_next_goal(localGrid_info=localGrid_info, current_state=(wp_x_start, wp_y_start), heading_c=robot.heading_c)
                goal = np.atleast_2d(goal)
                
                #if nearby subgoal is not found
                if len(goal[0]) == 0:
                    
                    global_planner.goal_sample_rate = min(0.35, global_planner.goal_sample_rate - 0.1)
                    # global_planner.utils.delta = min(0.1, global_planner.utils.delta - 0.05)
                    # planner.utils.delta = min(0.1, planner.utils.delta - 0.05)
                    # global_planner.utils.delta = min(0.45, global_planner.utils.delta - 0.1)
                    # planner.utils.delta = min(0.45, planner.utils.delta - 0.1)
                    
                    print("*------------------------------------------------------*")
                    print("     Replanning Event Triggered     ")
                    print("   > Couldn't find nearby goal based on current position")
                    print("*------------------------------------------------------*")
                    continue
                
                print("Found new subgoal.")
                
                
            
                print("Next subgoal: ", goal)                 
                print("distance to next goal", np.linalg.norm([goal[:,0] - wp_x_start, goal[:,1] - wp_y_start]))     
                
                break           
                    
                #----------------------------
            
            while np.linalg.norm([goal[:,0] - wp_x_start, goal[:,1] - wp_y_start]) <= goal_radius_local + digit_step_len \
                and np.linalg.norm([global_goal[0] - wp_x_start, global_goal[1] - wp_y_start]) > global_planner.goal_radius + jitter:
                
                print("Current local goal is too close, move onto the next one.")
                
                #locally plan toward next subgoal
                goal, localGridIndex = find_next_goal(localGrid_info=localGrid_info, current_state=goal, heading_c=robot.heading_c)
                goal = np.atleast_2d(goal)
                
                if len(goal[0]) == 0:
                    goal = np.atleast_2d(global_goal)
                    break
                
                
                # # print("local path: ", planner.path)
                # global_planner.goal_sample_rate = min(0.35, global_planner.goal_sample_rate - 0.1)
                # global_planner.utils.delta = min(0.001, global_planner.utils.delta - 0.05)
                # planner.utils.delta = min(0.001, planner.utils.delta - 0.05)
                # print("*---------------------------------*")
                # print("     Replanning Event Triggered     ")
                # print("   > Local Path only have one step! Digit may be stucked.")
                # print("*---------------------------------*")
                    
                    # planner.utils.delta = min(0.01, planner.utils.delta - 0.05)
    
                
                # print("Replanning Local Trajectory!")
                # planner.reset_tree(start=(wp_x_start, wp_y_start), goal=tuple(np.array(goal).flatten()))
                # planner.plan(heading_c = robot.state[2])
                
                # if len(planner.path) < 3:
                #     print("local path: ", planner.path)
                #     global_planner.goal_sample_rate = min(0.35, global_planner.goal_sample_rate - 0.1)
                #     global_planner.utils.delta = min(0.001, global_planner.utils.delta - 0.05)
                #     planner.utils.delta = min(0.001, planner.utils.delta - 0.05)
                #     print("*---------------------------------*")
                #     print("     Replanning Even Triggered     ")
                #     print("   > Local Path only have one step! Digit may be stucked.")
                #     print("*---------------------------------*")
                #     continue
                                    
            
            # entropies = 0.5 * np.log(2 * np.pi * np.square(std)) + 0.5
            # print("max Entropy", np.max(entropies, axis=0))
            # print("min Entropy", np.min(entropies, axis=0) )
            
            # GENERATE MULTIPLE PATHS for EVALUATION
            if replanning_local:
                print("Replanning Local Trajectory!")
                planner.reset_tree(start=(wp_x_start, wp_y_start), goal=tuple(np.array(goal).flatten()))
        
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
                    
                    entropies, uncertainties = path_evaluator.evaluate_path(model=model, trajectory=planner.path, heading_c=robot.heading_c, current_stance=robot.stance)
                    print("Path " + str(i) + " :")
                    print("Information Entropy: ", entropies)
                    print("PSP square error: ", uncertainties)
                    print("----------------------------------")
                    
                    all_errors.append(uncertainties)
                    

                    new_cost = 1000 #call Path Evaluator Here
                    if new_cost < cost:
                        low_cost_path = []
                        low_cost_vertex = []
                        low_cost_path_vertex = []
                    
                    planner.goal_sample_rate = min(0.35, planner.goal_sample_rate - 0.1)
                    planner.reset_tree() 
                    
                    i += 1
                    
                #Randomly Choose for now
                # visualizer.plot_path_candidate(all_path, localGrid_info, localGridIndex)
                # visualizer.pause()
                
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
                    
                    # if len(planner.path) < 3:
                    
                    #     print("Path is too short, Digit might be stucked.")
                    #     global_planner.goal_sample_rate = min(0.35, global_planner.goal_sample_rate - 0.1)
                    #     global_planner.utils.delta = min(0.7, global_planner.utils.delta - 0.05)
                    #     planner.utils.delta = min(0.3, planner.utils.delta - 0.05)
                    #     continue
                    
                    # wp_x_start_global = goal[0,0]
                    # wp_y_start_global = goal[0,1]
                    
                
                
            
            # print("--- Evaluate entropy along next local waypoints ---")
        
            # z_new, std_new = model.predict(planner.path)
            # entropy = 0.5 * np.log(2 * np.pi * np.square(std_new)) + 0.5
            # print("Waypts : highest mean", np.max(z_new))
            # index = np.where(z_new ==  np.max(z_new))[0].flatten()
            # print("@point : ", planner.path[index[0]])
            # print("")
            
            # print("Waypts : lowest mean", np.min(z_new))
            # index = np.where(z_new ==  np.min(z_new))[0].flatten()
            # print("@point : ", planner.path[index[0]])
            # print("")
            
            # print("Waypts : highest entropy", np.max(entropy))
            # index = np.where(entropy ==  np.max(entropy))[0].flatten()
            # print("@point : ", planner.path[index[0]])
            # print("")
            
            # print("Waypts : lowest entropy", np.min(entropy))
            # index = np.where(entropy ==  np.min(entropy))[0].flatten()
            # print("@point : ", planner.path[index[0]])
            # print("")
            
            # print("Local Path", np.array(planner.path).reshape(-1,2))
            # print("predicted mean", z_new)
            # print("predicted std", std_new)
            # print("predicted entropy", entropy)
            
                            
            #Reset Goal Sample Rate
            global_planner.goal_sample_rate = goal_sample_rate_global
            global_planner.utils.delta = obstacles_margin_global
            
            # planner.goal_sample_rate = goal_sample_rate_local
            # planner.utils.delta = obstacles_margin_local
                
            # END OF IF STATEMENT (REPLANNING CHECK)

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
    
    
    
    # if log_data_flag:
    # # column_names = ["apex_x", "apex_y", "apex_z", "dl_switch", "ds_switch", "step_length", "step_width", "step_time", "dheading", "heading_n", \
    # #         "t1", "t2", "opt_vn", "foot_x", "foot_y", "wp_x_n", "wp_y_n", "wp_z_n", "foot_z", "dz", "s_switch_g", "l_switch_g"]
    #     np.savetxt(log_dir, np.column_stack((log_data[:,0], 
    #                                         log_data[:,1], 
    #                                         log_data[:,2],
    #                                         log_data[:,3],
    #                                         log_data[:,4],
    #                                         log_data[:,5],
    #                                         log_data[:,6],
    #                                         log_data[:,7],
    #                                         log_data[:,8],
    #                                         log_data[:,9],
    #                                         log_data[:,10],
    #                                         log_data[:,11],
    #                                         log_data[:,12],
    #                                         log_data[:,13],
    #                                         log_data[:,14])), delimiter=',', fmt='%s' ,comments='')

    #     print(f'Data has been stored in {log_dir}')
    
    visualizer.clear()
    visualization(visualizer, evaluator, final_goal=global_goal)
    visualizer.plot_PSP_traj(cfg, robot, planner, global_planner, localGridIndex, localGrid_info)
    visualizer.pause()
    visualizer.show()


if __name__ == "__main__":
    main()
