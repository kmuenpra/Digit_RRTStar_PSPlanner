from time import time
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import torch
import pypolo
import gpytorch
from pypolo.scalers import MinMaxScaler, StandardScaler
from PIL import Image



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


def get_sensor(cfg, env, rng):
    
    sensor = pypolo.sensors.LidarSensor(
        matrix=env,
        env_extent=cfg.env_extent,
        rate=cfg.sensing_rate,
        noise_scale=cfg.noise_scale,
        rng=rng,
        max_distance = 3, #https://community.robotshop.com/blog/show/lidar-light-amp-laser-based-distance-sensors
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
    
    # robot = pypolo.robots.DiffDriveRobot(
    #     sensor=sensor,
    #     state=np.array([cfg.task_extent[1], cfg.task_extent[2], -np.pi]),
    #     control_rate=cfg.control_rate,
    #     max_lin_vel=cfg.max_lin_vel,
    #     max_ang_vel=cfg.max_ang_vel,
    #     goal_radius=cfg.goal_radius,
    # )
    print(f"Initialized robot with control rate {cfg.control_rate}.")
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
    
    # Use DiffDriveRobot to survey around the terrain
    
    # bezier_planner = pypolo.planners.BezierPlanner(cfg.task_extent, rng)
    # goals = bezier_planner.plan(num_points=cfg.num_bezier_points)
    # robot.goals = goals
    
    # while len(robot.goals) > 0:
    #     robot.step_with_heading(current_heading=robot.state[2], num_targets = 3)
    #     # heading_c = np.arctan2(robot.state[1] - prev_state[1], robot.state[0] - prev_state[0])
    #     # prev_state = robot.state[:2]
        
    # x_init, y_init = robot.commit_samples()
    
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
            cfg.task_extent, rng, start, goal, step_len=step_len, goal_sample_rate=goal_sample_rate, search_radius=search_radius, iter_max=iter_max, max_turn_angle=max_turn_angle
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
    visualizer.plot_data(evaluator.x_train, final_goal=final_goal)
    if x_inducing is not None:
        print("Plotting inducing inputs...")
        visualizer.plot_inducing_inputs(x_inducing)
    visualizer.plot_metrics(evaluator)
    
    
def information_gathering(model, robot, planner, num_step=1, samples_per_dt=10, visualizer=None):
    print("Run " + str(num_step) + "-steps PSP")
    
    #Final local goal
    final_goal = np.array([planner.s_goal.x , planner.s_goal.y])
    
    iters = 0
    
    while iters < num_step and (np.linalg.norm([final_goal[0] - robot.wp_c_x, final_goal[1] - robot.wp_c_y]) > planner.goal_radius):
    
        robot.goals = np.atleast_2d([robot.wp_n_x, robot.wp_n_y]) #next waypoint
        print("goal:", robot.goals)
        
        visualizer.plot_goal(robot.goals, final_goal)        
        visualizer.pause()
        plot_counter = 0
        
        while robot.has_goals:
            
            print("state: " ,robot.state)
            plot_counter += 1
            
            robot.step(model, num_targets = samples_per_dt)
                        
            if visualizer.interval > 0 and plot_counter % visualizer.interval == 0:
                visualizer.plot_robot(robot.state)
                visualizer.pause()
                
        iters += 1
        
    if len(robot.sampled_observations) > 0:

        x_new, y_new = robot.commit_samples()
        print("End of " + str(num_step) + "-steps PSP")
        return x_new, y_new
    

# ----- Functions for allocating sub-goals -----


def local_grid_init(cfg, global_goal, digit_step_len, num_step_local):
    localGrid_info = {"center":[],
                "bound_x":[],
                "bound_y":[],
             "local_goal": [],
             "local_data":[],
             "all_centers":[],
             "k_grids":[],
             "(num_grid_x, num_grid_y)":[],
             "global_goal":[]
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
    
    return localGrid_info

def locate_state(localGrid_info ,state):
    
    #Get param
    k_grids = localGrid_info["k_grids"]
    
    state = np.atleast_2d(state)
    for index in range(k_grids):
        lb_x = localGrid_info["bound_x"][index][0]
        ub_x = localGrid_info["bound_x"][index][1]
        ub_y = localGrid_info["bound_y"][index][0]
        lb_y = localGrid_info["bound_y"][index][1]
        
        if (state[:,0] >= lb_x and state[:,0] <= ub_x) and (state[:,1] >= lb_y and state[:,1] <= ub_y):
            return index
    
def assign_subgoals(localGrid_info, global_path):
    
    #Get param
    global_goal = localGrid_info["global_goal"]
    k_grids = localGrid_info["k_grids"]
    
    #Clear old data
    localGrid_info["local_data"] = []
    localGrid_info["local_goal"] = []
    
    for _ in range(k_grids):
        localGrid_info["local_data"].append(np.array([]))
        localGrid_info["local_goal"].append(np.array([]))
        
    
    #Assgin global waypoint to the different subgrid
    for subgoal in global_path:

        index = locate_state(localGrid_info, subgoal)
                
        if localGrid_info["local_goal"][index].shape[0] == 0:
            localGrid_info["local_goal"][index] = np.atleast_2d(np.append(localGrid_info["local_goal"][index] , subgoal))
        else:
            localGrid_info["local_goal"][index] = np.atleast_2d(np.vstack([localGrid_info["local_goal"][index] , subgoal]))
                    
    #Only keep the subgoal closest to the global goal
    for i in range(k_grids):
        if localGrid_info["local_goal"][i].shape[0] < 2:
            continue
        # print(i)
        distances = np.linalg.norm(localGrid_info["local_goal"][i] - global_goal, axis=1)
        min_distance_index = np.argmin(distances)
        localGrid_info["local_goal"][i] = localGrid_info["local_goal"][i][min_distance_index]
            
def find_next_goal(localGrid_info, current_state):
    
    num_grid_x, num_grid_y = localGrid_info["(num_grid_x, num_grid_y)"]
    
    #Convert to matrix indices
    current_index = locate_state(localGrid_info, current_state)
    i, j = (current_index % (num_grid_y)), int(current_index / (num_grid_x)) 
    
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
        
    for i_next, j_next in adjacent_indices:
        #Convert back to vectorized list
        next_index = j_next * num_grid_x + i_next
        
        if len(localGrid_info["local_goal"][next_index]) > 0:
            
            goal = np.atleast_1d(localGrid_info["local_goal"][next_index]) 
            return goal, next_index


# ----- Functions for allocating sub-goals -----





'''
-------------
MAIN FUNCTION
-------------
'''

@hydra.main(version_base=None, config_path="configs", config_name="main")
def main(cfg: DictConfig) -> None:
    
    
    #Update config
    max_height = 0.37
    max_z_digit = 0.3
    max_uncertainty = 0.2
    cfg.noise_scale = 0.1 * max_height
    cfg.planner.name = "LocalRRTStar"
    cfg.task_extent = [cfg.env_extent[0] + 1, cfg.env_extent[1] - 1, cfg.env_extent[2] + 1, cfg.env_extent[3] - 1 ]
    # cfg.goal_radius = 0.2
    
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)
    env_path = '/home/kmuenpra/Desktop/terrains/terrain_n38_w087_1arc_v2.png'

    #---------------------------------
    
    #Set up environment, visualizer, evaluator, and sensors

    rng = set_random_seed(cfg.seed)
    env = get_environment_from_image(cfg, png_path=env_path, resize=(50,50), max_height=max_height)
    visualizer = get_visualizer(cfg, env)
    sensor = get_sensor(cfg, env, rng)
    evaluator = pypolo.utils.Evaluator(sensor, cfg.task_extent, cfg.eval_grid)
    
    #---------------------------------
    
    # GLOBAL PLANNER INIT.
    
    #Define Global Planner and Local grid tracker
    global_start = (-10,0)
    global_goal = (5,5)

    #Initialize Global Planner and generate global trajectory
    global_planner = pypolo.planners.GlobalRRTStar(
                    cfg.task_extent, rng, global_start, global_goal, step_len=0.5, goal_sample_rate=1, search_radius=0.6, iter_max=10000)
    global_planner.plan()
    
    #Define subgoal in each grid
    digit_step_len = 0.1
    num_step_local = 20
    localGrid_info = local_grid_init(cfg, global_goal, digit_step_len=digit_step_len, num_step_local=num_step_local)
    assign_subgoals(localGrid_info, global_planner.path)
    
    #---------------------------------
    
    # ROBOT INIT.
    
    #Define Digit robot
    start = np.array(global_start)
    robot =  get_robot(cfg, sensor, start, heading_start=0) 
    
    #---------------------------------
    
    # TERRAIN GP MODEL INIT.
    
    #Collect training data from pilot survey
    x_init, y_init =  pilot_survey(cfg, sensor, rng)

    x_scaler = MinMaxScaler()
    x_scaler.fit(x_init) #find min/max of (x1,x2) coordinates
    y_scaler = StandardScaler()
    y_scaler.fit(y_init) #find mean and std of y_init
    evaluator.add_data(x_init, y_init) #set training data
    
    #Define GP Model
    model = get_model(cfg, x_init, y_init, x_scaler, y_scaler)
    model_update(cfg.num_train_steps, model, evaluator)
    evaluation(model, evaluator)

    #Collect training data from the terrain at starting point
    x_at_start, y_at_start = robot.sensor.sense( robot.state, robot.heading_c, ray_tracing = True, num_targets = 30)

    model.add_data(x_at_start, y_at_start)
    evaluator.add_data(x_at_start, y_at_start) #set training data
    model_update(cfg.num_train_steps, model, evaluator)
    evaluation(model, evaluator)
    
    #---------------------------------
    
    #Define the obstacles in terrain, if found at the starting point
    #TODO:
    
    #---------------------------------
    
    #Replan global planner again for any new obstacles
    global_planner.goal_sample_rate = 0.6
    global_planner.utils.delta = 1
    global_planner.reset_tree()
    global_planner.plan()
    # global_planner.plot_map()
    assign_subgoals(localGrid_info=localGrid_info, global_path=global_planner.path)

    #---------------------------------
  
    # LOCAL PLANNER INIT.
    
    #Local Planner that robot tracks
    localGridIndex = locate_state(localGrid_info, robot.state[:2]) #Find which subgrid robot locates in
    plot_axes_limit = localGrid_info["bound_x"][localGridIndex] + localGrid_info["bound_y"][localGridIndex]
    
    goal = np.atleast_1d(localGrid_info["local_goal"][localGridIndex]) 
    planner =  get_planner(cfg, rng, start, goal=goal, \
                           step_len=digit_step_len, goal_sample_rate=0.6, search_radius=0.6, iter_max=10000, max_turn_angle=np.deg2rad(10))

    #Generate initial trajectory toward subgoal
    planner.plan(heading_c = robot.state[2])
    robot.update_new_path(model, planner.path) #make sure robot has the same path, generated by the local planner
    
     #---------------------------------

    visualization(visualizer, evaluator, final_goal=goal)
    visualizer.pause()

    decision_epoch = 0
    start_time = time()
    end_flag = True
    
    
    
    while end_flag and (np.linalg.norm([global_goal[0] - robot.state[0], global_goal[1] - robot.state[1]]) > global_planner.goal_radius):
        time_elapsed = time() - start_time
        decision_epoch += 1
        visualizer.plot_title(decision_epoch, time_elapsed)
        
        x_new, y_new = information_gathering(model, robot, planner, num_step=2, samples_per_dt=5, visualizer=visualizer)
        evaluator.add_data(x_new, y_new)
        model.add_data(x_new, y_new)
        model_update(cfg.num_train_steps, model, evaluator)
        evaluation(model, evaluator)
    
        
        #---------------------------------
        
        if np.linalg.norm([planner.s_goal.x - robot.wp_c_x, planner.s_goal.y - robot.wp_c_y]) <= planner.goal_radius:
            print("Local Subgoal reached!")
        
        
            # if local goal is reached,
            # --> update the global planner and assign new subgoal to local planner
            
            #Define the obstacles in terrain, if found at the starting point
            #TODO:
            
            
            #Define new starting waypoint
            robot.vapex = 0.1
            if robot.stance: #Right foot stance
                wp_x_start = robot.apex_x
                wp_y_start = robot.apex_y - robot.state_deviate
            else:
                wp_x_start = robot.apex_x
                wp_y_start = robot.apex_y + robot.state_deviate
                
            print("current waypoint", (wp_x_start, wp_y_start))
                
                
            print("Replanning Global Trajectory!")
            global_planner.reset_tree(start=(wp_x_start, wp_y_start))
            global_planner.plan()
            assign_subgoals(localGrid_info=localGrid_info, global_path=global_planner.path)
            
            
            #locally plan toward next subgoal
            print("Replanning Next Local Trajectory!")
            goal, localGridIndex = find_next_goal(localGrid_info=localGrid_info, current_state=(wp_x_start, wp_y_start))
            print("Next subgoal: ", goal)
            plot_axes_limit = localGrid_info["bound_x"][localGridIndex] + localGrid_info["bound_y"][localGridIndex]
            
            planner.reset_tree(start=(wp_x_start, wp_y_start), goal=tuple(np.array(goal).flatten()))
            planner.plan(heading_c = robot.state[2])
            robot.update_new_path(model, planner.path)

        #---------------------------------

        visualizer.clear()
        # if cfg.kernel.name == "AK":
        #     visualizer.plot_lengthscales(
        #         model, evaluator, cfg.kernel.min_lengthscale, cfg.kernel.max_lengthscale
        #     )
        visualization(visualizer, evaluator, final_goal=goal)
        visualizer.plot_PSP_traj(robot, planner, plot_axes_limit)
        visualizer.pause()
    
    print("Global Goal Reached!")
    print("Done!")
    # visualizer.plot_PSP_traj(robot, planner, end_flag=True)
    # visualizer.show()


if __name__ == "__main__":
    main()
