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
        max_distance = 2,
        perception_angle=90,
    )
    
    print(
        f"Initialized sensor with rate {cfg.sensing_rate} and noise scale {cfg.noise_scale}."
    )
    return sensor


def get_robot(cfg, sensor, robot_start):
    
    robot = pypolo.robots.DigitRobot(
        sensor=sensor,
        state=robot_start,
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


def pilot_survey(cfg, robot, rng):
    
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
    
    #Randomly Sample
    x_min, x_max, y_min, y_max = cfg.task_extent
    x_grid = np.linspace(x_min, x_max, 10)
    y_grid = np.linspace(y_min, y_max, 10)
    xx, yy = np.meshgrid(x_grid, y_grid)
    x_init = np.column_stack((xx.flatten(), yy.flatten()))
    y_init = robot.sensor.get(x_init[:, 0], x_init[:, 1]).reshape(-1, 1)
    
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


def get_planner(cfg, rng, robot, goal=None, obstacles=None):
    if cfg.planner.name == "MaxEntropy":
        planner = pypolo.planners.MaxEntropyPlanner(
            cfg.task_extent, rng, cfg.planner.num_candidates
        )
    elif cfg.planner.name == "AStar":
        
        start = robot.state[:2]
        planner = pypolo.planners.AStarPlanner(
            cfg.task_extent, rng, start, goal, obstacles
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
    
    print("Run information_gathering")
    
    final_goal = planner.goal
    
    while True:

        goal = planner.get_next_goal(robot, num_step)
        print("goal:", goal)
        # print("path", planner.path)
        # print("current", (robot.wp_c_x, robot.wp_c_y))
    
        visualizer.plot_goal(goal, final_goal)
        robot.goals = goal
        visualizer.pause()
        plot_counter = 0
        
        print("Sampling...")
        while robot.has_goals:
            
            print("state: " ,robot.state)
            plot_counter += 1
            
            # robot.step()
            
            robot.step(model, num_targets = samples_per_dt)
                        
            if visualizer.interval > 0 and plot_counter % visualizer.interval == 0:
                visualizer.plot_robot(robot.state)
                visualizer.pause()
        if len(robot.sampled_observations) > 0:
            # print("Append Sample Elevation: ", robot.sampled_observations)
            # print("corresponding Sample Location: ", robot.sampled_locations)
            x_new, y_new = robot.commit_samples()
            print("End information_gathering")
            return x_new, y_new


@hydra.main(version_base=None, config_path="configs", config_name="main")
def main(cfg: DictConfig) -> None:
    
    #Update config
    max_height = 0.37
    max_z_digit = 0.3
    max_uncertainty = 0.2
    cfg.noise_scale = 0.1 * max_height
    cfg.planner.name = "AStar"
    cfg.goal_radius = 0.2
    
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)
    env_path = '/home/kmuenpra/Desktop/terrains/terrain_n38_w087_1arc_v2.png'

    rng = set_random_seed(cfg.seed)
    env = get_environment_from_image(cfg, png_path=env_path, resize=(50,50), max_height=max_height)
    visualizer = get_visualizer(cfg, env)
    sensor = get_sensor(cfg, env, rng)
    evaluator = pypolo.utils.Evaluator(sensor, cfg.task_extent, cfg.eval_grid)
    
    
    #Define start and goal location
    robot_start = np.array([0,0,0]) 
    goal = np.array([5,5])
    obstacles = np.array([])

    robot =  get_robot(cfg, sensor, robot_start)
    planner =  get_planner(cfg, rng, robot, goal=goal, obstacles=obstacles)

    #Generate a trajectory toward goal
    planner.plan(robot)

    #Only works for Astart planner for now
    robot.track_planner(planner)
    
    # AK requires normalization of inputs and outputs.
    # Make sure x_init and y_init are representative of the environment.
    x_init, y_init = pilot_survey(cfg, robot, rng)
    
    #Add additional data measured at the start
    x_at_start, y_at_start = sensor.sense(states=robot.state[:2], global_heading=robot.state[2])
    x_init = np.vstack([x_init, x_at_start])
    y_init = np.vstack([y_init, y_at_start])
    
    x_scaler = MinMaxScaler()
    x_scaler.fit(x_init)
    y_scaler = StandardScaler()
    y_scaler.fit(y_init)
    evaluator.add_data(x_init, y_init)
    
    #Add untraversable terrain to planner map
    risk_index = np.where(y_init > max_z_digit)[0]    

    #Replan
    if not risk_index.shape[0] == 0:
        print(x_init[risk_index].shape)
        print("New risk region found")
        planner.add_obstacle(x_init[risk_index])
        planner.plan(robot)
        robot.track_planner(planner)
        
    model = get_model(cfg, x_init, y_init, x_scaler, y_scaler)
    # First optimization takes longer time, which affects the training time evaluation.
    # To avoid this, we run one optimization before the first evaluation.
    model.optimize(num_steps=1)
    model_update(cfg.num_train_steps, model, evaluator)
    
    #Evaluator and Visualizer
    evaluation(model, evaluator)
    visualization(visualizer, evaluator, goal)
    visualizer.pause()
    
    
    #Hard Update State
    vapex = 1.5
    robot.hard_update_state(vapex=vapex)

    decision_epoch = 0
    start_time = time()
    
    end_flag = True
    
    while end_flag and (planner.distance(robot.state[:2], goal) > robot.goal_radius):#len(evaluator.y_train) < cfg.max_num_samples:
        time_elapsed = time() - start_time
        decision_epoch += 1
        visualizer.plot_title(decision_epoch, time_elapsed)
        # x_new, y_new = information_gathering(robot, model, planner, visualizer)
        # x_new, y_new = information_gathering(robot, model, planner, samples_per_dt=10, final_goal=goal, visualizer=visualizer)
        x_new, y_new = information_gathering(model, robot, planner, num_step=1, samples_per_dt=5, visualizer=visualizer)
        evaluator.add_data(x_new, y_new)
        model.add_data(x_new, y_new)
        model_update(cfg.num_train_steps, model, evaluator)
        evaluation(model, evaluator)

        #Add untraversable terrain to planner map
        risk_index = np.where(y_new > max_z_digit)[0]    

        #Replan
        if not risk_index.shape[0] == 0:
            print("New risk region found")
            planner.add_obstacle(x_new[risk_index])
            planner.plan(robot)
            
            if len(planner.path) < 2:
                print("End the Loop, infeasible trajectory")
                end_flag = False
                break
            
            robot.track_planner(planner)
                    
                # risk_index = np.unique(np.concatenate((arr,arr2),0))

        visualizer.clear()
        # if cfg.kernel.name == "AK":
        #     visualizer.plot_lengthscales(
        #         model, evaluator, cfg.kernel.min_lengthscale, cfg.kernel.max_lengthscale
        #     )
        visualization(visualizer, evaluator, goal)
        visualizer.plot_PSP_traj(robot, planner)
        visualizer.pause()
    
    print("Done!")
    visualizer.plot_PSP_traj(robot, planner, end_flag=True)
    visualizer.show()


if __name__ == "__main__":
    main()
