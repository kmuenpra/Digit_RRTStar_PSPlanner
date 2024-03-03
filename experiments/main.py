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


def get_environment(cfg):
    with np.load("./data/n44w111.npz") as data:
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
    sensor = pypolo.sensors.PointSensor(
        matrix=env,
        env_extent=cfg.env_extent,
        rate=cfg.sensing_rate,
        noise_scale=cfg.noise_scale,
        rng=rng,
    )
    print(
        f"Initialized sensor with rate {cfg.sensing_rate} and noise scale {cfg.noise_scale}."
    )
    return sensor


def get_robot(cfg, sensor):
    robot = pypolo.robots.DiffDriveRobot(
        sensor=sensor,
        state=np.array([cfg.task_extent[1], cfg.task_extent[2], -np.pi]),
        control_rate=cfg.control_rate,
        max_lin_vel=cfg.max_lin_vel,
        max_ang_vel=cfg.max_ang_vel,
        goal_radius=cfg.goal_radius,
    )
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
    bezier_planner = pypolo.planners.BezierPlanner(cfg.task_extent, rng)
    goals = bezier_planner.plan(num_points=cfg.num_bezier_points)
    robot.goals = goals
    while len(robot.goals) > 0:
        robot.step()
    x_init, y_init = robot.commit_samples()
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


def get_planner(cfg, rng):
    if cfg.planner.name == "MaxEntropy":
        planner = pypolo.planners.MaxEntropyPlanner(
            cfg.task_extent, rng, cfg.planner.num_candidates
        )
    else:
        raise ValueError(f"Unknown planner: {cfg.planner.name}")
    print(f"Initialized planner {cfg.planner.name}.")
    return planner


def model_update(num_steps, model, evaluator):
    # print("Optimization...")
    start = time()
    losses = model.optimize(num_steps=num_steps)
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


def visualization(visualizer, evaluator, x_inducing=None):
    # print(f"Visualization...")
    visualizer.plot_prediction(evaluator.mean, evaluator.std, evaluator.abs_error)
    visualizer.plot_data(evaluator.x_train)
    if x_inducing is not None:
        print("Plotting inducing inputs...")
        visualizer.plot_inducing_inputs(x_inducing)
    visualizer.plot_metrics(evaluator)


def information_gathering(robot, model, planner, visualizer):
    while True:
        # print("Planning...")
        goal = planner.plan(model, robot.state[:2])
        visualizer.plot_goal(goal)
        robot.goals = goal
        visualizer.pause()
        plot_counter = 0
        # print("Sampling...")
        while robot.has_goals:
            plot_counter += 1
            robot.step()
            if visualizer.interval > 0 and plot_counter % visualizer.interval == 0:
                visualizer.plot_robot(robot.state)
                visualizer.pause()
        if len(robot.sampled_observations) > 0:
            x_new, y_new = robot.commit_samples()
            return x_new, y_new


@hydra.main(version_base=None, config_path="configs", config_name="main")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)
    rng = set_random_seed(cfg.seed)
    env = get_environment(cfg)
    visualizer = get_visualizer(cfg, env)
    sensor = get_sensor(cfg, env, rng)
    evaluator = pypolo.utils.Evaluator(sensor, cfg.task_extent, cfg.eval_grid)
    robot = get_robot(cfg, sensor)
    planner = get_planner(cfg, rng)
    # AK requires normalization of inputs and outputs.
    # Make sure x_init and y_init are representative of the environment.
    x_init, y_init = pilot_survey(cfg, robot, rng)
    x_scaler = MinMaxScaler()
    x_scaler.fit(x_init)
    y_scaler = StandardScaler()
    y_scaler.fit(y_init)
    evaluator.add_data(x_init, y_init)
    model = get_model(cfg, x_init, y_init, x_scaler, y_scaler)
    # First optimization takes longer time, which affects the training time evaluation.
    # To avoid this, we run one optimization before the first evaluation.
    model.optimize(num_steps=1)
    model_update(cfg.num_train_steps, model, evaluator)
    evaluation(model, evaluator)
    visualization(visualizer, evaluator)
    visualizer.pause()

    decision_epoch = 0
    start_time = time()
    while len(evaluator.y_train) < cfg.max_num_samples:
        time_elapsed = time() - start_time
        decision_epoch += 1
        visualizer.plot_title(decision_epoch, time_elapsed)
        x_new, y_new = information_gathering(robot, model, planner, visualizer)
        evaluator.add_data(x_new, y_new)
        model.add_data(x_new, y_new)
        model_update(cfg.num_train_steps, model, evaluator)
        evaluation(model, evaluator)
        visualizer.clear()
        if cfg.kernel.name == "AK":
            visualizer.plot_lengthscales(
                model, evaluator, cfg.kernel.min_lengthscale, cfg.kernel.max_lengthscale
            )
        visualization(visualizer, evaluator)
        visualizer.pause()
    print("Done!")
    visualizer.show()


if __name__ == "__main__":
    main()
