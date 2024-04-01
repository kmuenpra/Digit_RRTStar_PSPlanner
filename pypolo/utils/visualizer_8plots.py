from copy import copy
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Arrow
from matplotlib import ticker as ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patches as patches
import time


plt.rcParams["image.origin"] = "lower"
plt.rcParams["image.cmap"] = "jet"
plt.rcParams["image.interpolation"] = "gaussian"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 10
plt.rcParams["axes.labelsize"] = 10
plt.rcParams["legend.fontsize"] = 8
plt.rcParams["xtick.labelsize"] = 8
plt.rcParams["ytick.labelsize"] = 8


class OOMFormatter(ticker.ScalarFormatter):
    def __init__(self, values, fformat="% 1.1f", offset=True, mathText=True):
        self.fformat = fformat
        self.oom = np.floor(np.log10(values.ptp()))
        super().__init__(
            useOffset=offset,
            useMathText=mathText,
        )

    def _set_order_of_magnitude(self):
        self.orderOfMagnitude = self.oom


class Visualizer:
    def __init__(self, env_extent, task_extent, interval):
        self.init_axes()
        self.env_extent = env_extent
        self.task_extent = task_extent
        self.env_rectangle = self._init_rectangle(env_extent, "black")
        self.task_rectangle = self._init_rectangle(task_extent, "gray")
        self.vmins = [None] * 4
        self.vmaxs = [None] * 4
        env_size = np.hypot(
            env_extent[1] - env_extent[0], env_extent[3] - env_extent[2]
        )
        self.arrow = None
        self.interval = interval
        
        self.apex0 = None
        self.apex1 = None
        self.foot1 = None
        self.apex2 = None
        self.apex3 = None
        self.apex4 = None
        self.foot4 = None
        self.traj4 = None

    def _init_rectangle(self, extent, color):
        x_min, x_max, y_min, y_max = extent
        return plt.Rectangle(
            (x_min, y_min),
            x_max - x_min,
            y_max - y_min,
            fill=False,
            edgecolor=color,
            linewidth=3,
            alpha=0.8,
        )

    def init_axes(self) -> None:
        #for jupiter Notebook
        # self.fig = plt.figure(figsize=(16, 6)
        # axes = self.fig.add_subplot(2,4,1)
        self.fig, axes = plt.subplots(2, 4, figsize=(16, 6), height_ratios=[2,1]) #height_ratios=[2,1]
        self.fig.subplots_adjust(
            top=0.95,
            bottom=0.1,
            left=0.1,
            right=0.9,
            hspace=0.1,
            wspace=0.2,
        )
        axes = np.asarray(axes).ravel()
        caxes = []
        for ax in axes[:4]:
            ax.axis("off")
            cax = make_axes_locatable(ax).append_axes(
                "right",
                size="5%",
                pad=0.05,
            )
            caxes.append(cax)
        caxes = np.asarray(caxes)
        self.axes = axes
        self.caxes = caxes

    def plot_image(self, index, matrix, title, vmin=None, vmax=None):
        ax, cax = self.axes[index], self.caxes[index]
        im = ax.imshow(
            matrix,
            extent=self.env_extent if index == 0 else self.task_extent,
            vmin=vmin,
            vmax=vmax,
        )
        plt.colorbar(im, cax, format=OOMFormatter(matrix))
        ax.add_patch(copy(self.env_rectangle))
        ax.add_patch(copy(self.task_rectangle))
        ax.set_xlim(self.env_extent[:2])
        ax.set_ylim(self.env_extent[2:])
        ax.set_title(title)

    def plot_prediction(self, mean, std, abs_error):
        self.plot_image(
            index=1,
            matrix=mean,
            title="Global Map (Mean)",
            vmin=self.vmins[1],
            vmax=self.vmaxs[1],
        )
        self.plot_image(
            index=2,
            matrix=std,
            title="Standard Deviation",#"Standard Deviation",
            vmin=self.vmins[2],
            vmax=self.vmaxs[2],
        )
        self.plot_image(
            index=3,
            matrix=abs_error,
            title="Absolute Error",
            vmin=self.vmins[3],
            vmax=self.vmaxs[3],
        )

    def plot_lengthscales(self, model, evaluator, min_lengthscale, max_lengthscale):
        self.axes[0].clear()
        self.axes[0].axis("off")
        self.caxes[0].clear()
        lenscale = model.get_ak_lengthscales(evaluator.eval_inputs).reshape(
            *evaluator.eval_grid
        )
        self.plot_image(
            index=0,
            matrix=lenscale,
            title="Lengthscales",
            vmin=min_lengthscale,
            vmax=max_lengthscale,
        )

    def plot_metrics(self, evaluator):
        # self.axes[4].plot(evaluator.num_samples, evaluator.smses)
        # self.axes[4].set_title("Standardized Mean Squared Error")
        # self.axes[4].set_xlabel("Number of Samples")
        # self.axes[4].grid("on")
        # self.axes[4].yaxis.set_major_formatter(
        #     ticker.FuncFormatter(lambda x, _: "% .2f" % x)
        # )
        # self.axes[4].set_ylim([0, None])
        
        # self.axes[5].plot(evaluator.num_samples, evaluator.mslls)
        # self.axes[5].set_title("Mean Standardized Log Loss")
        # self.axes[5].set_xlabel("Number of Samples")
        # self.axes[5].set_ylim([None, 0])
        # self.axes[5].grid("on")
        # self.axes[5].yaxis.set_major_formatter(
        #     ticker.FuncFormatter(lambda x, _: "% .2f" % x)
        # )

        # self.axes[6].plot(evaluator.num_samples, evaluator.training_times)
        # self.axes[6].yaxis.set_major_formatter(
        #     ticker.FuncFormatter(lambda x, _: "% .2f" % x)
        # )
        # self.axes[6].set_xlabel("Number of Samples")
        # self.axes[6].grid("on")
        # self.axes[6].set_title("Training Time (Seconds)")



        self.axes[7].plot(evaluator.num_samples, evaluator.prediction_times)
        self.axes[7].yaxis.set_major_formatter(
            ticker.FuncFormatter(lambda x, _: "% .2f" % x)
        )
        self.axes[7].set_xlabel("Number of Samples")
        self.axes[7].grid("on")
        self.axes[7].set_title("Prediction Time (Seconds)")


    def plot_data(self, x_train: np.ndarray, final_goal, alpha=0.1):
        self.axes[2].plot(x_train[:, 0], x_train[:, 1], ".", color="k", alpha=alpha)
        self.axes[1].plot(
            final_goal[0],
            final_goal[1],
            "*",
            color="red",
            markersize=20,
            alpha=0.8,
        )

    def plot_inducing_inputs(self, x_inducing: np.ndarray):
        self.axes[3].plot(x_inducing[:, 0], x_inducing[:, 1], "+", color="w", alpha=0.9)

    def plot_robot(self, state: np.ndarray, scale: float = 1.0):
        if self.arrow is not None:
            self.arrow.remove()
        self.arrow = Arrow(
            state[0],
            state[1],
            scale * np.cos(state[2]),
            scale * np.sin(state[2]),
            width=scale,
            color="white",
            alpha=0.8,
            label="Digit"
        )
        self.axes[1].add_patch(self.arrow)

    def plot_goal(self, goal: np.ndarray, final_goal: np.ndarray):
        
        # k_grids = localGrid_info["k_grids"]
        # for i in range(k_grids):
        #     # print(localGrid_info["local_goal"][i].shape)
        #     if np.array(localGrid_info["local_goal"][i]).shape[0] > 0:
                
        #         goal = np.atleast_2d(localGrid_info["local_goal"][i])
        #         self.axes[1].plot(
        #             goal[:, 0],
        #             goal[:, 1],
        #             "*",
        #             color="yellow",
        #             markersize=5,
        #             alpha=0.8,
        #             label = "subgoal"
        #         )
        
        self.axes[1].plot(
            goal[0],
            goal[1],
            "*",
            color="yellow",
            markersize=10,
            alpha=0.8,
        )
        
        self.axes[1].plot(
            final_goal[0],
            final_goal[1],
            "*",
            color="red",
            markersize=20,
            alpha=0.8,
            label='Global Goal'
        )
        
        # self.axes[1].legend()
        
    #--------------- ADDED --------------------------
    
    def save_plot_svg(self, de):
        plt.savefig("Decision_epoch_" + str(de) + ".svg", format="svg")
        
    def plot_obstacles(self, planner):
        
        for (ox, oy, w, h) in planner.obs_boundary:
            self.axes[4].add_patch(
                patches.Rectangle(
                    (ox, oy), w, h,
                    edgecolor='black',
                    facecolor='black',
                    fill=True
                )
            )
    
        
        for (ox, oy, r) in planner.obs_circle:
            self.axes[4].add_patch(
                patches.Circle(
                    (ox, oy), r,
                    edgecolor='black',
                    facecolor='red', alpha=0.3
                )
        )
            
        for (ox, oy, r) in planner.obs_circle:
            self.axes[4].add_patch(
                patches.Circle(
                    (ox, oy), (r+planner.utils.delta),
                    edgecolor='black',
                    facecolor='blue', alpha=0.1
                )
        )
            
        for (ox, oy, r) in planner.obs_circle:
            self.axes[5].add_patch(
                patches.Circle(
                    (ox, oy), r,
                    edgecolor='black',
                    facecolor='red', alpha=0.3
                )
        )
        
        
        for (ox, oy, r) in planner.obs_circle:
            self.axes[1].add_patch(
                patches.Circle(
                    (ox, oy), r,
                    edgecolor='black',
                    facecolor='red', alpha=0.7
                )
        )

    def plot_path_candidate(self, path_list, localGrid_info, localGridIndex):
        
       
        plot_axes_limit = localGrid_info["bound_x"][localGridIndex] + localGrid_info["bound_y"][localGridIndex]
        self.axes[5].clear()
        self.axes[5].grid("on")
        # self.axes[5].set_xlim(plot_axes_limit[0], plot_axes_limit[1])
        # self.axes[5].set_ylim(plot_axes_limit[2], plot_axes_limit[3])
        for path in path_list:
            path = np.array(path)
            self.axes[5].plot(path[:(len(path) - 1),0], path[:(len(path) - 1),1], "-")
        # self.axes[5].relim()      # make sure all the data fits
        self.axes[5].autoscale()
        time.sleep(0.5)
      
        
            
    def plot_PSP_traj(self, cfg, robot, planner, global_planner, localGridIndex, localGrid_info, end_flag=False):
        
        global_goal = (global_planner.s_goal.x, global_planner.s_goal.y)
    
        # #-----------------------------------
        
        all_sag = np.array([])
        all_lat = np.array([])
        body_x_axis = np.array([])
        body_y_axis = np.array([])

        xhat = np.array([0.07, 0])
        yhat = np.array([0, 0.07])

        for i in range(len(robot.history["sagittal"])):
            
            sag = robot.history["sagittal"][i]
            lat = robot.history["lateral"][i]
            wp_x, wp_y, heading = robot.history["frame"][i]
            
            sag_global = np.array(sag) *np.cos(heading) - np.array(lat) *np.sin(heading) + wp_x
            lat_global = np.array(sag) *np.sin(heading) + np.array(lat) *np.cos(heading) + wp_y
            
            all_sag = np.append(all_sag,sag_global)
            all_lat = np.append(all_lat,lat_global)
            
            # Rotation matrix
            rotation = np.array([[np.cos(heading), -np.sin(heading)], 
                                 [np.sin(heading),  np.cos(heading)]])
            
            if i == 0:
                body_x_axis = np.append(body_x_axis, rotation.dot(xhat))
                body_y_axis = np.append(body_y_axis, rotation.dot(yhat))
            else:
                body_x_axis = np.vstack([body_x_axis, rotation.dot(xhat)])
                body_y_axis = np.vstack([body_y_axis, rotation.dot(yhat)])
                
        all_wp_x = np.array(planner.path)[:,0]
        all_wp_y = np.array(planner.path)[:,1]

       
        
        self.axes[4].set_title("PSP Trajectory")
        self.axes[4].plot(all_wp_x[:(len(all_wp_x))],all_wp_y[:(len(all_wp_y))], "m--")
        self.axes[4].plot(all_wp_x[:(len(all_wp_x))],all_wp_y[:(len(all_wp_y))], "bo", label="Desired waypoint")
        self.axes[4].plot(all_wp_x[-1],all_wp_y[-1],"*",color="magenta",markersize=10)
        self.traj4 = self.axes[4].plot(all_sag, all_lat, label="trajectory")
        self.apex4 = self.axes[4].plot(np.array(robot.history["apex_state"])[:,0], np.array(robot.history["apex_state"])[:,1], "or", label="return apex state")
        self.foot4 = self.axes[4].plot(np.array(robot.history["foot_position"])[:,0], np.array(robot.history["foot_position"])[:,1], "*", label="foot position")
        
        self.axes[4].plot(
            global_goal[0],
            global_goal[1],
            "*",
            color="red",
            markersize=20,
            alpha=0.8,
        )


        # Draw body frame axes
        # if len(all_wp_x) == len(body_x_axis):
        #     for i in range(len(body_x_axis)):
        #         self.axes[4].arrow(all_wp_x[i], all_wp_y[i],  *body_x_axis[i], head_width=0.01, head_length=0.01, fc='g', ec='g')
        #         self.axes[4].arrow(all_wp_x[i], all_wp_y[i],  *body_y_axis[i], head_width=0.01, head_length=0.01, fc='g', ec='g')
        #         self.axes[4].arrow(all_wp_x[i], all_wp_y[i],  *-4*body_y_axis[i], head_width=0.001, head_length=0.001, fc='k', ec='k', alpha=0.1)
        #         self.axes[4].arrow(all_wp_x[i], all_wp_y[i],  *4*body_y_axis[i], head_width=0.001, head_length=0.001, fc='k', ec='k', alpha=0.1)
        self.axes[4].set_xlim(np.array(robot.history["apex_state"])[-1, 0] - 2, np.array(robot.history["apex_state"])[-1, 0] + 2) #0.4 margin usually
        self.axes[4].set_ylim(np.array(robot.history["apex_state"])[-1, 1] - 2, np.array(robot.history["apex_state"])[-1, 1] + 2) #0.4 margin usually
        self.axes[4].grid()
        self.axes[4].legend()
        
        #-------------------------------------------
        
        num_grid_x, num_grid_y = localGrid_info["(num_grid_x, num_grid_y)"]
        plot_axes_limit = localGrid_info["bound_x"][localGridIndex] + localGrid_info["bound_y"][localGridIndex]
        i_current, j_current = (localGridIndex % (num_grid_y)), int(localGridIndex / (num_grid_x)) 
        
        self.axes[5].set_title("Local Map, Current-Grid : (" + str(i_current) + ", " + str(j_current) + ")")
        self.axes[5].plot(all_wp_x,all_wp_y, "bo", label="Desired waypoint")
        self.axes[5].plot(all_sag, all_lat, label="trajectory")
        self.axes[5].plot(planner.path[-1][0], planner.path[-1][1],"*",color="magenta",markersize=10)
        self.axes[5].plot(np.array(robot.history["apex_state"])[:,0], np.array(robot.history["apex_state"])[:,1], "or", label="return apex state")
        self.axes[5].set_xlim(plot_axes_limit[0] - 1, plot_axes_limit[1] + 1)
        self.axes[5].set_ylim(plot_axes_limit[2] - 1, plot_axes_limit[3] + 1)
        self.axes[5].grid("on")
        
        self.axes[5].plot(
            global_goal[0],
            global_goal[1],
            "*",
            color="red",
            markersize=20,
            alpha=0.8,
        )
        
      
        
        #-------------------------------------------
        
        #Plot trajectory on Global Map
        self.plot_obstacles(planner=global_planner)
        self.apex0 = self.axes[0].plot(np.array(robot.history["apex_state"])[:,0], np.array(robot.history["apex_state"])[:,1], "ro", markersize=2)
        self.apex1 = self.axes[1].plot(np.array(robot.history["foot_position"])[:,0], np.array(robot.history["foot_position"])[:,1], "y*",markersize=2)
        self.foot1 = self.axes[1].plot(np.array(robot.history["apex_state"])[:,0], np.array(robot.history["apex_state"])[:,1], "ro", markersize=2)
        self.apex2 = self.axes[2].plot(np.array(robot.history["apex_state"])[:,0], np.array(robot.history["apex_state"])[:,1], "ro", markersize=2)
        self.apex3 = self.axes[3].plot(np.array(robot.history["apex_state"])[:,0], np.array(robot.history["apex_state"])[:,1], "ro", markersize=2)
        
        #-------------------------------------------
        
        k_grids = localGrid_info["k_grids"]
        for i in range(k_grids):
        
            bound_x = localGrid_info["bound_x"][i]
            bound_y = localGrid_info["bound_y"][i]
                        
            self.axes[1].axvline(x=bound_x[0], color='gray', linestyle='--', linewidth=0.1)
            self.axes[1].axvline(x=bound_x[1], color='gray', linestyle='--', linewidth=0.1)
            self.axes[1].axhline(y=bound_y[1], color='gray', linestyle='--', linewidth=0.1)
            self.axes[1].axhline(y=bound_y[0], color='gray', linestyle='--', linewidth=0.1)
            
    #------------------------------------------------
   
    
        
        self.axes[6].set_title("Sub-Goal : " + str(localGrid_info["local_goal"][localGridIndex]))
        if len(global_planner.path) != 0:
            self.axes[6].plot([x[0] for x in global_planner.path], [x[1] for x in global_planner.path], '-k', linewidth=1)
        
        k_grids = localGrid_info["k_grids"]
        for i in range(k_grids):
            # print(localGrid_info["local_goal"][i].shape)
            if np.array(localGrid_info["local_goal"][i]).shape[0] > 0:
                
                goal = np.atleast_2d(localGrid_info["local_goal"][i])
                self.axes[6].scatter(goal[:,0], goal[:,1], marker='*', color='m')
                
            bound_x = localGrid_info["bound_x"][i]
            bound_y = localGrid_info["bound_y"][i]
                        
            self.axes[6].axvline(x=bound_x[0], color='gray', linestyle='--', linewidth=0.1)
            self.axes[6].axvline(x=bound_x[1], color='gray', linestyle='--', linewidth=0.1)
            self.axes[6].axhline(y=bound_y[1], color='gray', linestyle='--', linewidth=0.1)
            self.axes[6].axhline(y=bound_y[0], color='gray', linestyle='--', linewidth=0.1)
            self.axes[6].set_xlim(cfg.env_extent[0], cfg.env_extent[1])
            self.axes[6].set_ylim(cfg.env_extent[2], cfg.env_extent[3])
            
        x_min, x_max = localGrid_info["bound_x"][localGridIndex]
        y_min, y_max = localGrid_info["bound_y"][localGridIndex]
        square_x = [x_min, x_max, x_max, x_min, x_min]
        square_y = [y_min, y_min, y_max, y_max, y_min]
        # self.axes[6].plot(square_x, square_y)
        self.axes[6].fill(square_x, square_y, 'green', alpha=0.3)
        
            # self.axes[6].pause(0.01)
        # self.axes[6].show()
        
   
    
    #------------------------------------------------

    def plot_title(self, decision_epoch, time_elapsed):
        self.fig.suptitle(
            f"RRT* Phase Space Planner | "
            + f"Decision Epoch: {decision_epoch} | "
            + f"Time Elapsed: {time_elapsed:.2f} Seconds."
        )

    def clear(self):
        for i in range(1, 4):
            self.axes[i].clear()
            self.axes[i].axis("off")
            self.caxes[i].clear()
        
        
      
        for i in range(4, 8):
            self.axes[i].clear()
      

    def pause(self, interval=1e-3):
        plt.gcf().canvas.mpl_connect(
            "key_release_event",
            lambda event: [exit(0) if event.key == "escape" else None],
        )
        plt.pause(interval)

    def flash(self):
        self.pause()
        self.clear()

    def show(self):
        plt.show()