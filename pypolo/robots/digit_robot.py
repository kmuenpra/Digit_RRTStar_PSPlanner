import numpy as np
from typing import Tuple

from ..sensors import BaseSensor
from . import BaseRobot

import numpy as np
import math


class DigitRobot(BaseRobot):
        
    def __init__(self, sensor: BaseSensor, start: np.ndarray, control_rate: float, max_lin_vel: float, max_ang_vel: float, goal_radius: float) -> None:
        super().__init__(sensor, start, control_rate, max_lin_vel, max_ang_vel, goal_radius)
        
        self.state_deviate = 0.02
        self.foot_deviate = 0.03012
        
        #Initialize Apex State
        self.heading_c =   start[2]
        self.apex_x    =   start[0] - self.state_deviate*np.sin(self.heading_c)
        self.apex_y    =   start[1] + self.state_deviate*np.cos(self.heading_c)
        self.apex_z    =   1.01
        self.wp_c_x    =   start[0] 
        self.wp_c_y    =   start[1]
        self.wp_n_x    =   0.0
        self.wp_n_y    =   0.0   
        self.vapex     =   0.15 #0.25
        
        self.foot_x    =   self.apex_x - self.foot_deviate*np.sin(self.heading_c)
        self.foot_y    =   self.apex_y  + self.foot_deviate*np.cos(self.heading_c)
        self.foot_z    =   0.0      
        self.step_l , self.dheading, self.dz  = 0.0,0.0,0.0
        self.dz        =   0.0
        self.stance    =   1
        self.frame     =   (self.wp_c_x, self.wp_c_y, self.heading_c)
        
        self.history = {"apex_state":[np.array([self.apex_x, self.apex_y])],
                        "waypoint_track":[],
                        "foot_position":[np.array([self.foot_x,self.foot_y])],
                        "sagittal":[],
                        "lateral":[],
                        "frame":[],
                        "heading":[self.heading_c]}
        
        self.path = np.array([]) #Path to track
        
        #Update Robot State
        self.update_state()
        
        
        
    def update_new_path(self, model, path:list):
        #Update Apex State to keep track with the waypoints
        
        self.path = np.array(path)
        self.path = self.path[:len(self.path) - 1] #Remove the goal
        
        print("Local Path Tracked! (digit_robot.py): ", self.path)
        
        self.wp_c_x    =   self.path[0][0] 
        self.wp_c_y    =   self.path[0][1]
        
        if len(self.path) > 1:
            self.wp_n_x    =   self.path[1][0]   
            self.wp_n_y    =   self.path[1][1]       
            self.step_l , self.dheading, self.dz  = self.compute_action(model, (self.wp_n_x, self.wp_n_y), (self.wp_c_x, self.wp_c_y))
        else:
            self.wp_n_x    =   self.wp_c_x  
            self.wp_n_y    =   self.wp_c_y    
            self.step_l , self.dheading, self.dz = 0,0,0
        
        self.history["waypoint_track"] = []   
        self.history["waypoint_track"].append(np.array([self.wp_c_x, self.wp_c_y]))
    
    
    def compute_action(self, model, next, current):
        current = np.array(current)
        next = np.array(next)
        
        #this return a 2d prediction value
        z_foot_next, _ = model.predict(next)
        print("z_foot_next",z_foot_next)
        print("z_foot_next",z_foot_next.shape)
        
        #high level action
        dz = z_foot_next[0][0] - self.foot_z
        step_length = np.linalg.norm(next - current)
        dheading = np.arctan2(next[1] - current[1], next[0]- current[0]) - self.heading_c
        
        return step_length, dheading, dz

    def update_state(self) -> None:
        #Update Robot State
        self.state = np.hstack([self.apex_x, self.apex_y, self.heading_c])
        
        
    def step(self, model, num_targets, log_data_flag=False, verbose=True) -> None:
        
        if verbose:
            print("\n---------- Action Command ---------")
            print("A_HL: step_l", self.step_l)
            print("A_HL: dz", self.dz)
            print("A_HL: dtheta", np.rad2deg(self.dheading))
        
        #Track reference frame
        self.frame = (self.wp_c_x, self.wp_c_y, self.heading_c)
        self.history["frame"].append(self.frame)

        self.apex_x, self.apex_y, self.apex_z, dl_switch, ds_switch, step_length, step_width, step_time, dheading, heading_n, \
            t1, t2, opt_vn, self.foot_x, self.foot_y, wp_x_n, wp_y_n, wp_z_n, self.foot_z, dz, s_switch_g, l_switch_g, sag, lat = \
                apex_Vel_search(self.apex_x, self.apex_y, self.apex_z, self.wp_c_x, self.wp_c_y, self.vapex, self.foot_x, self.foot_y, \
                    self.foot_z, np.rad2deg(self.heading_c), self.step_l, np.rad2deg(self.dheading), self.dz, self.stance, stop = False, start_flag = True)

        #Update apex_vel
        self.vapex = opt_vn
        
        #Update heading 
        self.heading_c = np.deg2rad(heading_n)#self.heading_c + self.dheading 

        #Update Stance
        self.stance = 0 if self.stance else 1
        
        #Update Robot State
        self.update_state()
        
        #Update the current waypoint
        self.wp_c_x = self.wp_n_x
        self.wp_c_y = self.wp_n_y 
        
        #Append new information
        self.history["apex_state"].append(np.array([self.apex_x, self.apex_y]))
        self.history["waypoint_track"].append(np.array([self.wp_c_x, self.wp_c_y]))
        self.history["foot_position"].append(np.array([self.foot_x, self.foot_y]))
        self.history["sagittal"].append(sag)
        self.history["lateral"].append(lat)
        self.history["heading"].append(self.heading_c)
                
        # #Replanning every 4 steps
        # a = np.where((self.path_ordered == [self.wp_c_x, self.wp_c_y]).all(1))[0][0] #Check that it is not the last step
        
        # if (np.linalg.norm([self.wp_c_y - self.apex_y, self.wp_c_x - self.apex_x]) > 2 or\
        #     self.utils.is_inside_obs(Node((self.apex_x, self.apex_y)), delta=0.005) or\
        #         self.utils.is_inside_obs(Node((self.foot_x, self.foot_y)), delta=0.005) or\
        #             (step % 4 == 0)) and a < step_max: 
            
        #     print("Replanning...")
            
        #     #Check deviation between current apex state and the optimal path
        #     self.wp_n_x    =   self.path_ordered[a + 1][0]
        #     self.wp_n_y    =   self.path_ordered[a + 1][1]
        #     _ , self.dheading, _ = self.compute_action((self.wp_n_x, self.wp_n_y), (self.apex_x, self.apex_y))
                
        #     n_turns = float(self.dheading / self.max_turn_ang) 
        #     self.step_len = max(float(self.step_len_max / n_turns), self.step_len_min)

        #     self.replan((self.apex_x, self.apex_y), heading_c=self.heading_c, obstacle_margin=0.01)

        #     #new path
        #     self.path_ordered = np.array(self.path)
        #     step_max = len(self.path_ordered) - 1
        #     step = 0
            
        #     #Update the current waypoint
        #     self.wp_c_x = self.path_ordered[0][0] 
        #     self.wp_c_y = self.path_ordered[0][1] 
            
                
        #Get next waypoint and next action
        a = np.where((self.path == [self.wp_c_x, self.wp_c_y]).all(1))[0][0]      
        if a < len(self.path) - 1:
            self.wp_n_x    =   self.path[a + 1][0]
            self.wp_n_y    =   self.path[a + 1][1]
            self.step_l , self.dheading, self.dz  = self.compute_action(model, (self.wp_n_x, self.wp_n_y), (self.wp_c_x, self.wp_c_y))
        else:
            self.step_l , self.dheading, self.dz = 0,0,0 #goal is reached
        
        #Clear Goal after reaching each waypoints
        self.goals = self.goals[1:]
                
        if verbose and  a < len(self.path) - 1:
            print("---------- Digit's PSP Results ---------")
            print("apex state after stepping: ", (self.apex_x, self.apex_y))
            print("current waypoint [PSP]", (wp_x_n, wp_y_n, wp_z_n))
            # print("current waypoint", (self.wp_c_x, self.wp_c_y))
            print("next waypoint", (self.wp_n_x, self.wp_n_y))
            print("vapex", self.vapex)
            # print("PSP output 'step_length': ", step_length)
            # print("previous dtheta [PSP]", dheading)
            # print("current head", np.rad2deg(self.heading_c))
            print("PSP current head", heading_n)
            print("PSP t1", t1)
            print("PSP t2", t2)
            print("stance", self.stance)
            print("----------")
            print("next A_HL: step_l", self.step_l)
            print("next A_HL: dz", self.dz)
            print("next A_HL: dtheta", np.rad2deg(self.dheading))
            print("-----------------------------------\n")
        
        replanning_local = False
        

        
        
        
        #----------------------
        #GP Prediction
        
        foot_xy = np.array([self.foot_x, self.foot_y])
        foot_z_pred, foot_z_var = model.predict(foot_xy)
        
        wp_xy = np.array([wp_x_n, wp_y_n])
        wp_z_pred, wp_z_var = model.predict(wp_xy)
        
        # print("foot_z_pred",foot_z_pred)
        # print("wp_z_pred",wp_z_pred)
    
        
        
        #---------------------
        #Actual values
        
        foot_z_actual = self.sensor.get(np.array([self.foot_x]), np.array([self.foot_y]))
        wp_z_actual = self.sensor.get(np.array([wp_x_n]), np.array([wp_y_n]))
        
        # print("foot_z_actual",foot_z_actual)
        # print("wp_z_actual",wp_z_actual)
        
        #------------------
        #Model Error
        
        #Label:      apex_x,    apex_y,   apex_z,   foot_x,    foot_y, foot_z_psp, foot_z_pred, foot_z_actual, wp_x_n, wp_y_n, wp_z_psp, wp_z_pred, wp_z_actual, heading_n, step_l,     dheading, dz,  
        
        #Collect new data
        if a < len(self.path) - 4:
            location, observations, replanning_local = self.sensor.sense(model, self.state, self.heading_c, ray_tracing=True, num_targets=num_targets, candidate_point=self.path[(a+1):(a+5)])
        else:
            location, observations, replanning_local = self.sensor.sense(model, self.state, self.heading_c, ray_tracing=True, num_targets=num_targets)

        
        self.sampled_locations.append(location)
        self.sampled_observations.append(observations)
        
        if log_data_flag or replanning_local:
            
            # return replanning_local, np.array([self.apex_x, self.apex_y, self.apex_z, dl_switch, ds_switch, step_length, step_width, step_time, dheading, heading_n, \
            # t1, t2, opt_vn, self.foot_x, self.foot_y, wp_x_n, wp_y_n, wp_z_n, self.foot_z, dz, s_switch_g, l_switch_g])
            
            return replanning_local, \
                np.array([self.apex_x, self.apex_y, self.apex_z, dl_switch, ds_switch, step_length, step_width, step_time, dheading, heading_n, \
            t1, t2, opt_vn, self.foot_x, self.foot_y, wp_x_n, wp_y_n, wp_z_n, self.foot_z, dz, s_switch_g, l_switch_g]), \
                np.array([dl_switch, ds_switch, step_length, step_width, step_time, self.dheading, t1, t2, opt_vn, self.foot_x, self.foot_y, wp_x_n, wp_y_n, s_switch_g, l_switch_g]), \
                    np.array([self.apex_x, self.apex_y, self.apex_z, self.foot_x, self.foot_y, self.foot_z, foot_z_pred[0][0], foot_z_var[0][0], foot_z_actual[0], wp_x_n, wp_y_n, wp_z_n, wp_z_pred[0][0], wp_z_var[0][0], wp_z_actual[0], heading_n, step_length, dheading, dz])
                    
            
            
            
    #------------------ Phase Space Planner ---------------------------

def phase_space_planner (apex_x, apex_y, apex_z, wp_x, wp_y, vapex, vapex_n, foot_x, 
                         foot_y, foot_z, heading_c, step_l, dheading, dz, stance, start_flag):
    #new heading
    COS = np.cos((heading_c + dheading) * np.pi/180) 
    SIN = np.sin((heading_c + dheading)* np.pi/180)
    h=1.01

    
    # sagital and lateral apex based on global apex position and global waypoint
    # global -> local
    s1 = (apex_x - wp_x)*COS + (apex_y - wp_y)*SIN
    l1 = -(apex_x - wp_x)*SIN + (apex_y - wp_y)*COS
    v1 = apex_z
    #current sagittal and lateral apex velocities in new local frame (based on heading change)
    s1_dot = vapex*np.cos(dheading* np.pi/180) 
    l1_dot = -vapex*np.sin(dheading* np.pi/180)
    # print(start_flag)
    #current foot
    if start_flag:
        # print('here')
        if stance == 1:
            s1_foot = (foot_x - wp_x)*COS + (foot_y - wp_y)*SIN
            l1_foot = (foot_x - wp_x)*SIN + (foot_y - wp_y)*COS
            v1_foot = foot_z 
        else:
            s1_foot = (foot_x - wp_x)*COS + (foot_y - wp_y)*SIN
            l1_foot = -(foot_x - wp_x)*SIN + (foot_y - wp_y)*COS
            v1_foot = foot_z
    else:
        # print('here2')
        s1_foot = (foot_x - wp_x)*COS + (foot_y - wp_y)*SIN
        l1_foot = -(foot_x - wp_x)*SIN + (foot_y - wp_y)*COS
        v1_foot = foot_z


    
    #next sagital pos
    s2 = s1 + step_l - ((apex_x - wp_x)*COS + (apex_y - wp_y)*SIN) ## might not be nacessary it seems we are adding s1 and subtracting it again
    s2_foot = s2 

    v2 = v1_foot + h + dz 
    
    #next apex vel
    s2_dot = vapex_n #put it in a loop to look for best one later
    l2_dot = 0
    
    if stance == 1: #stance == 1 right foot is in stance  the next foot placment on the negative lateral side of the waypoint
        l2_foot = -0.10012
    else:
        l2_foot = 0.10012

        
    
    eps = 0.001
    forward_num = 0
    backward_num = 0
    aq = (v2 - v1) / (s2 - s1)
    bq = 0
    w1_sq = 9.81 / (h)
    w2_sq = 9.81 / h
    # Collect data points for plotting
    sag_f=[]
    sag_b=[]
    lat_f=[]

    #forward and backward prop
    while (np.abs(s2) > np.abs(s1)):
        #forward
        if(np.abs(s1_dot) < np.abs(s2_dot)):

            forward_num = forward_num + 1
            s1_ddot = w1_sq * (s1 - s1_foot)
            inc_s1 = eps * s1_dot + 0.5 * eps * eps * s1_ddot
            s1 = s1 + inc_s1
            sag_f.append(s1)
            s1_dot = s1_dot + eps * s1_ddot

            l1_ddot = w2_sq * (l1 - l1_foot)
            inc_l1 = eps * l1_dot + 0.5 * eps * eps * l1_ddot
            l1 = l1 + inc_l1
            lat_f.append(l1)
            l1_dot = l1_dot + eps * l1_ddot
            
            

            v1_ddot = aq * s1_ddot + bq * l1_ddot
            v1 = v1 + aq * inc_s1 + bq * inc_l1
            v1_dot = aq * s1_dot + bq * l1_dot
        else:
            #backward
            backward_num = backward_num + 1
            s2_ddot = w1_sq * (s2 - s2_foot)
            inc_s2 = - eps * s2_dot - 0.5 * eps * eps * s2_ddot
            s2 = s2 + inc_s2
            sag_b.insert(0,s2)
            s2_dot = s2_dot - eps * s2_ddot
            

            v2_ddot = aq * s2_ddot
            v2 = v2 + aq * inc_s2
            v2_dot = aq * s2_dot
    
    ds_switch = (s1_dot + s2_dot) /  2
    dl_switch = l1_dot
    
    s_switch = s1
    l_switch = l1
    #newton-raphson search for lateral foot placement
    n=1
    n_max = 150
    max_tol = 0.001
    #print(l1)
    #print(backward_num)
    res1, res2, lat_b = ForwardProp(l2_foot, l1, l1_dot, h, w2_sq, eps, backward_num)
    l2_dot = res2
    l2_ddot = 0.002

    while ((n < n_max) and (np.abs(l2_dot) > max_tol)):

        pre_foot = l2_foot
        l2_foot = l2_foot - l2_dot/l2_ddot
        pre_dot = l2_dot

        res1, res2, lat_b = ForwardProp(l2_foot, l1, l1_dot, h, w2_sq, eps, backward_num)
        l2_dot =res2
        l2_ddot = (l2_dot - pre_dot)/(l2_foot - pre_foot)
        n = n +1
    
    
    #re setting the veriables
    s2 = s2_foot
    s2_dot = vapex_n #change to a loop for search for best apex vel

    l2 = res1
    l2_dot = res2
    
    v2 = foot_z + h + dz
    v2_foot = foot_z + dz
    v2_dot = aq* s2_dot

    #this will be needed for full dynamics simulation later
    #'''
    #delta_y1_c = np.abs(l1_foot -(-(apex_x - wp_x)*SIN + (apex_y - wp_y)*COS))
    #v_apex_c = vapex*np.cos(dheading* np.pi/180) 
    #v_apex_n = s2_dot #will change based on optimal value next
    #step_length = s2_foot - s1_foot
    #step_width = l2_foot - l1_foot
    #step_time = eps*(forward_num+backward_num)
    #dtheta = dheading
    #'''

    
    
    
    #local -> global
    foot_x_g = s2_foot*COS - l2_foot*SIN + wp_x
    foot_y_g = s2_foot*SIN + l2_foot*COS + wp_y
    foot_z_g = foot_z + dz

    apex_x_g = s2*COS - l2*SIN + wp_x
    apex_y_g = s2*SIN + l2*COS + wp_y
    apex_z_g = foot_z_g + h
    sag = sag_f + sag_b
    lat = lat_f + lat_b

    wp_x_n = wp_x + step_l*COS
    wp_y_n = wp_y + step_l*SIN
    wp_z_n = v2

    ### psp_log
    # dl_switch, ds_switch, s2_foot-s1_foot, l2_foot-l1_foot, eps*forward_num+eps*backward_num, prim(1,0),eps*forward_num,
    # eps*backward_num, opt_vn, p_foot(0, 0), p_foot(1, 0), X_d(0, 0), X_d(1, 0), X_switch(0, 0),  X_switch(1, 0);
    step_length = s2_foot - s1_foot
    step_width = l2_foot - l1_foot
    t1 = eps*forward_num
    t2 = eps*backward_num
    step_time = t1 + t2



    return s2, s2_foot, l2, l2_foot, step_time, wp_x_n, wp_y_n, wp_z_n, step_length, step_width, t1, t2, ds_switch, dl_switch, s_switch, l_switch, sag, lat #apex_x_g, apex_y_g, apex_z_g, foot_x_g, foot_y_g, foot_z_g, eps*(forward_num+backward_num)


def ForwardProp(p_f, p, p_dot, h, aq, eps, backward_num):
    lat_b=[]
    w_sq = 9.81/h
    i_inc = np.arange(0,backward_num,1)
    for i in i_inc:
        p_ddot = w_sq * (p - p_f)
        inc_p = eps * p_dot + 0.5 * eps * eps * p_ddot
        p = p + inc_p
        lat_b.append(p)
        p_dot = p_dot + eps * p_ddot

    return p, p_dot, lat_b


def apex_Vel_search(apex_x, apex_y, apex_z, wp_x, wp_y, vapex, foot_x, foot_y, 
                    foot_z, heading_c, step_l, dheading, dz, stance, stop, start_flag):
    v_inc = np.arange(0.1,0.8,0.01) #np.arange(0.1,0.3,0.01)
    h = 1.01
    cost = 10000
    #cost weight
    cy1 = 4
    cy2 =4
    ct = 6
    c_sw = 2

    #desired values
    y1_d = 0
    y2_d = 0.135
    t_d = 0.45
    Sw_d = 0.45

    # opt_vn = 0.1

    for vapex_n in v_inc:
        if(step_l == 0.37*0.3839):
            vapex_n = 0.15
        elif stop:
            vapex_n = 0.1
            # print('here')
        #print(vapex_n)
        s2, s2_foot, l2, l2_foot, step_time, wp_x_n, wp_y_n, wp_z_n, step_length, step_width, t1, t2, ds_switch, dl_switch, s_switch, l_switch, sag, lat =\
            phase_space_planner (apex_x, apex_y, apex_z, wp_x, wp_y, vapex, vapex_n, foot_x, foot_y, foot_z, heading_c, step_l, dheading, dz, stance, start_flag)
        # if (t1 != 0.0 and t2 !=0.0):
        
        # if(stance == 1):
        #     if(dheading < 5) and (dheading > -5):
        #         new_cost = cy1*np.abs(-y1_d - l2) + cy2*(np.abs(-y2_d - (l2_foot-l2))) + ct*(np.abs(t_d - step_time)) + c_sw*(np.abs(Sw_d - np.abs(step_width)))
        #         if(new_cost < cost):
        #             cost = new_cost
        #             opt_vn = vapex_n
        #     # elif(dheading > -0.1):
        #     #     new_cost = cy1*np.abs(y1_d - l2) + cy2*(np.abs(y2_d - (l2_foot-l2))) + ct*(np.abs(t_d - step_time)) + c_sw*(np.abs(Sw_d - np.abs(step_width)))
        #     #     if(new_cost < cost):
        #     #         cost = new_cost
        #     #         opt_vn = vapex_n
        #     else:
        #         new_cost = 0*cy1*np.abs(-y1_d - l2) + cy2*(np.abs(-y2_d - (l2_foot-l2))) + ct*(np.abs(0.35 - step_time)) + 2*c_sw*(np.abs((Sw_d-0.05) - np.abs(step_width))) #+ ct*(np.abs(t_d - t))
        #         if(new_cost < cost):
        #             cost = new_cost
        #             if (vapex_n > 0.25):
        #                 opt_vn = 0.25
        #             else:
        #                 opt_vn = vapex_n

        # else:
        #     if(dheading < 5) and (dheading > -5):
        #         new_cost = cy1*np.abs(y1_d - l2) + cy2*(np.abs(y2_d - (l2_foot-l2))) + ct*(np.abs(t_d - step_time)) + c_sw*(np.abs(Sw_d - np.abs(step_width)))
        #         if(new_cost < cost):
        #             cost = new_cost
        #             opt_vn = vapex_n
        #     # elif(dheading > -0.1):
        #     #     new_cost = cy1*np.abs(y1_d - l2) + cy2*(np.abs(y2_d - (l2_foot-l2))) + ct*(np.abs(t_d - step_time)) + c_sw*(np.abs(Sw_d - np.abs(step_width)))
        #     #     if(new_cost < cost):
        #     #         cost = new_cost
        #     #         opt_vn = vapex_n
        #     else:
        #         new_cost = 0*cy1*np.abs(y1_d - l2) + cy2*(np.abs(y2_d - (l2_foot-l2))) + ct*(np.abs(0.35 - step_time)) + 2*c_sw*(np.abs((Sw_d-0.05) - np.abs(step_width))) #+ ct*(np.abs(t_d - t))
        #         if(new_cost < cost):
        #             cost = new_cost
        #             if (vapex_n > 0.25):
        #                 opt_vn = 0.25
        #             else:
        #                 opt_vn = vapex_n
                        
        
        # OLDEN DAYS CODE
        
        if(stance == 1):
            if(dheading == 0):
                new_cost = cy1*np.abs(-y1_d - l2) + cy2*(np.abs(-y2_d - (l2_foot-l2))) + ct*(np.abs(t_d - step_time)) + c_sw*(np.abs(Sw_d - np.abs(step_width)))
                if(new_cost < cost):
                    cost = new_cost
                    opt_vn = vapex_n
            else:
                new_cost = 0*cy1*np.abs(-y1_d - l2) + cy2*(np.abs(-y2_d - (l2_foot-l2))) + ct*(np.abs(0.35 - step_time)) + 2*c_sw*(np.abs((Sw_d-0.05) - np.abs(step_width))) #+ ct*(np.abs(t_d - t))
                if(new_cost < cost):
                    cost = new_cost
                    if (vapex_n > 0.25): #vapex_n > 0.5
                        opt_vn = 0.2
                    else:
                        opt_vn = vapex_n
        else:
            if(dheading == 0):
                new_cost = cy1*np.abs(y1_d - l2) + cy2*(np.abs(y2_d - (l2_foot-l2))) + ct*(np.abs(t_d - step_time)) + c_sw*(np.abs(Sw_d - np.abs(step_width)))
                if(new_cost < cost):
                    cost = new_cost
                    opt_vn = vapex_n
            else:
                new_cost = 0*cy1*np.abs(y1_d - l2) + cy2*(np.abs(y2_d - (l2_foot-l2))) + ct*(np.abs(0.35 - step_time)) + 2*c_sw*(np.abs((Sw_d-0.05) - np.abs(step_width))) #+ ct*(np.abs(t_d - t))
                if(new_cost < cost):
                    cost = new_cost
                    if (vapex_n > 0.25): #vapex_n > 0.5
                        opt_vn = 0.2
                    else:
                        opt_vn = vapex_n

    s2, s2_foot, l2, l2_foot, step_time, wp_x_n, wp_y_n, wp_z_n, step_length, step_width, t1, t2, ds_switch, dl_switch, s_switch, l_switch, sag, lat =\
        phase_space_planner (apex_x, apex_y, apex_z, wp_x, wp_y, vapex, opt_vn, foot_x, foot_y, foot_z, heading_c, step_l, dheading, dz, stance, start_flag)
    # local to global
    COS = np.cos((heading_c + dheading) * np.pi/180) 
    SIN = np.sin((heading_c + dheading)* np.pi/180)
    foot_x_g = s2_foot*COS - l2_foot*SIN + wp_x
    foot_y_g = s2_foot*SIN + l2_foot*COS + wp_y
    foot_z_g = foot_z + dz

    apex_x_g = s2*COS - l2*SIN + wp_x
    apex_y_g = s2*SIN + l2*COS + wp_y
    apex_z_g = foot_z_g + h
    heading_n = heading_c + dheading

    s_switch_g = s_switch*COS - l_switch*SIN + wp_x
    l_switch_g = s_switch*SIN + l_switch*COS + wp_y

     # dl_switch, ds_switch, s2_foot-s1_foot, l2_foot-l1_foot, eps*forward_num+eps*backward_num, prim(1,0),eps*forward_num,
    # eps*backward_num, opt_vn, p_foot(0, 0), p_foot(1, 0), X_d(0, 0), X_d(1, 0), X_switch(0, 0),  X_switch(1, 0);

    return apex_x_g, apex_y_g, apex_z_g, dl_switch, ds_switch, step_length, step_width, step_time, dheading, heading_n, \
        t1, t2, opt_vn, foot_x_g, foot_y_g, wp_x_n, wp_y_n, wp_z_n, foot_z_g, dz, s_switch_g, l_switch_g, sag, lat
    

    #return apex_x_g, apex_y_g, apex_z_g, foot_x_g, foot_y_g, foot_z_g, opt_vn, t, traj_xs, traj_ys, wp_x_n, wp_y_n, wp_z_n, heading_n















#to run the code use function apex_vel_search

#apex_Vel_search(apex_x, apex_y, apex_z, wp_x, wp_y, vapex, foot_x, foot_y, foot_z, heading_c, step_l, dheading, dz, stance):
    #inputs are in global coordinate
    #apex_xyz=  COM in global
    #wp_xy= waypoint in global
    #vapex= current velocity (keep track of the output opt_vn to use in future steps), for first step use 0.1
    #foot_xyz= globale foot position 
    #heading_c= current heading in global coord
    #step_l= step length ('d')
    #dheading= commanded heading change 
    #dz= commanded step height
    #stance= foot stance flag stance = 1: right foot stance (postive y in the local waypoint coord)

    #outputs
    #global apex position and global foot position, optimal next apex vel, and step time 


# apex_x_g, apex_y_g, apex_z_g, foot_x_g, foot_y_g, foot_z_g, opt_vn, t, traj_xs, traj_ys = apex_Vel_search(1.47655761, 0.211341894, 0.983, 1.47655761, 0.211341894, 0.1, 1.47663691, 0.211581, -0.03771821, 341.652253, 0.43262422, 0, 0, 1)
#apex_x_g, apex_y_g, apex_z_g, foot_x_g, foot_y_g, foot_z_g, opt_vn, t = apex_Vel_search(.55, 0.05, 0.983, 0.55, .35, 0.1, 0.65, 0.05, 0, 90, 0.3, 0, 0, 0)
# apex_x_g, apex_y_g, apex_z_g, foot_x_g, foot_y_g, foot_z_g, opt_vn, t, traj_xs, traj_ys = apex_Vel_search(.55, 0.05, 0.983, 0.55, .35, 0.1, 0.65, 0.05, 0, 90, 0.3, 0, 0, 0)

# print apex_x_g