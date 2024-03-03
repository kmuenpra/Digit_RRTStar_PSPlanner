import numpy as np
import psp  as psp
import json

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def HL_keyframe(stepL, stepH, turn, stanceFoot):
    if(stanceFoot == 0):
        stanceFoot = 1
    else:
        stanceFoot = 0

    # obs1x = obs1/8
    # obs1y = obs1 - obs1/8

    # step height
    if stepH == 3:
        dheight = 0
    elif stepH == 2:
        dheight = -0.1
    elif stepH == 4:
        dheight = 0.1
    elif stepH == 0:
        dheight = 0.0
    
    # heading change
    if turn == 0:
        dheading = 0.314159
    elif turn == 1:
        dheading = 0.314159
    elif turn == 2:
        dheading = 0.0
    elif turn == 3:
        dheading = -0.314159
    elif turn == 4:
        dheading = -0.314159
    
    #step lenght
    if stepL == 0:
        step_length = 0.41538461538
    elif stepL == 1:
        step_length = 0.28
    elif stepL == 2:
        step_length = 0.43
    elif stepL == 3:
        step_length = 0.3839
    elif stepL == 4:
        step_length = 0.47389
    elif stepL == 5:
        step_length = 0.20769230769
    elif stepL == 6:
        step_length = 0.31153846153
    elif stepL == 7:
        step_length = 0.51923076923
    elif stepL == 8:
        step_length = 0.83076923076
    
    step_factor = 0.37
    step_length = step_length*step_factor

    return step_length, dheight, dheading*180/(np.pi), stanceFoot
    
def write2txt(dl_switch_list, ds_switch_list, step_length_list, step_width_list, step_time_list,
               dheading_list, t1_list, t2_list, opt_vn_list, foot_list_x, foot_list_y,
                 waypoint_list_x, waypoint_list_y, s_switch_list, l_switch_list, foot_list_z, dh_list):

    # psp_log << dl_switch, ds_switch, s2_foot-s1_foot, l2_foot-l1_foot, eps*forward_num+eps*backward_num,
    # prim(1,0),eps*forward_num, eps*backward_num, opt_vn, p_foot(0, 0), p_foot(1, 0), 
    # X_d(0, 0), X_d(1, 0), X_switch(0, 0),  X_switch(1, 0);

    

    # stack the arrays horizontally
    data = np.column_stack((dl_switch_list, ds_switch_list, step_length_list, step_width_list, step_time_list,
    dheading_list , t1_list , t2_list, opt_vn_list, foot_list_x , foot_list_y ,
    waypoint_list_x , waypoint_list_y , s_switch_list, l_switch_list, foot_list_z, dh_list))

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data[i][j] = round(data[i][j], 8)

    # specify the file name and open it for writing
    filename = "output_multi_Stair.txt"
    with open(filename, "w") as f:
        # loop through each row of the data array
        for row in data:
            # convert each element to a string and join them with spaces
            line = " ".join(str(x) for x in row)
            # write the line to the file
            f.write(line + "\n")




# this to read highlevel ltl commands 
with open('coarse_grid_multi_obs_belief_collision_TP_stair_larg_step.json') as user_file:
  file_contents = user_file.read()
parsed_json = json.loads(file_contents)

cellsize = 2.7035
#initial keyframe state
wp_x = 0*5.5*0.37*cellsize
wp_y = 0*3.5*0.37*cellsize 
apex_x = wp_x #+ 0.03012
apex_y = wp_y + 0.03 + 0.1
apex_z = 1.01
wp_z = apex_z
vapex = 0.25
foot_x = wp_x #+ 0.13012
foot_y = wp_y + 0.13012 + 0.1
foot_z = 0.0
heading_c = 0


apex_list_x = []
apex_list_y = []
apex_list_z = []
apex_list_x.append(apex_x)
apex_list_y.append(apex_y)
apex_list_z.append(apex_z)


dl_switch_list = []; ds_switch_list = []; step_length_list = []; step_width_list = []; step_time_list = []
dheading_list = []; t1_list = []; t2_list = []; opt_vn_list = []; foot_list_x = []; foot_list_y = []
waypoint_list_x = []; waypoint_list_y = []; foot_list_z = []; dh_list = []; s_switch_list = []; l_switch_list = []

waypoint_list_z = []

foot_list_z = []

stance = 2

start_flag = 1

#### initial turn 
for i in np.arange(0,19):
            # need to keep velocity zero and just command dheading 
            dl_switch_list.append(0); ds_switch_list.append(0); step_length_list.append(0)
            step_width_list.append(0.4); step_time_list.append(0.4); dheading_list.append(0.0872665)
            t1_list.append(0.2); t2_list.append(0.2); opt_vn_list.append(0.1); dh_list.append(0)

            s_switch_list.append(wp_x)
            l_switch_list.append(wp_y)

            apex_list_x.append(apex_x)
            apex_list_y.append(apex_y)
            apex_list_z.append(apex_z)
            waypoint_list_x.append(wp_x)
            waypoint_list_y.append(wp_y)
            waypoint_list_z.append(wp_z)
            foot_list_x.append(foot_x)
            foot_list_y.append(foot_y)
            foot_list_z.append(foot_z)

dheight_prev = 0
# traj_x_list = []
# traj_y_list = []
# traj_x_list.append(apex_x)
# traj_y_list.append(apex_y)
border = Rectangle((0, 0), 8, -5, fc = "None", ec="black" )
for k in np.arange(1, 34):

    
    stepL = parsed_json['saved_states'][str(k)]['action_info']['State']['stepL'] #meters
    stepH = parsed_json['saved_states'][str(k)]['action_info']['State']['stepH'] #step H meters
    turn = parsed_json['saved_states'][str(k)]['action_info']['State']['turn'] #radians
    Stop = parsed_json['saved_states'][str(k)]['action_info']['State']['stop'] #bolean for stop
    forward = parsed_json['saved_states'][str(k)]['action_info']['State']['forward'] #bolean for moving forward
    stanceFoot = parsed_json['saved_states'][str(k)]['action_info']['State']['stanceFoot'] #bolean for which stance leg


    step_length, dheight, dheading, stanceFoot = HL_keyframe(stepL, stepH, turn, stanceFoot)
    # print(stanceFoot)
    # print('stop', Stop)
    # dheight = 0.0
    # if(k == 21):
    #     dheight = 0.1


    
    #hard code stopping for stair step
    if dheight_prev > 0.05:
        in_place_steps = 8
    else:
        in_place_steps = 24

    if dheight > 0.0:
        for i in np.arange(0,in_place_steps):
            hyper_param = psp.apex_Vel_search(apex_x, apex_y, apex_z, wp_x, wp_y, vapex, foot_x, foot_y,
                                            foot_z, heading_c, step_length, dheading, dheight,
                                              stanceFoot, Stop, start_flag)
            
            dl_switch_list.append(0); ds_switch_list.append(0); step_length_list.append(0)
            step_width_list.append(0.4); step_time_list.append(0.4); dheading_list.append(0)
            t1_list.append(0.2); t2_list.append(0.2); opt_vn_list.append(0.1); dh_list.append(0)

            s_switch_list.append(hyper_param[20])
            l_switch_list.append(hyper_param[21])
            apex_list_x.append(apex_x)
            apex_list_y.append(apex_y)
            apex_list_z.append(apex_z)
            waypoint_list_x.append(wp_x)
            waypoint_list_y.append(wp_y)
            waypoint_list_z.append(wp_z)
            foot_list_x.append(foot_x)
            foot_list_y.append(foot_y)
            foot_list_z.append(foot_z)

        # intial walking step after a stop
        dl_switch_list.append(0); ds_switch_list.append(0.2); step_length_list.append(0)
        step_width_list.append(0.4); step_time_list.append(0.4); dheading_list.append(0)
        t1_list.append(0.2); t2_list.append(0.2); opt_vn_list.append(0.1); dh_list.append(0)

        s_switch_list.append(hyper_param[20])
        l_switch_list.append(hyper_param[21])

        apex_list_x.append(apex_x)
        apex_list_y.append(apex_y)
        apex_list_z.append(apex_z)
        waypoint_list_x.append(wp_x)
        waypoint_list_y.append(wp_y)
        waypoint_list_z.append(wp_z)
        foot_list_x.append(foot_x)
        foot_list_y.append(foot_y)
        foot_list_z.append(foot_z)

    # if walking
    if forward == 1:
        # if walkng after a stop
        if stance == 2:
            dl_switch_list.append(0); ds_switch_list.append(0.2); step_length_list.append(0.15)
            step_width_list.append(0.3); step_time_list.append(0.4); dheading_list.append(0)
            t1_list.append(0.3); t2_list.append(0.2); opt_vn_list.append(0.1); dh_list.append(0)

            apex_list_x.append(apex_list_x[-1])
            apex_list_y.append(apex_list_y[-1])
            apex_list_z.append(apex_list_z[-1])
            waypoint_list_x.append(waypoint_list_x[-1])
            waypoint_list_y.append(waypoint_list_y[-1])
            waypoint_list_z.append(waypoint_list_z[-1])
            s_switch_list.append(waypoint_list_x[-1])
            l_switch_list.append(waypoint_list_y[-1])
            foot_list_x.append(foot_list_x[-1])
            foot_list_y.append(foot_list_y[-1])
            foot_list_z.append(foot_list_z[-1])

            dl_switch_list.append(0); ds_switch_list.append(0.2); step_length_list.append(0.15)
            step_width_list.append(0.3); step_time_list.append(0.4); dheading_list.append(0)
            t1_list.append(0.3); t2_list.append(0.2); opt_vn_list.append(0.1); dh_list.append(0)

            apex_list_x.append(apex_list_x[-1])
            apex_list_y.append(apex_list_y[-1])
            apex_list_z.append(apex_list_z[-1])
            waypoint_list_x.append(waypoint_list_x[-1])
            waypoint_list_y.append(waypoint_list_y[-1])
            waypoint_list_z.append(waypoint_list_z[-1])
            s_switch_list.append(waypoint_list_x[-1])
            l_switch_list.append(waypoint_list_y[-1])
            foot_list_x.append(foot_list_x[-1])
            foot_list_y.append(foot_list_y[-1])
            foot_list_z.append(foot_list_z[-1])

            

            hyper_param = psp.apex_Vel_search(apex_x, apex_y, apex_z, wp_x, wp_y, vapex, foot_x, foot_y,
                                            foot_z, heading_c, step_length, dheading, dheight,
                                              stanceFoot, Stop, start_flag)
            
            stance = 0

        # if next step is a STOP
        elif Stop:
            stance = 2
            start_flag = 0
            hyper_param = psp.apex_Vel_search(apex_x, apex_y, apex_z, wp_x, wp_y, vapex, foot_x, foot_y,
                                            foot_z, heading_c, step_length, dheading, dheight,
                                              stanceFoot, Stop, start_flag)
            

            # wp_x = hyper_param[15] #wp_x_n
            # wp_y = hyper_param[16] #wp_y_n
            # wp_z = hyper_param[17] #wp_z_n
            # heading_c = hyper_param[9] #heading_n
            apex_list_x.append(hyper_param[0])
            apex_list_y.append(hyper_param[1])
            apex_list_z.append(hyper_param[2])
            waypoint_list_x.append(wp_x)
            waypoint_list_y.append(wp_y)
            waypoint_list_z.append(wp_z)
            foot_list_x.append(hyper_param[13])
            foot_list_y.append(hyper_param[14])
            foot_list_z.append(hyper_param[18])

            # apex_x = hyper_param[0]
            # apex_y = hyper_param[1]
            # apex_z = hyper_param[2]

            # foot_x = hyper_param[13]
            # foot_y = hyper_param[14]
            # foot_z = hyper_param[18]
            # vapex = hyper_param[12]

            s_switch_list.append(hyper_param[20])
            l_switch_list.append(hyper_param[21])

            # logging for txt file
            dl_switch_list.append(hyper_param[3]); ds_switch_list.append(hyper_param[4]); step_length_list.append(hyper_param[5])
            step_width_list.append(hyper_param[6]); step_time_list.append(hyper_param[7]); dheading_list.append(hyper_param[8])
            t1_list.append(0.2); t2_list.append(0.2); opt_vn_list.append(hyper_param[12]); dh_list.append(hyper_param[19])

            # print('here')

        # walking and next step is walking
        else:
            # 0 apex_x_g, 1 apex_y_g, 2 apex_z_g, 3 dl_switch, 4 ds_switch, 5 step_length,
            # 6 step_width, 7 step_time, 8 dheading, 9 heading_n, 10 t1, 11 t2, 12 opt_vn,
            # 13 foot_x_g, 14 foot_y_g, 15 wp_x_n, 16 wp_y_n, 17 wp_z_n, 18 foot_z_g, 19 h, 20 s_switch, 21 l_switch
            hyper_param = psp.apex_Vel_search(apex_x, apex_y, apex_z, wp_x, wp_y, vapex, foot_x, foot_y,
                                            foot_z, heading_c, step_length, dheading, dheight,
                                              stanceFoot, Stop, start_flag)
            wp_x = hyper_param[15] #wp_x_n
            wp_y = hyper_param[16] #wp_y_n
            wp_z = hyper_param[17] #wp_z_n
            heading_c = hyper_param[9] #heading_n
            # hyper_param[9] = hyper_param[9]
            # print('t1=', hyper_param[10]) 
            # print('t2=', hyper_param[11])
            # print('step time =', hyper_param[7])
            # print('step width:', hyper_param[6])
            # print('step length:', hyper_param[5])
            # print('ds_switch:', hyper_param[4])
            # print('opt_vn:', hyper_param[12]) #opt_vn

            # logging for txt file
            apex_list_x.append(hyper_param[0])
            apex_list_y.append(hyper_param[1])
            apex_list_z.append(hyper_param[2])
            waypoint_list_x.append(wp_x)
            waypoint_list_y.append(wp_y)
            waypoint_list_z.append(wp_z)
            foot_list_x.append(hyper_param[13])
            foot_list_y.append(hyper_param[14])
            foot_list_z.append(hyper_param[18])

            apex_x = hyper_param[0]
            apex_y = hyper_param[1]
            apex_z = hyper_param[2]

            foot_x = hyper_param[13]
            foot_y = hyper_param[14]
            foot_z = hyper_param[18]
            vapex = hyper_param[12]

            # logging for txt file
            s_switch_list.append(hyper_param[20])
            l_switch_list.append(hyper_param[21])
            dl_switch_list.append(hyper_param[3]); ds_switch_list.append(hyper_param[4]); step_length_list.append(hyper_param[5])
            step_width_list.append(hyper_param[6]); step_time_list.append(hyper_param[7]); dheading_list.append(hyper_param[8]*np.pi/180)
            t1_list.append(0.2); t2_list.append(0.2); opt_vn_list.append(hyper_param[12]); dh_list.append(hyper_param[19])

            if dheight > 0.0:

                hyper_param = psp.apex_Vel_search(apex_x, apex_y, apex_z, wp_x, wp_y, vapex, foot_x, foot_y,
                                            foot_z, heading_c, step_length, dheading, dheight,
                                              stanceFoot, Stop, start_flag)

                dl_switch_list.append(0); ds_switch_list.append(0.2); step_length_list.append(0)
                step_width_list.append(0.4); step_time_list.append(0.4); dheading_list.append(0)
                t1_list.append(0.2); t2_list.append(0.2); opt_vn_list.append(0.1); dh_list.append(0)

                s_switch_list.append(hyper_param[20])
                l_switch_list.append(hyper_param[21])

                apex_list_x.append(apex_x)
                apex_list_y.append(apex_y)
                apex_list_z.append(apex_z)
                waypoint_list_x.append(wp_x)
                waypoint_list_y.append(wp_y)
                waypoint_list_z.append(wp_z)
                foot_list_x.append(foot_x)
                foot_list_y.append(foot_y)
                foot_list_z.append(foot_z)

            

        # traj_x_list.append(traj_xs)
        # traj_y_list.append(traj_ys)

    #forward is zero
    else:
        dl_switch_list.append(0); ds_switch_list.append(0); step_length_list.append(0)
        step_width_list.append(0.4); step_time_list.append(0.4); dheading_list.append(0)
        t1_list.append(0.2); t2_list.append(0.2); opt_vn_list.append(0.1); dh_list.append(0)

        s_switch_list.append(wp_x)
        l_switch_list.append(wp_y)

        apex_list_x.append(apex_x)
        apex_list_y.append(apex_y)
        apex_list_z.append(apex_z)
        waypoint_list_x.append(wp_x)
        waypoint_list_y.append(wp_y)
        waypoint_list_z.append(wp_z)
        foot_list_x.append(foot_x)
        foot_list_y.append(foot_y)
        foot_list_z.append(foot_z)
        stance = 2

        # print('here')
    dheight_prev = dheight

    # print('opt_vn:', opt_vn_list[-1])
    # print('foot_y:', step_width_list[-1])
    # print('t1:', t1_list[-1])
    # print('t2:', t2_list[-1])
    
    
    



write2txt(dl_switch_list, ds_switch_list, step_length_list, step_width_list, step_time_list, 
          dheading_list, t1_list, t2_list, opt_vn_list, foot_list_x, foot_list_y, waypoint_list_x,
            waypoint_list_y, s_switch_list, l_switch_list, foot_list_z, dh_list)

fig = plt.figure()

# ax1 =  fig.add_subplot(1,1,1)
# ax1.add_patch(border)
# ax1.scatter(apex_list_x,apex_list_y)
# ax1.scatter(waypoint_list_x,waypoint_list_y, color = 'black')
# ax1.scatter(foot_list_x,foot_list_y, color = 'red')

# 3d requires python3
ax1 =  fig.add_subplot(projection='3d') # fig.add_subplot(1,1,1)
ax1.scatter(apex_list_x,apex_list_y, apex_list_z)
ax1.scatter(waypoint_list_x,waypoint_list_y, waypoint_list_z, color = 'black')
ax1.scatter(foot_list_x,foot_list_y, foot_list_z, color = 'red')
ax1.set_box_aspect((1, 1, 1))
plt.show()