import numpy as np
from numba import njit, cuda
from CONS import *

############################################
########  Part 1. CPU  #####################

@njit
# Detect whether there is a car in front of current car
def check_front_h(road, position, NC, cell_size):
    i = position[0]
    j = position[1]
    
    j_max = j + int(MAX_FORWARD/cell_size) + 1 # maximum distance to detect
    df = MAX_FORWARD
    
    # check for car in each cell, one by one
    for j_f in range(j + 1, j_max):            
        if j_f >= NC:
            j_f_temp = j_f - NC
        else:
            j_f_temp = j_f
            
        # if there is a car    
        if road[i][j_f_temp]:
            df = (j_f - j) * cell_size # distance to the front car
            return df
    
    return df # current distance with the fore car and the position index of the fore car

@njit
def check_necessity_h(road, position, speed, NC, cell_size):
    i = position[0]
    j = position[1]
    
    left_necessary = False
    right_necessary = False
    df = check_front_h(road, (i, j), NC, cell_size)
    
    left_eligible = i > 0
    if left_eligible:
        df_l = check_front_h(road, (i - 1, j), NC, cell_size)
        left_necessary = (df_l - df) > TC * speed

    right_eligible = i < NL - 1
    if right_eligible:
        df_r = check_front_h(road, (i + 1, j), NC, cell_size)
        right_necessary = (df_r - df) > TC * speed


    return left_necessary, right_necessary

@njit
def check_safe_h(direct, road, position, speed, NC, cell_size):
    assert direct == 1 or direct == -1
    i = position[0]
    j = position[1]
    
    # check if safe for immediate adjacent lane 
    # (only need to look back, because check_necessity did forward check)
    d_safe_0 = max(speed * TR, CAR_SIZE + SAFE_MARGIN)
    j_min = j - int(d_safe_0/cell_size)
    safe_0 = True
    
    j_b = j
    i_b = i + direct
    while safe_0 and j_b > j_min:
        if road[i_b][j_b]:
            safe_0 = False
        j_b -= 1
    
    # check if safe for the lane right next to the immediate adjacent lane, if applicable
    # (need to look both rear and front to avoid two cars change to the same lane and get to close to each other)
    i_s = i + direct + direct
    if i_s >= 0 and i_s < NL:
        d_safe_1 = CAR_SIZE + 2 * SAFE_MARGIN
        j_max = j + int(d_safe_1/cell_size)
        j_min = j - int(d_safe_1/cell_size)
        safe_1 = True
        
        j_s = j_min
        while safe_1 and j_s <= j_max:
            if j_s >= NC:
                j_s_temp = NC - j_s
            else:
                j_s_temp = j_s
            if road[i_s][j_s_temp]:
                safe_1 = False
            j_s += 1
            
    return (safe_0 and safe_1)


@njit
# update Cell, used for forwarding
def update_cell_h(speed, accel, df):

    # update speed based on acceleration
    speed += accel * DT
    speed = min(VMAX, max(VMIN, speed))
    
    # update acceleration
    d_safe = max(speed * TR, CAR_SIZE + SAFE_MARGIN)
    accel = ALPHA * (df - d_safe)
    
    return speed, accel

@njit
# Calculate the displacement current car can move forward
def cal_displacement_h(j, speed, distance2front, NC, cell_size):
    
    displacement = speed * DT
    displacement = max(min(displacement, distance2front - CAR_SIZE - SAFE_MARGIN), 0)
    dj = int(displacement/cell_size)

    # determine new cell index
    if (j + dj) >= NC:
        j_new = j + dj - NC
    else:
        j_new = j + dj

    return j_new

@njit
# Rule for lane changing
def changeLane_h(road, speed, accel, pos_x, pos_y, rands, cell_size):
    
    NC = road.shape[1]
    n_car = pos_y.shape[0]
    road_copy = road.copy()
    counter_lc = 0
    
    for k in range(n_car):
        i = pos_y[k]
        j = pos_x[k]
        v = speed[i][j]
        a = accel[i][j]
        
        left_necessary, right_necessary = check_necessity_h(road_copy, (i, j), v, NC, cell_size)
        
        if left_necessary:
            left_safe = check_safe_h(-1, road_copy, (i, j), v, NC, cell_size)
            left = left_necessary and left_safe
        else:
            left = False
        
        if right_necessary:
            right_safe = check_safe_h(1, road_copy, (i, j), v, NC, cell_size)
            right = right_necessary and right_safe
        else:
            right = False
                      
        if left and right: # Both left and right are good for lane changing
            bl, br = PC/2, 1 - PC/2
        elif left and (not right):
            bl, br = PC, 1
        elif (not left) and right:
            bl, br = 0, 1 - PC
        else:
            bl, br = 0, 1
            
        x = rands[k]
        if x < bl:
            di = -1
        elif x > br:
            di = 1
        else:
            di = 0
        i_new = i + di
        
        road[i][j] = 0
        speed[i][j] = 0
        accel[i][j] = 0
        
        road[i_new][j] = k + 1
        speed[i_new][j] = v
        accel[i_new][j] = a
        pos_y[k] = i_new
        
    return road, speed, accel, pos_y

@njit
def forward_h(road, speed, accel, pos_x, pos_y, cell_size):
    
    NC = road.shape[1]
    n_car = pos_y.shape[0]
    road_copy = road.copy()
    
    for k in range(n_car):
        i = pos_y[k]
        j = pos_x[k]
        v = speed[i][j]
        a = accel[i][j]

        df = check_front_h(road_copy, (i,j), NC, cell_size)        
        j_new = cal_displacement_h(j, v, df, NC, cell_size)
        
        road[i][j] = 0
        speed[i][j] = 0
        accel[i][j] = 0

        pos_x[k] = j_new
        road[i][j_new] = k + 1
        speed[i][j_new], accel[i][j_new] = update_cell_h(v, a, df)
                
    return road, speed, accel, pos_x


#############################################
########  Part 2. CUDA  #####################
@cuda.jit(device=True)
# Detect whether there is a car in front of current car
def check_front_d(road, position, NC, cell_size):
    i = position[0]
    j = position[1]
    
    j_max = j + np.int(MAX_FORWARD/cell_size) + 1 # maximum distance to detect
    df = MAX_FORWARD
    
    # check for car in each cell, one by one
    for j_f in range(j + 1, j_max):            
        if j_f >= NC:
            j_f_temp = j_f - NC
        else:
            j_f_temp = j_f
            
        # if there is a car    
        if road[i][j_f_temp]:
            df = (j_f - j) * cell_size # distance to the front car
            return df
    
    return df # current distance with the fore car and the position index of the fore car

@cuda.jit(device=True)
def check_necessity_d(road, position, speed, NC, cell_size):
    i = position[0]
    j = position[1]
    
    left_necessary = False
    right_necessary = False
    df = check_front_d(road, (i, j), NC, cell_size)
    
    left_eligible = i > 0
    if left_eligible:
        df_l = check_front_d(road, (i - 1, j), NC, cell_size)
        left_necessary = (df_l - df) > TC * speed

    right_eligible = i < NL - 1
    if right_eligible:
        df_r = check_front_d(road, (i + 1, j), NC, cell_size)
        right_necessary = (df_r - df) > TC * speed


    return left_necessary, right_necessary

@cuda.jit(device=True)
def check_safe_d(direct, road, position, speed, NC, cell_size):
    assert direct == 1 or direct == -1
    i = position[0]
    j = position[1]
    
    # check if safe for immediate adjacent lane 
    # (only need to look back, because check_necessity did forward check)
    d_safe_0 = max(speed * TR, CAR_SIZE + SAFE_MARGIN)
    j_min = j - int(d_safe_0/cell_size)
    safe_0 = True
    
    j_b = j
    i_b = i + direct
    while safe_0 and j_b > j_min:
        if road[i_b][j_b]:
            safe_0 = False
        j_b -= 1
    
    # check if safe for the lane right next to the immediate adjacent lane, if applicable
    # (need to look both rear and front to avoid two cars change to the same lane and get to close to each other)
    i_s = i + direct + direct
    if i_s >= 0 and i_s < NL:
        d_safe_1 = CAR_SIZE + 2 * SAFE_MARGIN
        j_max = j + int(d_safe_1/cell_size)
        j_min = j - int(d_safe_1/cell_size)
        safe_1 = True
        
        j_s = j_min
        while safe_1 and j_s <= j_max:
            if j_s >= NC:
                j_s_temp = NC - j_s
            else:
                j_s_temp = j_s
            if road[i_s][j_s_temp]:
                safe_1 = False
            j_s += 1
            
    return (safe_0 and safe_1)

@cuda.jit(device=True)
# update speed and acceleration
def update_cell_d(speed, accel, df):

    # update speed based on acceleration
    speed += accel * DT
    speed = min(VMAX, max(VMIN, speed))
    
    # update acceleration
    d_safe = max(speed * TR, CAR_SIZE + SAFE_MARGIN)
    accel = ALPHA * (df - d_safe)
    
    return speed, accel

@cuda.jit(device=True)
# Calculate the displacement current car can move forward
def cal_displacement_d(j, speed, distance2front, NC, cell_size):
    
    displacement = speed * DT
    displacement = max(min(displacement, distance2front - CAR_SIZE - SAFE_MARGIN), 0)
    dj = int(displacement/cell_size)

    # determine new cell index
    if (j + dj) >= NC:
        j_new = j + dj - NC
    else:
        j_new = j + dj

    return j_new
        
