import numpy as np
from numba import njit, cuda
from CONS import *


############################################
########  Part 1. CPU without jit  #########

def update_cell(speed, accel, df):

    # update speed based on acceleration
    speed += accel * DT
    speed = min(VMAX, max(VMIN, speed))
    
    # update acceleration
    d_safe = max(speed * TR, CAR_SIZE + SAFE_MARGIN)
    accel = ALPHA * (df - d_safe)
    
    return speed, accel

# Detect whether there is a car in front of current car
def check_front(road, position, NC, cell_size):
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

# Calculate the displacement current car can move forward
def cal_displacement(j, speed, distance2front, NC, cell_size):
    
    displacement = speed * DT
    displacement = max(min(displacement, distance2front - CAR_SIZE - SAFE_MARGIN), 0)
    dj = int(displacement/cell_size)

    # determine new cell index
    if (j + dj) >= NC:
        j_new = j + dj - NC
    else:
        j_new = j + dj

    return j_new

def forward(road, speed, accel, cell_size):
    
    NC = road.shape[1]
    road_update = np.zeros((NL, NC), dtype = np.int32)
    road_copy = road.copy()
    
    for i in range(NL):
        for j in range(NC):
            if road[i][j] and road_update[i][j] == 0:
                # Calculate distance to the front car 
                df = check_front(road_copy, (i,j), NC, cell_size)        

                # Calculate the new position index
                j_new = cal_displacement(j, speed[i][j], df, NC, cell_size)
                
                road[i][j_new] = road[i][j]
                road_update[i][j_new] = 1
                speed[i][j_new], accel[i][j_new]\
                    = update_cell(speed[i][j], accel[i][j], df)
                
                if j_new != j:
                    road[i][j] = 0
                    speed[i][j] = 0
                    accel[i][j] = 0
                
    return road, speed, accel

############################################
########  Part 2. CPU with jit  ###########

@njit
def update_cell_h(speed, accel, df):

    # update speed based on acceleration
    speed += accel * DT
    speed = min(VMAX, max(VMIN, speed))
    
    # update acceleration
    d_safe = max(speed * TR, CAR_SIZE + SAFE_MARGIN)
    accel = ALPHA * (df - d_safe)
    
    return speed, accel

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
def forward_h(road, speed, accel, cell_size):
    
    NC = road.shape[1]
    road_update = np.zeros((NL, NC), dtype = np.int32)
    road_copy = road.copy()
    
    for i in range(NL):
        for j in range(NC):
            if road[i][j] and road_update[i][j] == 0:
                # Calculate distance to the front car 
                df = check_front_h(road_copy, (i,j), NC, cell_size)        

                # Calculate the new position index
                j_new = cal_displacement_h(j, speed[i][j], df, NC, cell_size)
                
                road[i][j_new] = road[i][j]
                road_update[i][j_new] = 1
                speed[i][j_new], accel[i][j_new]\
                    = update_cell_h(speed[i][j], accel[i][j], df)
                
                if j_new != j:
                    road[i][j] = 0
                    speed[i][j] = 0
                    accel[i][j] = 0
                
    return road, speed, accel

############################################
########  Part 3. CUDA with jit  ###########

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
                