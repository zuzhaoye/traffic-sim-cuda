import numpy as np
from numba import njit
from CONS import *

# Generation matrices of road, speed, and acceleration at t = 0
@njit
def initialize(road_length, cell_size, traffic_density):
    
    NC = int(road_length * 1000/cell_size) # number of longitudinal cells
    traffic_density = min(traffic_density, 90) # traffic density shall not exceed a reasonable threshold
    n_car_per_lane = road_length * traffic_density # number of cars per lane
    spacing = np.int(NC / n_car_per_lane) # space between each car
    
    road = np.zeros((NL, NC), dtype = np.int32)
    speed = np.zeros((NL, NC), dtype = np.float32)
    accel = np.zeros((NL, NC), dtype = np.float32)
    
    n_car = int(NL * np.arange(0, NC - spacing, spacing).shape[0])
    pos_x = np.zeros(n_car, dtype = np.int32)
    pos_y = np.zeros(n_car, dtype = np.int32)
    rands = np.zeros(n_car, dtype = np.float32)
    
    if spacing <= CAR_SIZE + SAFE_MARGIN:
        raise ValueError('The density setting is too high. Please lower the density.') 
    
    np.random.seed(0)
    counter = 0
    while counter < n_car:
        i = np.random.randint(NL)
        j = np.random.randint(NC)
        position_valid = check_spacing(road, (i, j), NC, cell_size)
        if position_valid:
            road[i][j] = counter + 1
            speed[i][j] = max(VMIN, min(VMAX, np.random.normal(VM, DV)))
            accel[i][j] = 0
            pos_x[counter] = j
            pos_y[counter] = i
            rands[counter] = np.random.uniform(0, 1)
            counter += 1
            
    return road, speed, accel, pos_x, pos_y, rands

@njit
def check_spacing(road, position, NC, cell_size):
    i = position[0]
    j = position[1]
    position_valid = True
    
    j_max = j + int((CAR_SIZE + SAFE_MARGIN)/cell_size) + 1
    j_min = j - int((CAR_SIZE + SAFE_MARGIN)/cell_size) - 1
    for j in range(j_min, j_max + 1):
        if j >= NC:
            j_temp = j - NC
        else:
            j_temp = j
        if road[i][j_temp]:
            position_valid = False
            return position_valid
    return position_valid


def initialize_no_jit(road_length, cell_size, traffic_density):
    
    NC = int(road_length * 1000/cell_size) # number of longitudinal cells
    traffic_density = min(traffic_density, 90) # traffic density shall not exceed a reasonable threshold
    n_car_per_lane = road_length * traffic_density # number of cars per lane
    spacing = np.int(NC / n_car_per_lane) # space between each car
    
    road = np.zeros((NL, NC), dtype = np.int32)
    speed = np.zeros((NL, NC), dtype = np.float32)
    accel = np.zeros((NL, NC), dtype = np.float32)
    
    n_car = int(NL * np.arange(0, NC - spacing, spacing).shape[0])
    pos_x = np.zeros(n_car, dtype = np.int32)
    pos_y = np.zeros(n_car, dtype = np.int32)
    rands = np.zeros(n_car, dtype = np.float32)
    
    if spacing <= CAR_SIZE + SAFE_MARGIN:
        raise ValueError('The density setting is too high. Please lower the density.') 
    
    np.random.seed(0)
    counter = 0
    while counter < n_car:
        i = np.random.randint(NL)
        j = np.random.randint(NC)
        position_valid = check_spacing_no_jit(road, (i, j), NC, cell_size)
        if position_valid:
            road[i][j] = counter + 1
            speed[i][j] = max(VMIN, min(VMAX, np.random.normal(VM, DV)))
            accel[i][j] = 0
            pos_x[counter] = j
            pos_y[counter] = i
            rands[counter] = np.random.uniform(0, 1)
            counter += 1
            
    return road, speed, accel, pos_x, pos_y, rands

def check_spacing_no_jit(road, position, NC, cell_size):
    i = position[0]
    j = position[1]
    position_valid = True
    
    j_max = j + int((CAR_SIZE + SAFE_MARGIN)/cell_size) + 1
    j_min = j - int((CAR_SIZE + SAFE_MARGIN)/cell_size) - 1
    for j in range(j_min, j_max + 1):
        if j >= NC:
            j_temp = j - NC
        else:
            j_temp = j
        if road[i][j_temp]:
            position_valid = False
            return position_valid
    return position_valid