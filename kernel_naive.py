from numba import cuda
from support_naive import check_front_d, cal_displacement_d, update_cell_d
from CONS import *

@cuda.jit
# Rule for moving cars forward
def detect_d(road, speed, accel, road_record, speed_record, accel_record, cell_size):
    
    i = cuda.threadIdx.y
    j0 = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    stride = cuda.blockDim.x * cuda.gridDim.x
    
    NC = road.shape[1]
    
    if j0 < NC and i < NL:
        for j in range(j0, NC, stride):
            if road[i][j]:
                # Calculate distance to the front car
                df = check_front_d(road, (i,j), NC, cell_size)
                j_new = cal_displacement_d(j, speed[i][j], df, NC, cell_size)
                speed_new, accel_new\
                    = update_cell_d(speed[i][j], accel[i][j], df)

                road_record[i][j] = j_new
                speed_record[i][j] = speed_new 
                accel_record[i][j] = accel_new
            else:
                road_record[i][j] = -1
                speed_record[i][j] = -1 
                accel_record[i][j] = -1

@cuda.jit
# Rule for moving cars forward
def move_d(road, speed, accel, road_record, speed_record, accel_record):
    
    i = cuda.threadIdx.y
    j0 = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    stride = cuda.blockDim.x * cuda.gridDim.x
    
    NC = road.shape[1]
    
    if j0 < NC and i < NL:
        for j in range(j0, NC, stride):
            j_new = road_record[i][j]
            if j_new != -1:
                
                temp = road[i][j]                
                road[i][j] = 0
                speed[i][j] = 0
                accel[i][j] = 0
                
                road[i][j_new] = temp
                speed[i][j_new] = speed_record[i][j]
                accel[i][j_new] = accel_record[i][j]