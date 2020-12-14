import sys
import time
import numpy as np
from numba import cuda
from initialize import initialize
from support import forward_h, changeLane_h
from kernel import detect_d, move_d, detect_lane_change_d, perform_lane_change_d
from CONS import *

############################################
########  Part 1. CPU  #####################
def traffic_cpu(sim_steps = 1, road_length = 1, cell_size = 1, traffic_density = 30):
    
    # initilize matrices
    road, speed, accel, pos_x, pos_y, _ = initialize(road_length, cell_size, traffic_density)

    counter = 0
    while counter < sim_steps:
            
        road, speed, accel, pos_x =\
            forward_h(road, speed, accel, pos_x, pos_y, cell_size)
        
        counter += 1
    return road, speed, accel

def traffic_cpu_lc(sim_steps = 1, road_length = 1, cell_size = 1, traffic_density = 30, verbose = False):
    
    # initilize matrices
    t0 = time.time()
    road, speed, accel, pos_x, pos_y, rands = initialize(road_length, cell_size, traffic_density)
    t1 = time.time()
    
    counter = 0
    while counter < sim_steps:
        
        road, speed, accel, pos_y =\
            changeLane_h(road, speed, accel, pos_x, pos_y, rands, cell_size)
        
        road, speed, accel, pos_x =\
            forward_h(road, speed, accel, pos_x, pos_y, cell_size)
        
        counter += 1
    t2 = time.time()
    
    t_ini = t1 - t0
    t_sim = t2 - t1
    t_loop = t_sim / sim_steps
    t_total = t2 - t0

    if verbose:
        sys.stdout.write('####### CPU Computation Info\n')
        sys.stdout.write('Intializing matrices: {:.1e} sec\n'.format(t_ini))
        sys.stdout.write('Simulation time: {:.1e} sec\n'.format(t_sim))
        sys.stdout.write('Simulation time per step {:.1e} sec\n'.format(t_loop))
        sys.stdout.write('Total time: {:.1e} sec\n'.format(t_total))
        sys.stdout.write('\n')
        
    return road, speed, accel

#############################################
########  Part 2. CUDA  #####################
def traffic_cuda(sim_steps = 1, road_length = 1, cell_size = 1, traffic_density = 30):
    
    # initilize matrices
    road, speed, accel, pos_x, pos_y, _ = initialize(road_length, cell_size, traffic_density)
    
    # send matrices to device
    road_d = cuda.to_device(road)
    speed_d = cuda.to_device(speed)
    accel_d = cuda.to_device(accel)
    pos_x_d = cuda.to_device(pos_x)
    pos_y_d = cuda.to_device(pos_y)
    
    pos_x_cp_d = cuda.to_device(pos_x.copy())
    pos_y_cp_d = cuda.to_device(pos_y.copy())
    
    # define blocks
    dim_block = 256
    dim_grid = 32
    
    counter = 0
    while counter < sim_steps:
        
        detect_d[dim_grid, dim_block](road_d, speed_d, accel_d, pos_x_d, pos_y_d, pos_x_cp_d, cell_size)

        move_d[dim_grid, dim_block](road_d, pos_x_d, pos_y_d, pos_x_cp_d)
        
        counter += 1
     
    #cudaDeviceSync
    cuda.synchronize()
    road = road_d.copy_to_host()
    speed = speed_d.copy_to_host()
    accel = accel_d.copy_to_host()
    
    return road, speed, accel

def traffic_cuda_lc(sim_steps = 1, road_length = 1, cell_size = 1, traffic_density = 30, verbose = False):
    
    # initilize matrices
    t0 = time.time()
    road, speed, accel, pos_x, pos_y, rands = initialize(road_length, cell_size, traffic_density)
    t1 = time.time()
    
    # send matrices to device
    road_d = cuda.to_device(road)
    speed_d = cuda.to_device(speed)
    accel_d = cuda.to_device(accel)
    pos_x_d = cuda.to_device(pos_x)
    pos_y_d = cuda.to_device(pos_y)
    rands_d = cuda.to_device(rands)
    
    pos_x_cp_d = cuda.to_device(pos_x.copy())
    pos_y_cp_d = cuda.to_device(pos_y.copy())
    t2 = time.time()
    
    # define blocks
    dim_block = 256
    dim_grid = 32
    
    t_detect = 0
    t_move = 0
    
    counter = 0
    while counter < sim_steps:
        counter += 1
        
        detect_lane_change_d[dim_grid, dim_block](road_d, speed_d, accel_d, pos_x_d, pos_y_d, pos_y_cp_d, rands_d, cell_size)
        perform_lane_change_d[dim_grid, dim_block](road_d, pos_x_d, pos_y_d, pos_y_cp_d)
        
        detect_d[dim_grid, dim_block](road_d, speed_d, accel_d, pos_x_d, pos_y_d, pos_x_cp_d, cell_size)
        move_d[dim_grid, dim_block](road_d, pos_x_d, pos_y_d, pos_x_cp_d)
    t3 = time.time()
    
    #cudaDeviceSync
    cuda.synchronize()
    road = road_d.copy_to_host()
    speed = speed_d.copy_to_host()
    accel = accel_d.copy_to_host()
    t4 = time.time()
    
    t_ini = t1 - t0
    t_h2d = t2 - t1
    t_kernel = t3 - t2
    t_loop = t_kernel / sim_steps
    t_d2h = t4 - t3
    t_total = t4 - t0

    if verbose:
        sys.stdout.write('\n')
        sys.stdout.write('####### CUDA Computation Info\n')
        sys.stdout.write('Intializing matrices: {:.1e} sec\n'.format(t_ini))
        sys.stdout.write('Host to device transfer: {:.1e} sec\n'.format(t_h2d))
        sys.stdout.write('Kernel time: {:.1e} sec\n'.format(t_kernel))
        sys.stdout.write('Kernel time per step {:.1e} sec\n'.format(t_loop))
        sys.stdout.write('Device to host transfer: {:.1e} sec\n'.format(t_d2h))
        sys.stdout.write('Total time: {:.1e} sec\n'.format(t_total))
        sys.stdout.write('\n')
    return road, speed, accel