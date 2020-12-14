import sys
import time
import numpy as np
from integrate import traffic_cpu_lc, traffic_cuda_lc

def run(sim_steps = 1, road_length = 1, cell_size = 1, traffic_density = 30):
    
    sys.stdout.write('\n')
    sys.stdout.write('####### Model Info \n')
    sys.stdout.write('Total simumaltion steps: {}\n'.format(sim_steps))
    sys.stdout.write('Road length: {} km\n'.format(road_length))
    sys.stdout.write('Cell size: {} m\n'.format(cell_size))
    sys.stdout.write('Traffic Density: {} veh/km \n'.format(traffic_density))
    
    road, speed, accel = traffic_cuda_lc(sim_steps, road_length, cell_size, traffic_density, verbose = True)
    road0, speed0, accel0 = traffic_cpu_lc(sim_steps, road_length, cell_size, traffic_density, verbose = True)
    
    ncar = np.sum(np.array(road, dtype = np.bool))
    vm = np.sum(speed)/ncar * 3.6
    vm_mile = vm/1.6
    
    ncar0 = np.sum(np.array(road0, dtype = np.bool))
    vm0 = np.sum(speed0)/ncar0 * 3.6
    vm_mile0 = vm0/1.6
    
    sys.stdout.write('####### Verification \n')
    sys.stdout.write('(CUDA) Total number of vehicles: {}\n'.format(ncar))
    sys.stdout.write('(CPU)  Total number of vehicles: {}\n'.format(ncar0))
    sys.stdout.write('(CUDA) Average speed: {:.1f} km/h or {:.1f} mph\n'.format(vm, vm_mile))
    sys.stdout.write('(CPU)  Average speed: {:.1f} km/h or {:.1f} mph\n'.format(vm0, vm_mile0))
    sys.stdout.write('(CUDA) Traffic Flow: {:.0f} veh/h \n'.format(traffic_density * vm))
    sys.stdout.write('(CPU)  Traffic Flow: {:.0f} veh/h \n'.format(traffic_density * vm0))
    sys.stdout.write('\n')