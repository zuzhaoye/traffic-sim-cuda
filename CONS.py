# This file contains necessary parameters for the model

NL = 4 # number of lanes

VMAX = 30 # max speed，unit m/s，~108km/h ~ 67.5 mph
VMIN = 0 
VM = 5 # initial mean speed，unit m/s
DV = 5 # std for initial speed，unit m/s
CAR_SIZE = 5 # length of a car， 5m

TR = 2 # in sec, reaction time of a human driver, in seconds
MAX_FORWARD = 100 # in m, the car will be impacted by a car as much as 100m in front of it 
ALPHA = 1 # parameter for updating acceleration
SAFE_MARGIN = 3 # in m, minimum distance between cars

PC = 0.5 # possibility of lane changing
TC = 1 # in sec, time gain for lane changing

DT = 0.1 # in sec, step time