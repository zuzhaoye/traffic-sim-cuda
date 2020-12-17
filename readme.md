This is an implementation of Numba CUDA to accelerate agent-based traffic flow simulation. Forward driving and lane changing are included in the model.

![road_gif](imgs/animation.gif)

### Dependencies
- python 3.7.4
- [numpy 1.19.2](https://numpy.org/install/)
- [numba 0.51.1](https://numba.readthedocs.io/en/stable/user/installing.html)
- [cudatoolkit 10.2](https://developer.nvidia.com/cuda-downloads)

Click the link for installation guide.

### Demo File

A Jupyter notebook demo file is included in this repo. Detailed instructions and case studies are included (highly recommended). 

### Usage

To use from command line, in the root folder, activate Python:
 ```
 python
 ```
Then import ```run``` module:
```
from traffic import run
```
Then just:
```
run()
```

There are 4 parameters of the ```run()```: 
- ```sim_steps```         ```(10)```, (1 - any integer) (Use 0 can cause an error, which will be addressed later)
- ```road_length```      ```(1km)```, (0.1 - 1000)
- ```cell_size```        ```(1m)```, (0.1 - 1)
- ```traffic_density```  ```(30 veh/km/lane)```, (5 - 80)

Note, the parentheses means: 
- ```Parameter```  ```(default value)```, (recommended & tested range)

An example: simulate 5 steps, with road length 0.5:
```
run(5, 0.5)
```
Another example: cell size = 0.1, everything else default:
```
run(cell_size = 0.1)
```

### Expected Output
- Model Info
- CUDA Computation Info
- CPU Computation Info
- Verification
