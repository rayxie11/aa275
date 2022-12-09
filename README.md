# AA275 Final Project: Cost-Efficient Localization Method using GPS, Camera, IMU, and Google Maps Data

## Introduction
This repository contains all the code for AA275 Final Project which uses GPS, camera, IMU and Google Maps data to do localization on autonomous vehicles

## File Explanation
`lanenet`: contains the LaneNet neural network for lane extraction, original repository: https://github.com/MaybeShewill-CV/lanenet-lane-detection

`load_measurements.py`: load and preprocess GPS, camera, IMU and Google Maps data

`kf.py`: extended kalman filter object

`sim.py`: simulator object, uses EKF object to simulate trajectory

`gpx_interpolate.py`: GPS data interpolation using piece-wise cubic Hermite splines 

`main.py`: main execution file to calculate the positions of the vehicle during the trip, calls simulator in sim.py

`utils.py`: utility functions for coordinate transformation, data extraction and preprocessing

`requirements.txt`: all required dependencies and packages

## Installation
We recommend creating this work space within a virtual conda environment. The requirements are specified within `requirements.txt`. Use command ```pip3 install -r requirements.txt``` to install all the required dependencies. 

Some changes need to be made in order for LaneNet to run on local machine:

1. Add these lines to the beginning of `tools/evaluate_lanenet_on_tusimple.py`
```
import sys
sys.path.append('PATH/TO/THIS/REPO/lanenet')
sys.path.append('PATH/TO/THIS/REPO/lanenet/config')
sys.path.append('PATH/TO/THIS/REPO/lanenet/data_provider')
sys.path.append('PATH/TO/THIS/REPO/lanenet/lanenet_model')
sys.path.append('PATH/TO/THIS/REPO/lanenet/tools')
```

2. Change "lanenet_cfg" at line 227 in `local_utils/config_utils/parse_config_utils.py` to
```
lanenet_cfg = Config(config_path='PATH/TO/THIS/REPO/lanenet/config/tusimple_lanenet.yaml')
```

3. Add this line to the beginning of `lanenet_model/lanenet_postprocess.py`
```
PATH = "C:PATH/TO/THIS/REPO/lanenet"
```
And then change line 259 to
```
def __init__(self, cfg, ipm_remap_file_path=PATH+"/data/tusimple_ipm_remap.yml"):
```

## Dataset
All the datafiles should be constructed in the same directory as this repository. The Ford AV dataset can be downloaded from here: https://avdata.ford.com/home/default.aspx. The csv files are extracted from the Ford AV dataset rosbag files. Please refer to that repository for more detailed instructions: https://github.com/Ford/AVData.

## Test
Open `load_measurements.py` and run each cell one by one. Then, run `main.py` to see the sensor fusion results.
