# AA275 Final Project: Cost-Efficient Localization Method using GPS, Camera, IMU, and Google Maps Data

## Introduction
This repository contains all the code for AA275 Final Project which uses GPS, camera, IMU and Google Maps data to do localization on autonomous vehicles

## File Explanation
`lanenet`: contains the LaneNet neural network for lane extraction, original repository: https://github.com/MaybeShewill-CV/lanenet-lane-detection

`load_measurements.py`: load and preprocess GPS, camera, IMU and Google Maps data

`kf.py`: extended kalman filter object

`gpx_interpolate.py`: GPS data interpolation using piece-wise cubic Hermite splines 

`main.py`: main execution file to calculate the positions

`utils.py`: utility functions for coordinate transformation, data extraction and preprocessing

## Installation
We recommend creating this work space within a virtual conda environment. The requirements are specified within `requirements.txt`. Use command ```pip3 install -r requirements.txt``` to install all the required dependencies. 

## Dataset
All the datafiles should be constructed in the same directory as this repository. The Ford AV dataset can be downloaded from here: https://avdata.ford.com/home/default.aspx. The csv files are extracted from the Ford AV dataset rosbag files. Please refer to that repository for more detailed instructions: https://github.com/Ford/AVData.

## Test
Open `load_measurements.py`and run each cell one by one. Then, run `main.py` to see the sensor fusion results.
