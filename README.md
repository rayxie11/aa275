# AA275 Final Project: Cost-Efficient Localization Method using GPS, Camera, IMU, and Google Maps Data

## Introduction
This repository contains all the code for AA275 Final Project which uses GPS, camera, IMU and Google Maps data to do localization on autonomous vehicles

## File Explanation
`lanenet`: contains the LaneNet neural network for lane extraction, original repository: https://github.com/MaybeShewill-CV/lanenet-lane-detection

`load_measurements.py`: load and preprocess GPS, camera, IMU and Google Maps data

`kf.py`: extended kalman filter object

`gpx_interpolate.py`: GPS data interpolation using piece-wise cubic Hermite splines 

`utils.py`: utility functions for coordinate transformation, data extraction and preprocessing
