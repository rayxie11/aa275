import subprocess
subprocess.call('cls', shell=True)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm

from sim import Simulator
from kf import ExtendedKalmanFilter
from utils import quat2euler


def estimate_process_noise(meas: pd.DataFrame):
    """ Estimate the intensity of the process noise. 
    Based on comments from Bar-Shalom et. al. chapter 6 
    Estimation with Applications to Tracking and Navigation: Theory, Algorithms and Software, 2001 """
    # "The changes in acceleration       over a sampling period T are of the order of sqrt(q33*T)"
    # "The changes in (angluar) velocity over a sampling period T are of the order of sqrt(q22*T)"
    
    delta_t = (meas['time'].diff() * 1e-9).mean()

    delta_a = meas['imu_lin_acc_x'].diff().apply(abs).mean()
    q33 = delta_a**2 / delta_t

    delta_ang_v = meas['imu_ang_vel_x'].diff().apply(abs).mean()
    q22 = delta_ang_v**2 / delta_t

    return q22, q33

def get_ekf_params(meas: pd.DataFrame):
    # initial state
    meas0 = meas.iloc[0]
    meas1 = meas.iloc[1]
    meas2 = meas.iloc[2]

    theta0, phi0, varphi0 = quat2euler((0.004144869, 6.85e-5, 0.855333038, 0.518061975)) # TODO: is this right!

    x0 = {
        'x': meas0['gps_pos_x'], 
        'y': meas0['gps_pos_y'],
        'z': meas0['gps_pos_z'],
        'x_d': (meas1['gps_pos_x'] - meas0['gps_pos_x']) / ((meas1['time'] - meas0['time'])*1e-9),
        'y_d': (meas1['gps_pos_y'] - meas0['gps_pos_y']) / ((meas1['time'] - meas0['time'])*1e-9),
        'z_d': (meas1['gps_pos_z'] - meas0['gps_pos_z']) / ((meas1['time'] - meas0['time'])*1e-9),
        'x_dd': 0, 
        'y_dd': 0, 
        'z_dd': 0, 
        'theta': theta0,
        'phi': phi0,
        'varphi': varphi0,
        'theta_d':  meas0['imu_ang_vel_x'],
        'phi_d':    meas0['imu_ang_vel_y'],
        'varphi_d': meas0['imu_ang_vel_z']
    }
    
    # initial covariance
    P0 = {
        'pos':      100 ** 2,
        'vel':      30 ** 2,
        'acc':      10 ** 2,
        'ang':      6 **2,
        'ang_vel' : 6 **2
    }
    
    # process noise intensity
    q22, q33 = estimate_process_noise(meas)
    q = {
        'linear':  q33,    # "The changes in acceleration over a sampling period T are of the order of sqrt(q33*T)"
        'angular': q22    # "The changes in velocity     over a sampling period T are of the order of sqrt(q22*T)"
    }
    
    # Measurement noise covariance TODO
    R = {
        'sigma2_CAM_pos':    3    ** 2,
        'sigma2_GPS_pos':    0.5  ** 2,
        'sigma2_GPS_ang':    0.5  ** 2,
        'sigma2_IMU_acc':    0.5  ** 2,
        'sigma2_IMU_angvel': 0.01 ** 2,
    }

    return x0, P0, q, R

def setup():
    logging.basicConfig(level=logging.DEBUG,
                        format='[%(asctime)s.%(msecs)03d] - %(message)s', datefmt='%H:%M:%S')
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

    plt.ion()



def main():
    logging.info('Reading data ...')
    meas = pd.read_csv('../data/measurements.csv').iloc[0:10000]
    gt   = pd.read_csv('../data/ground_truth.csv')

    logging.info('Creating EKF and simulator ...')
    x0, P0, q, R = get_ekf_params(meas)
    ekf  = ExtendedKalmanFilter(x0, P0, q, R)
    sim  = Simulator(ekf, meas, gt)

    logging.info('Starting simulation')
    num_steps = len(meas.index)
    with tqdm(range(1000)) as progbar:
        for step in range(num_steps):
            if np.mod(step+1, num_steps // 1000) == 0:
                progbar.update()
            
            sim.step()

            if np.mod(step+1, num_steps // 100) == 0:
                sim.visualize()
    
    logging.info('Simulation done')
    
    sim.visualize(save=True)

if __name__ == '__main__':
    setup()
    main()