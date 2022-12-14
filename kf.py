#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from scipy.linalg import block_diag

def blockdiag3(*args):
    """ Create a blockdiagonal matrix where block i is I_{3x3} * arg[i] """
    n = len(args) * 3
    mat = np.zeros((n, n))
    for i, arg in enumerate(args):
        mat[i*3:(i+1)*3, i*3:(i+1)*3] = np.eye(3)*arg
    return mat

def oneD_to_threeD(mat):
    """ Insert zeros to make a 1D transformation matrix into a 3D transformation matrix """
    n = mat.shape[1] * 3
    result = np.empty((n, n))
    for i, row in enumerate(mat):
        new_row = np.empty(n)
        for j, elem in enumerate(row):
            new_row[j*3 + 0] = elem
            new_row[j*3 + 1] = 0
            new_row[j*3 + 2] = 0
        result[i*3 + 0] = new_row
        result[i*3 + 1] = np.hstack((np.zeros(1), new_row[:-1]))
        result[i*3 + 2] = np.hstack((np.zeros(2), new_row[:-2]))
    return result

def create_Q(T, q_linear, q_angular):
    """ 
    Create process noise covriance matrix Q, as derived in Bar-Shalom et. al. page 272 
    Estimation with Applications to Tracking and Navigation: Theory, Algorithms and Software, 2001
    """
    block11 = oneD_to_threeD(np.array([[(T**5)/20, (T**4)/8, (T**3)/6],
                                       [(T**4)/8,  (T**3)/3, (T**2)/2],
                                       [(T**3)/6,  (T**2)/2,  T      ]]))
    block22 = oneD_to_threeD(np.array([[(T**3)/3, (T**2)/2],
                                       [(T**2)/2,  T      ]]))
    block12 = np.zeros((9, 6))
    block21 = np.zeros((6, 9))
    
    block1 = np.hstack((block11 * q_linear, block12))
    block2 = np.hstack((block21, block22 * q_angular))

    Q = np.vstack((block1, block2)) * int(5e3)
    return Q

def create_F(T):
    """ Create state transition matrix F """
    block11 = np.vstack((np.hstack((np.eye(3),        np.eye(3)*T,      np.eye(3)*(T**2)/2)),
                         np.hstack((np.zeros((3, 3)), np.eye(3),        np.eye(3)*T       )),
                         np.hstack((np.zeros((3, 3)), np.zeros((3, 3)), np.eye(3)         )) ))
    block22 = np.vstack((np.hstack((np.eye(3),        np.eye(3)*T)),
                         np.hstack((np.zeros((3, 3)), np.eye(3)  )) ))
    F = np.vstack((np.hstack((block11,          np.zeros((9, 6)))),
                   np.hstack((np.zeros((6, 9)), block22         )) ))
    return F

class ExtendedKalmanFilter(object):
    def __init__(self, x0: dict, P0: dict, q: dict, R: dict):
        # x = [x, y, z, x_d, y_d, z_d, x_dd, y_dd, z_dd, theta, phi, varphi, theta_d, phi_d, varphi_d]
        self.x = np.array([x0['x'],       x0['y'],     x0['z'],
                           x0['x_d'],     x0['y_d'],   x0['z_d'],
                           x0['x_dd'],    x0['y_dd'],  x0['z_dd'], 
                           x0['theta'],   x0['phi'],   x0['varphi'],
                           x0['theta_d'], x0['phi_d'], x0['varphi_d']])
        self.P = blockdiag3(P0['pos'], P0['vel'], P0['acc'], P0['ang'], P0['ang_vel'])
        self.Q = None   # depends on sample time so is created in predict()
        self.F = None   # depends on sample time so is created in predict()
        self.H_CAM     = np.hstack((np.eye(2), np.zeros((2, 13))))
        self.H_GPS_pos = np.hstack((np.eye(3), np.zeros((3, 12))))
        self.H_GPS_ang = np.hstack((np.zeros((3, 9)), np.eye(3), np.zeros((3, 3))))
        self.H_IMU     = np.vstack((np.zeros((3, 15)),                           # OBS: depends on measurements so is populated in self.update()
                                np.hstack((np.zeros((3, 12)), np.eye(3))) ))
        # H is stacked like this, depending on which measurements are available
        # self.H = np.vstack((self.H_CAM,
                            # self.H_GPS,
                            # self.H_IMU))
        self.R_dict = R
        # R is stacked like this, depending on which measurements are available
        # self.R = blockdiag3(R['sigma2_CAM_pos'], R['sigma2_GPS_pos'], R['sigma2_GPS_ang'], R['sigma2_IMU_acc'], R['sigma2_IMU_angvel'])
        self.q = q  # process noise intensities

    def predict(self, dt: float):
        self.Q = create_Q(dt, self.q['linear'], self.q['angular'])
        self.F = create_F(dt)
        self.x = np.matmul(self.F, self.x)
        self.P = np.matmul(np.matmul(self.F, self.P), self.F.T) + self.Q

    def update(self, z: pd.Series):
        # Pick z, H, and R depending on which measurements are available
        zs, Hs, Rs = [], [], []
        if z['has_cam']:
            phi, varphi = self.x[10], self.x[11]
            z_cam_x = z['road_x'] + ((((z['n_lanes']-1)/2)+z['curr_lane'])*z['LANEWIDTH'] + z['delta']) * np.cos(phi)*np.cos((np.pi/2)-varphi)
            z_cam_y = z['road_y'] + ((((z['n_lanes']-1)/2)+z['curr_lane'])*z['LANEWIDTH'] + z['delta']) * np.cos(phi)*np.sin((np.pi/2)-varphi)
            zs.extend([z_cam_x, z_cam_y])
            Hs.append(self.H_CAM)
            R_CAM = np.eye(2) * self.R_dict['sigma2_CAM_pos']
        if z['has_gps']:
            zs.extend(z[['gps_pos_x', 'gps_pos_y', 'gps_pos_z']].tolist())
            Hs.append(self.H_GPS_pos)
            Rs.append(self.R_dict['sigma2_GPS_pos'])
        if z['has_angle']:
            zs.extend(z[['gps_ang_theta', 'gps_ang_phi', 'gps_ang_varphi']].tolist())
            Hs.append(self.H_GPS_ang)
            Rs.append(self.R_dict['sigma2_GPS_ang'])
        if z['has_imu']:
            zs.extend(z[['imu_lin_acc_x', 'imu_lin_acc_y', 'imu_lin_acc_z', 'imu_ang_vel_x', 'imu_ang_vel_y', 'imu_ang_vel_z']].tolist())
            self.H_IMU[0:3] = self.h_imu_acc(z)
            Hs.append(self.H_IMU)
            Rs.append(self.R_dict['sigma2_IMU_acc'])
            Rs.append(self.R_dict['sigma2_IMU_angvel'])
        self.H = np.vstack(Hs)
        self.R = blockdiag3(*Rs)
        if z['has_cam']:
            self.R = block_diag(R_CAM, self.R)
        z = np.array(zs)

        # update
        K = np.matmul(np.matmul(self.P, self.H.T), np.linalg.inv(self.R + np.matmul(np.matmul(self.H, self.P), self.H.T)))
        self.x = self.x + np.matmul(K, (z - np.matmul(self.H, self.x)))
        self.P = np.matmul((np.eye(15) - np.matmul(K, self.H)), self.P)

        return self.x[0:3]

    def h_imu_acc(self, z: pd.Series):
        """ Get the non-linear part of H_IMU """
        phi, varphi = self.x[10], self.x[11]

        res = np.zeros((3, 15))
        res[0, 6] = 1 / ( np.cos(phi)*np.cos(varphi))
        res[1, 7] = 1 / ( np.cos(phi)*np.sin(varphi))
        res[2, 8] = 1 / (-np.sin(phi))

        return res