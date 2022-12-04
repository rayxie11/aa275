import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import time

from kf import ExtendedKalmanFilter
from utils import get_datetime

class FigTrajectory(object):
    def __init__(self):
        plt.ion()
        self.figure, self.ax = plt.subplots(figsize=(5, 4))
        self.line, = self.ax.plot(0, 0)
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')

    def update(self, x, y, t):
        self.line.set_xdata(np.array(x))
        self.line.set_ydata(np.array(y))
        
        x_width = x.max() - x.min()
        y_width = y.max() - y.min()
        width = np.max([x_width, y_width])
        x_mean = (x.max() + x.min()) / 2
        y_mean = (y.max() + y.min()) / 2
        x_lo = x_mean - width / 2
        x_hi = x_mean + width / 2
        y_lo = y_mean - width / 2
        y_hi = y_mean + width / 2
        plt.xlim([x_lo, x_hi])
        plt.ylim([y_lo, y_hi])

        ti = (t.max()-t.min())*1e-9
        m = str(int(ti // 60)).rjust(2, '0')
        s = str(int(np.mod(ti, 60))).rjust(2, '0')
        plt.title(f'Trajectory, {m}:{s}', fontsize=20)

        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
        time.sleep(0.1)

class Simulator(object):
    def __init__(self,
                 ekf:  ExtendedKalmanFilter,
                 measurements: pd.DataFrame,
                 ground_truth: pd.DataFrame):
        # measurements should have the columns [frame_id, time_nsecs, 6 measurements from GPS (can be NaN), 1 measurement from CAM (can be NaN), 6 measurements from IMU (can be NaN)]
        # ground_truth should have the columns [frame_id, time_nsecs, x_true (can be NaN), y_true (can be NaN), z_true (can be NaN)]
        # All frame_id and time_nsecs that is present in measurements must be present in ground_truth, and vice versa!
        self.ekf = ekf
        self.trajectory = pd.DataFrame(columns={'time', 'x', 'y', 'z'})
        self.trajectory_fig = FigTrajectory()
        self.z = measurements
        self.gt = ground_truth

        self.current_index = -1
        self.last_time_nsecs = None
    
    def step(self):
        """ Step forward one sample """
        # get data for this time step
        self.current_index += 1
        measurement = self.z.iloc[self.current_index]
        time_nsecs = measurement['time']

        if self.last_time_nsecs != None:
            # predict
            self.ekf.predict(dt=((time_nsecs - self.last_time_nsecs) * 1e-9))
            # update
            x, y, z = self.ekf.update(measurement).tolist()
            # save
            self.trajectory.loc[len(self.trajectory.index)] = pd.Series({'time':time_nsecs, 'x':x, 'y':y, 'z':z})
            
        self.last_time_nsecs = time_nsecs

    def visualize(self):
        """ Plot current trajectory """
        self.trajectory_fig.update(self.trajectory['x'], self.trajectory['y'], self.trajectory['time'])
    
    def evaluate(self, fig_handle=None):
        """ Calculate performance metrics and plot final trajectory """
        # calculate performance metrics TODO

        # save trajectory 
        foldername = get_datetime()
        subprocess.call(f'mkdir output\{foldername}', shell=True)
        self.trajectory.to_csv(f'output/{foldername}/trajectory.csv')

        # show final trajectory
        plt.ioff()
        self.visualize()
        plt.savefig(f'output/{foldername}/trajectory.png')
        plt.show()