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
        time.sleep(0.01)

class Simulator(object):
    def __init__(self,
                 ekf:  ExtendedKalmanFilter,
                 measurements: pd.DataFrame,
                 ground_truth: pd.DataFrame):

        self.ekf = ekf
        self.trajectory = pd.DataFrame(columns={'time', 'x', 'y', 'z'})
        self.trajectory_fig = FigTrajectory()
        self.z = measurements
        self.gt = ground_truth
        self.error_xy = pd.DataFrame(columns={'time', 'error_xy'})

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
    
    def evaluate(self, final=False):
        """ Calculate performance metrics and plot final trajectory """
        # calculate performance metrics
        if len(self.trajectory) > 0:
            latest_estimate = self.trajectory.loc[len(self.trajectory)-1]
            closest_time = (self.gt['time'] - latest_estimate['time']).abs().argmin()
            closest_gt = self.gt.iloc[closest_time]
            if closest_time < 1e8: # 0.1 s
                error_x = np.abs(latest_estimate['x'] - closest_gt['x_gt'])
                error_y = np.abs(latest_estimate['y'] - closest_gt['y_gt'])
                error_xy = np.sqrt(error_x**2 + error_y**2)
                self.error_xy.loc[len(self.error_xy.index)] = pd.Series({'time':latest_estimate['time'], 'error_xy':error_xy})

        if final:
            # save trajectory 
            foldername = get_datetime()
            subprocess.call(f'mkdir output\{foldername}', shell=True)
            self.trajectory.to_csv(f'output/{foldername}/trajectory.csv')

            # show final trajectory
            plt.ioff()
            self.visualize()
            plt.savefig(f'output/{foldername}/trajectory.png')
            plt.show()

            # plot final error_xy
            plt.figure()
            plt.plot(self.error_xy['time'], self.error_xy['error_xy'])
            plt.savefig(f'output/{foldername}/error_xy.png')
            plt.show()
            mean_error = self.error_xy['error_xy'].mean()
            print(f'Mean error: {mean_error:.2f} m')