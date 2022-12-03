import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from kf import ExtendedKalmanFilter 

class FigTrajectory(object):
    def __init__(self):
        self.figure, self.ax = plt.subplots()
        self.line, = self.ax.plot(0, 0)
        plt.title('Trajectory', fontsize=20)
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')

    def update(self, x, y):
        plt.plot(x, y)
        plt.show()
        # self.line.set_xdata(x)
        # self.line.set_ydata(y)
        # self.figure.canvas.draw()
        # self.figure.canvas.flush_events()

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

        # predict
        if self.last_time_nsecs != None:
            self.ekf.predict(dt=((time_nsecs - self.last_time_nsecs) * 1e-9))
            x, y, z = self.ekf.update(measurement).tolist()
            self.trajectory.loc[len(self.trajectory.index)] = [time_nsecs, x, y, z]
            
        self.last_time_nsecs = time_nsecs

    def visualize(self, save:bool=False):
        """ Plot current trajectory """
        self.trajectory_fig.update(self.trajectory['x'], self.trajectory['y'])
        if save:
            plt.savefig('output/fig1.png')
    
    def evaluate(self, fig_handle=None):
        """ Calculate performance metrics and plot final trajectory """
        # calculate performance metrics TODO

        # plot final trajectory
        self.visualize(fig_handle) if fig_handle else self.visualize()
        return fig_handle