import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time
import pickle


class DataPlot():
    def __init__(self, max_velocity=40, figsize=(10,5), dpi=200, save=False):
        self.fig, (self.ax1, self.ax2) = plt.subplots(ncols=2, nrows=1)
        self.figsize = figsize
        self.dpi = dpi
        self.time_step = np.empty([2])
        self.dis = np.empty([2])
        self.velocity = None
        self.max_vel = max_velocity
        self.save = save

    def plot(self, object_pts, colors=["r", "g"], pause=0.1):
        self.fig.set_size_inches(self.figsize)
        self.fig.set_dpi(self.dpi)
        self.ax2.set_ylim([0, self.max_vel])
        self.ax2.set_xlim(0,10)

        coors = np.array(list(object_pts.keys()), dtype=np.int)
        
        time_stamp = list(object_pts.values())
        
        new_sum_dis = np.array([np.linalg.norm((coors[i] - coors[i+1])) for i in range(len(coors)-1)], dtype=np.float32).sum()
        new_sum_time = np.array([x - time_stamp[0] for x in time_stamp], dtype=np.float32)[1:]

        self.dis = np.append(self.dis, new_sum_dis)
        self.time_step = np.append(self.time_step, new_sum_time[-1])
        dis_text = str(int(self.dis[-1])) + "px"


        self.velocity = (self.dis[-1] - self.dis[-2])/(self.time_step[-1] - self.time_step[-2])
        vel_text = str(int(self.velocity)) + "px/s"

        self.ax1.plot(self.time_step, self.dis, colors[0])
        self.ax1.set_title(f"Total distance: {dis_text}")
        if self.velocity > self.max_vel * 0.7:
            color = "red"
        elif self.velocity > self.max_vel * 0.3:
            color = "blue"
        else:
            color = "green"

        self.ax2.bar(5.0, self.velocity, 5, color=color)
        self.ax2.text(5*0.8, self.velocity+1, vel_text, style="italic")

            
        plt.draw()
        if self.save:
            self.fig.savefig("./output/plot/tracking_plot.png")

        plt.pause(pause)

        self.ax2.cla()

