import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2

from utils import *

# Get data directory
current_dir = os.getcwd()
par_dir = os.path.dirname(current_dir)
csv_data_dir = par_dir+"/data/csv_data"
img_data_dir = par_dir+"/data/img_data"
bi_dir = par_dir+"/data/bi_img_data"

# csv data file needed
csv_file_name = ["/gps.csv", "/imu.csv", "/pose_localized.csv"]

# Extract GPS data
xyz = get_gps_data(csv_data_dir+csv_file_name[0])

# Plot GPS data
#plt.scatter(xyz[:,0],xyz[:,1])
#plt.show()

#match_bi_img(bi_dir,img_data_dir)

x = extract_lane(bi_dir)


#print(xyz.shape)

# Get lanes from image data
#img = cv2.imread(img_data_dir+"/1508987519235232.png")
