import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle 

from utils import *

# Get data directory
current_dir = os.getcwd()
par_dir = os.path.dirname(current_dir)
csv_data_dir = par_dir+"/data/csv_data"
img_data_dir = par_dir+"/data/img_data"
bi_dir = par_dir+"/data/bi_img_data"
google_maps_reference = par_dir+"/data/map_to_gpx.gpx"

# csv data file needed
csv_file_name = ["/gps.csv", "/imu.csv", "/pose_localized.csv"]

# Extract GPS data
xyz1,alt = get_gps_data(csv_data_dir+csv_file_name[0])

# Plot GPS data
#plt.scatter(xyz1[:,0],xyz1[:,1])


# Extract Google Maps reference data
xyz2 = get_google_data(google_maps_reference, alt)
#plt.scatter(xyz2[:,0],xyz2[:,1])
#plt.show()

# Extract lanes from binary images
#lanes = extract_lane(bi_dir, lane_len=40, e=0.5)
#with open("lanes_data", "wb") as fp:
    #pickle.dump(lanes,fp)
lanes = None
with open("lanes_data", "rb") as fp:
    lanes = pickle.load(fp)
print(len(lanes))

x = gen_relative_pos(lanes)
print(x)

#print(xyz.shape)

# Get lanes from image data
#img = cv2.imread(img_data_dir+"/1508987519235232.png")
