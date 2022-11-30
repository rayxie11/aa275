import numpy as np
import os
import pptk
import open3d as o3d
from gpx_converter import Converter



current_dir = os.getcwd()
par_dir = os.path.dirname(current_dir)
map_dir1 = par_dir+"/data/map_data/ground_reflectivity/-64_448.pcd"
map_dir2 = par_dir+"/data/map_data/ground_reflectivity/-64_512.pcd"
gpx_dir = par_dir+"/data/map_to_gpx.gpx"

dic = Converter(input_file=gpx_dir).gpx_to_dictionary(latitude_key='latitude', longitude_key='longitude')
lat = np.array(dic["latitude"])
lon = np.array(dic["longitude"])

def get_cartesian(lat,lon):
    lat, lon = np.deg2rad(lat), np.deg2rad(lon)
    R = 6371 # radius of the earth
    x = R * np.cos(lat) * np.cos(lon)
    y = R * np.cos(lat) * np.sin(lon)
    z = R *np.sin(lat)
    return x,y,z
import matplotlib.pyplot as plt

x,y,z = get_cartesian(lat,lon)
plt.scatter(x,y)
plt.show()
print(lat.shape, lon.shape)