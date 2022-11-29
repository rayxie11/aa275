import numpy as np
import os
import pandas as pd
import cv2
from tqdm import tqdm

# Get directory
current_dir = os.getcwd()
par_dir = os.path.dirname(current_dir)

def lla_to_ecef(lat, lon, alt):
    """
    LLA to ECEF conversion.

    Parameters
    lat : Latitude in degrees (N°).
    lon : Longitude in degrees (E°).
    alt : Altitude in meters.

    Returns
    ecef : ECEF coordinates corresponding to input LLA.
    """
    A = 6378137  # Semi-major axis (radius) of the Earth [m].
    E1SQ = 6.69437999014*0.001  # First esscentricity squared of Earth (not orbit).
    lat = np.deg2rad(lat)
    lon = np.deg2rad(lon)
    xi = np.sqrt(1-E1SQ*np.sin(lat)**2)
    x = (A/xi+alt)*np.cos(lat)*np.cos(lon)
    y = (A/xi+alt)*np.cos(lat)*np.sin(lon)
    z = (A/xi*(1-E1SQ)+alt)*np.sin(lat)
    ecef = np.array([x,y,z]).T
    return ecef

def get_gps_data(file_loc):
    """
    Returns GPS data from file

    Parameters
    file_loc: gps data file location

    Returns
    xyz: np array of locations in cartesian coordinate system
    """
    gps = pd.read_csv(file_loc)
    lat = gps.loc[:,"latitude"].to_numpy()
    lon = gps.loc[:,"longitude"].to_numpy()
    alt = gps.loc[:,"altitude"].to_numpy()
    xyz = lla_to_ecef(lat,lon,alt)
    return xyz

def remove_repeat_lane(img, lines, e=0.5):
    """
    Remove lanes with near identical theta

    Parameters
    img: binary image
    lines: lines extracted using cv2.HoughLines
    e: difference in theta for each lane extracted

    Returns
    new_lines: trimmed lanes
    new_img: binary image with lanes added
    """
    new_lines = []
    new_img = img.copy()
    theta_arr = np.zeros(lines.shape[0])
    for i in range(len(lines)):
        rho = lines[i,0,0]
        theta = lines[i,0,1]
        diff = np.abs(theta_arr-theta)
        theta_arr[i] = theta
        if np.any(diff < e):
            continue
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0+1000*(-b))
        y1 = int(y0+1000*(a))
        x2 = int(x0-1000*(-b))
        y2 = int(y0-1000*(a))
        cv2.line(new_img,(x1,y1),(x2,y2),(0,0,255),2)
        line = [x1,y1,x2,y2]
        new_lines.append(line)
    return new_lines, new_img

def extract_lane(bi_file_loc, lane_len=50, e=0.5):
    """
    Extract lanes from binary images

    Parameters
    bi_img_loc: binary image file location
    line_len: minimum length of lane extracted
    e: difference in theta for each lane extracted

    Returns
    lanes: extracted lanes for each binary image
    """
    lanes = []
    bi_img_files = os.listdir(bi_file_loc)
    for i in tqdm(range(len(bi_img_files)),position=0):
        img_path = bi_file_loc+"/"+bi_img_files[i]
        img = cv2.imread(img_path)
        img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lines = cv2.HoughLines(img_grey,1,np.pi/180,lane_len)
        #print(lines.shape)
        if lines is None:
            lanes.append([])
            cv2.imwrite(par_dir+"/data/lane_bi_img_gen/"+bi_img_files[i], img)
        else:
            new_lines, new_img = remove_repeat_lane(img,lines,e)
            cv2.imwrite(par_dir+"/data/lane_bi_img_gen/"+bi_img_files[i], new_img)
            lanes.append(new_lines)
    return lanes
