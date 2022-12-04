import numpy as np
import os
import pandas as pd
import cv2
from tqdm import tqdm
from datetime import datetime
from gpx_converter import Converter
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from gpx_interpolate import gpx_interpolate

# Get directory
current_dir = os.getcwd()
par_dir = os.path.dirname(current_dir)

# Binary image dimensions
W = 512
H = 256

# Lane width (in meters)
LW = 3.05

# Local origin for Ford AV dataset
ORIGIN_LAT, ORIGIN_LON, ORIGIN_ALT = 42.294319, -83.223275, 146.0


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

def ecef2enu(x, y, z, lat_ref=ORIGIN_LAT, lon_ref=ORIGIN_LON, alt_ref=ORIGIN_ALT):
    """ECEF to ENU
    
    Convert ECEF (m) coordinates to ENU (m) about reference latitude (N°) and longitude (E°).

    Parameters
    ----------
    x : float
        ECEF x-coordinate
    y : float
        ECEF y-coordinate
    z : float
        ECEF z-coordinate
    lat_ref : float
        Reference latitude (N°) 
    lon_ref : float
        Reference longitude (E°) 
    alt_ref : float
        Reference altitude (m)
    
    Returns
    -------
    x : float
        ENU x-coordinate
    y : float
        ENU y-coordinate
    z : float
        ENU z-coordinate

    """
    ecef_ref = lla_to_ecef(lat_ref, lon_ref, alt_ref)
    lat_ref = np.deg2rad(lat_ref)
    lon_ref = np.deg2rad(lon_ref + 360)
    C = np.zeros((3,3))
    C[0,0] = -np.sin(lat_ref)*np.cos(lon_ref)
    C[0,1] = -np.sin(lat_ref)*np.sin(lon_ref)
    C[0,2] = np.cos(lat_ref)

    C[1,0] = -np.sin(lon_ref)
    C[1,1] = np.cos(lon_ref)
    C[1,2] = 0

    C[2,0] = np.cos(lat_ref)*np.cos(lon_ref)
    C[2,1] = np.cos(lat_ref)*np.sin(lon_ref)
    C[2,2] = np.sin(lat_ref)

    x, y, z = np.dot(C, np.array([x, y, z]) - ecef_ref)

    return x, y, z

def lla2enu(lat, lon, alt):
    return ecef2enu(*lla_to_ecef(lat, lon, alt), ORIGIN_LAT, ORIGIN_LON, ORIGIN_ALT)

def quat2euler(q):
    """Convert quaternion to Euler angles

    Parameters
    ----------
    q : array-like (4)
        Quaternion in form (qx, qy, qz, qw)
    
    Returns
    -------
    array-like (3)
        x,y,z Euler angles in radians (extrinsic)

    """
    r = R.from_quat(q)
    return r.as_euler('XYZ')
    
def get_gps_data(file_loc):
    """
    Returns GPS data from file

    Parameters
    file_loc: gps data file location

    Returns
    xyz: np array of locations in cartesian coordinate system
    alt: np array of altitude data
    """
    gps = pd.read_csv(file_loc)
    lat = gps.loc[:,"latitude"].to_numpy()
    lon = gps.loc[:,"longitude"].to_numpy()
    alt = gps.loc[:,"altitude"].to_numpy()
    xyz = lla_to_ecef(lat,lon,alt)
    return xyz, alt

def remove_repeat_lane(img, lines, e=0.5):
    """
    Remove lanes with near identical theta

    Parameters
    img: binary image
    lines: lines extracted using cv2.HoughLines
    e: difference in theta for each lane extracted

    Returns
    new_lines: trimmed lanes, single line: [rho, theta, x1, y1, x2, y2]
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
        line = [rho,theta,x1,y1,x2,y2]
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
        if lines is None:
            lanes.append([])
            cv2.imwrite(par_dir+"/data/lane_bi_img_gen/"+bi_img_files[i], img)
        else:
            new_lines, new_img = remove_repeat_lane(img,lines,e)
            cv2.imwrite(par_dir+"/data/lane_bi_img_gen/"+bi_img_files[i], new_img)
            lanes.append(new_lines)
    return lanes

def get_google_data(file_loc, alt):
    """
    Returns GPS data from Google Maps data

    Parameters
    file_loc: Google gpx data file location
    alt: altiude data from GPS data

    Returns
    xyz: np array of locations in cartesian coordinate system
    """
    data = Converter(input_file=file_loc).gpx_to_dictionary(latitude_key='latitude', longitude_key='longitude')
    lat = np.array(data["latitude"])
    lon = np.array(data["longitude"])
    target_size = len(alt)
    gpx_data = {"lat":lat,
                "lon":lon,
                "ele":None,
                "tstamp":None,
                "tzinfo":None}
    xy = gpx_interpolate(gpx_data,num=target_size)
    new_lat = xy['lat']
    new_lon = xy['lon']
    xyz = lla_to_ecef(new_lat,new_lon,alt)
    return xyz

def find_horiz_intersect(x1, y1, x2, y2, y):
    """
    Find the intersection of given line with a horizontal line

    Parameters
    x1, y1, x2, y2: points specifying a line
    y: horizontal line

    Returns
    x, y: intersection point coordinates
    k, b: line parameters
    """
    k = (y2-y1)/(x2-x1)
    b = y1-k*x1
    x = (y-b)/k
    return x, y, k ,b

def find_line_intersection(k1, b1, k2, b2):
    """
    Find intersection between 2 lines

    Parameters:
    k1, b1: parameters for line 1
    k2, b2: parameters for line 2

    Returns
    x, y: intersection coordinates
    """
    x = (b2-b1)/(k1-k2)
    y = k1*x+b1
    return x, y

def angle_bisector_dist(int_x, int_y, left_x, left_y, right_x, right_y):
    """
    Find the length split of angle bisector for 2 intersecting lines

    Parameters
    int_x, int_y: intersection point coordinates
    left_x, left_y: left lane intersection with bottom of image
    right_x, right_y: right lane intersection with bottom of image

    Returns
    a: distance from left side of the lane to the center
    b: distance from right side of the lane to the center

    Notes
    Using Angle Bisector Theorem a/b = x/y
    """
    x = np.linalg.norm(np.array([int_x,int_y])-np.array([left_x,left_y]))
    y = np.linalg.norm(np.array([int_x,int_y])-np.array([right_x,right_y]))
    tot_len = np.linalg.norm(np.array([left_x,left_y])-np.array([right_x,right_y]))
    a = tot_len/(x+y)*x
    b = tot_len-a
    return a, b

def find_dist_from_lane_sides(lane1, lane2):
    """
    Find distance from left and right side of the lane to its center

    Parameters
    lane1, lane2: lane parameters

    Returns
    center_lane_pos: center lane position
    """
    lane1_params = find_horiz_intersect(lane1[2],lane1[3],lane1[4],lane1[5],H)
    lane2_params = find_horiz_intersect(lane2[2],lane2[3],lane2[4],lane2[5],H)
    lane_intersection = find_line_intersection(lane1_params[2],lane1_params[3],lane2_params[2],lane2_params[3])
    if lane1_params[0] > lane2_params[0]:
        a, _ = angle_bisector_dist(lane_intersection[0],lane_intersection[1],lane2_params[0],lane2_params[1],lane1_params[0],lane1_params[1])
        center_lane_pos = lane2_params[0]+a
    else:
        a, _ = angle_bisector_dist(lane_intersection[0],lane_intersection[1],lane1_params[0],lane1_params[1],lane2_params[0],lane2_params[1])
        center_lane_pos = lane1_params[0]+a
    return center_lane_pos

def gen_relative_pos(lanes):
    """
    Generate relative positions of car with respect to lanes

    Parameters
    lanes: generated lanes

    Returns
    relative_dist: distance relative to the center lane

    Notes
    relative_dist > 0: car to the right of center lane
    relative_dist <= 0: car to the left of the center lane
    None: no center lane detected
    """
    camera_loc = W//2
    theta_0 = 0
    theta_90 = np.pi/2
    relative_dist = []
    for lane in lanes:
        if len(lane) <= 1:
            relative_dist.append(None)
            continue
        lane = np.array(lane)
        theta = lane[:,1]
        diff_to_0 = np.abs(theta-theta_0)
        diff_to_90 = np.abs(theta-theta_90)
        keep_idx = np.union1d(np.where(diff_to_90 > 0.1),np.where(diff_to_0 > 0.1))
        if len(keep_idx) != len(lane):
            lane = lane[keep_idx]
        if len(lane) <= 1:
            relative_dist.append(None)
            continue
        if len(lane) == 2:
            lane1 = lane[0]
            lane2 = lane[1]
            center_lane = find_dist_from_lane_sides(lane1,lane2)
            deviation = center_lane-camera_loc
        else:
            idx_sorted = np.argsort(theta)
            lane = lane[idx_sorted]
            potential_deviations = []
            for i in range(len(lane)):
                lane1 = lane[i]
                for j in range(len(lane)):
                    if i == j:
                        continue
                    lane2 = lane[j]
                    center_lane = find_dist_from_lane_sides(lane1,lane2)
                    potential_deviations.append(center_lane)
            potential_deviations = np.array(potential_deviations)
            min_diff_idx = np.argmin(np.abs(potential_deviations-camera_loc))
            deviation = potential_deviations[min_diff_idx]-camera_loc
        actual_deviation = deviation/W*LW
        relative_dist.append(actual_deviation)
    return relative_dist

def find_nearest_pos(road_x: np.ndarray, road_y: np.ndarray, 
                     gps_x:  np.ndarray, gps_y:  np.ndarray):
    """ 
    For each point in (gps_x, gps_y), find the best match in all of (road_x, road_y).
    Return a pd.DataFrame with the best match for each point.
    """
    road_x, road_y, gps_x, gps_y = np.array(road_x), np.array(road_y), np.array(gps_x), np.array(gps_y)
    res = np.zeros((gps_x.shape[0], 2))
    for i, (x0, y0) in enumerate(zip(gps_x, gps_y)):
        if np.mod(i+1, gps_x.shape[0] // 10) == 0:
            print(f'\033[A\33[2\r[{i+1}/{gps_x.shape[0]}]', end='')

        delta_x = road_x-x0
        delta_y = road_y-y0
        dist2 = delta_x*delta_x + delta_y*delta_y
        idx_min = np.argmin(dist2)
        res[i, :] = road_x[idx_min], road_y[idx_min]
    res = res.T
    res = pd.DataFrame({'road_x':res[0], 'road_y':res[1]})
    return res

def find_nearest_time(gps_t: np.ndarray, cam_t: np.ndarray):
    """ 
    For each time in cam_t, find the best match in all of gps_t.
    Return a pd.DataFrame with the ebst match for each point.
    """
    gps_t, cam_t = np.array(gps_t), np.array(cam_t)
    res = np.zeros_like(cam_t)
    for i, t0 in enumerate(cam_t):
        delta_t = np.abs(gps_t - t0)
        idx_min = np.argmin((delta_t))
        res[i] = gps_t[idx_min]
    res = pd.DataFrame({'time':res})
    return res

def get_datetime():
    now = datetime.now()
    y = str(now.year)[2:]
    m = str(now.month).rjust(2,'0')
    d = str(now.day).rjust(2,'0')
    h = str(now.hour).rjust(2,'0')
    min = str(now.minute).rjust(2,'0')
    s = str(now.second).rjust(2,'0')
    return y+m+d+'-h'+h+min+s