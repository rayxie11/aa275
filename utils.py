import numpy as np

def lla_to_ecef(lat, lon, alt):
    """LLA to ECEF conversion.

    Parameters
    ----------
    lat : float
        Latitude in degrees (N°).
    lon : float
        Longitude in degrees (E°).
    alt : float
        Altitude in meters.

    Returns
    -------
    ecef : np.ndarray
        ECEF coordinates corresponding to input LLA.

    Notes
    -----
    Based on code from https://github.com/Stanford-NavLab/gnss_lib_py.

    """
    A = 6378137  # Semi-major axis (radius) of the Earth [m].
    E1SQ = 6.69437999014 * 0.001  # First esscentricity squared of Earth (not orbit).
    lat = np.deg2rad(lat); lon = np.deg2rad(lon)
    xi = np.sqrt(1 - E1SQ * np.sin(lat)**2)
    x = (A / xi + alt) * np.cos(lat) * np.cos(lon)
    y = (A / xi + alt) * np.cos(lat) * np.sin(lon)
    z = (A / xi * (1 - E1SQ) + alt) * np.sin(lat)
    ecef = np.array([x, y, z]).T
    return ecef