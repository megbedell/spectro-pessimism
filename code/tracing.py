import math
import numpy as np


def lambda_to_x_y(wl, offset_x, offset_y, disp_x, disp_y, wl_start):
    """ Function that translates wavelength to XY position

    To mimic a high-res spectrograph the dispersion is set to 1nm -> 500px @600nm -> R~100.000 for a 3px sampling
    Args:
        wl (np.ndarray): wavelength [nm]
        offset_x (float): offset in X-direction
        offset_y (float): offset in Y-direction
        disp_x (float): dispersion in x-direction [px/nm]
        disp_y (float): dispersion in y-direction [px/nm]
        wl_start (float): to be substracted from wl before multiplying with dispersion factors

    Returns:
        tuple: x and y offsets for affine transformation
    """
    """
    
    
    :param wl: wavelength [nm]
    :type wl: np.ndarray
    :return: XY position (to be more precise tx and ty translation parameter of the affine transformation matrix
    :rtype: np.ndarray
    """
    x = (wl - wl_start)*disp_x + offset_x
    y = (wl - wl_start)*disp_y + offset_y
    return x, y


def compose_matrix(sx,sy,rot,shear,tx,ty):
    """ generate affine transformation matrix with given parameters

    Args:
        sx (float): scale in x (unit is ratio of input to output units)
        sy (float): scale in y (unit is ratio of input to output units)
        rot (float): slit rotation [rad]
        shear (float):  slit shear
        tx (float): translation in X (lower left corner)
        ty (float): translation in Y (lower left corner)

    Returns:
        np.ndarray: 3x3 affine transformation matrix
    """
    return np.array([[sx * math.cos(rot), -sy * math.sin(rot + shear), 0],
                     [sx * math.sin(rot),  sy * math.cos(rot + shear), 0],
                     [tx, ty, 1]], dtype=np.float).T


def trace(wl, x_vec, y_vec, sx, sy, rot, shear, offset_x, offset_y, disp_x, disp_y, wl_start):
    """ Performs 'raytracing' for a given wavelength vector and XY input vectors

    Args:
        wl (np.ndarray): wavelength to be traced
        x_vec (np.ndarray): random X positions within the slit
        y_vec (np.ndarray): random Y positions within the slit
        sx (float): desired scaling in X direction
        sy (float):  desired scalinig in Y direction
        rot (float): desired slit rotation [rad]
        shear (float): desired slit shear
        offset_x (float): tx of affine matrix
        offset_y (float): ty of affine matrix
        disp_x (float): dispersion in x-direction [px/nm]
        disp_y (float): dispersion in y-direction [px/nm]
        wl_start (float): to be substracted from wl before multiplying with dispersion factors

    Returns:
        np.ndarray: transformed XY positions for given input
    """
    transformed = []
    xyz = np.vstack((x_vec, y_vec, np.ones_like(x_vec))).T
    for xy_vec, w in zip( xyz, wl):
        x,y = lambda_to_x_y(w, offset_x, offset_y, disp_x, disp_y, wl_start)
        m = compose_matrix(sx,sy,rot,shear,x,y)
        transformed.append(np.dot(m, xy_vec.T))
    transformed = np.array(transformed)[:, :2].T
    return transformed
