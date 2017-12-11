import math
import numpy as np


def lambda_to_x_y(wl):
    """
    Arbitrary function that translates wavelength to XY position
    To mimic a high-res spectrograph the dispersion is set to 1nm -> 1000px @600nm -> R~100.000 for a 3px sampling
    :param wl: wavelength [nm]
    :type wl: np.ndarray
    :return: XY position (to be more precise tx and ty translation parameter of the affine transformation matrix
    :rtype: np.ndarray
    """
    x = (wl - 600.)*500. + 50
    y = 10.
    return x, y


def compose_matrix(sx,sy,rot,shear,tx,ty):
    """
    generate affine transformation matrix with given parameters
    :param sx: scale in x (unit is ratio of input to output units)
    :type sx: float
    :param sy: scale in y (unit is ratio of input to output units)
    :type sy: float
    :param rot: slit rotation [rad]
    :type rot: float
    :param shear: slit shearing
    :type shear: float
    :param tx: translation in X
    :type tx: float
    :param ty: translation in Y
    :type ty: float
    :return: 3x3 affine transformation matrix
    :rtype: np.ndarray
    """
    return np.array([[sx * math.cos(rot), -sy * math.sin(rot + shear), 0],
                     [sx * math.sin(rot),  sy * math.cos(rot + shear), 0],
                     [tx, ty, 1]], dtype=np.float).T


def trace(wl, x_vec, y_vec, sx, sy, rot, shear):
    """
    Performs 'raytracing' for a given wavelength vector and XY input vectors
    :param wl: wavelength to be traced (right now should be between 600 and 600.2 nm)
    :type wl: np.ndarray
    :param x_vec: random X position within the slit
    :type x_vec: np.ndarray
    :param y_vec: ranndom Y position within the slit
    :type y_vec: np.ndarray
    :param sx: desired scaling in X direction
    :type sx: float
    :param sy: desired scalinig in Y direction
    :type sy: float
    :param rot: desired slit rotation [rad]
    :type rot: float
    :param shear: desired slit shearing
    :type shear: float
    :return: transformed XY positions for given input
    :rtype: np.ndarray
    """
    transformed = []
    xyz = np.vstack((x_vec, y_vec, np.ones_like(x_vec))).T
    for xy_vec, w in zip( xyz, wl):
        x,y = lambda_to_x_y(w)
        m = compose_matrix(sx,sy,rot,shear,x,y)
        transformed.append(np.dot(m, xy_vec.T))
    transformed = np.array(transformed)[:, :2].T
    return transformed
