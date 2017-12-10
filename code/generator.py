import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
import pyximport; pyximport.install()
import cython
from tracing import trace


def bin_2d(x, y, xmin=0, xmax=4096, ymin=0, ymax=4096):
    """
    Bin XY position into 2d grid
    :param x: X positions
    :type x: np.ndarray
    :param y: Y positions
    :type y: np.ndarray
    :param xmin: minimum X value of the grid
    :type xmin: int
    :param xmax: maximum X value of the grid
    :type xmax: int
    :param ymin: minimum Y value of the grid
    :type ymin: int
    :param ymax: maximum Y value of the grid
    :type ymax: int
    :return: binned XY positions
    :rtype: np.ndarray
    """
    ny = ymax - ymin
    nx = xmax - xmin
    weights = np.ones(x.size)

    # Basically, this is just doing what np.digitize does with one less copy
    xyi = np.vstack((y,x)).T
    xyi -= [ymin, xmin]
    xyi = np.floor(xyi, xyi).T

    grid = scipy.sparse.coo_matrix((weights, xyi), shape=(ny, nx)).toarray()

    return grid


def generate_slit_xy(N, width=1, height=1):
    """
    Generate uniform distributed XY position within a box
    :param N: number of random numbers
    :type N: int
    :param width: width of the box
    :type width: float
    :param height: height of the box
    :type height: float
    :return: random XY position
    :rtype: np.ndarray
    """
    x = np.random.random(N) * width
    y = np.random.random(N) * height
    return np.array([x,y])


def generate_round_slit_xy(N, diam=1):
    """
    Generate uniform distributed XY position within a disk
    :param N: number of random numbers
    :type N: int
    :param diam: diameter of the disk
    :type diam: float
    :return: random XY position
    :rtype: np.ndarray
    """
    r = np.sqrt(np.random.random(N)) * diam/2.
    phi = np.random.random(N) * np.pi * 2.
    return np.array([r*np.cos(phi), r * np.sin(phi)])


def generate_gaussian_distortion(N, sig_x=0.5, sig_y=0.5):
    """
    Generate gaussian distributed random numbers to emulate gaussian PSF
    :param N: number of random numbers
    :type N: int
    :param sig_x: sigma in x direction
    :type sig_x: float
    :param sig_y: sigma in y direction
    :type sig_y: float
    :return: gaussian distributed XY distortions
    :rtype: np.ndarray
    """
    return np.random.multivariate_normal([0,0],[[sig_x, 0],[0,sig_y]], size=(N,)).T


def apply_rv(wl, rv=100):
    """
    Apply radial velocity shift to wavelength vector
    :param wl: wavelength
    :type wl: np.ndarray
    :param rv: radial velocity shift [m/s]
    :type rv: float
    :return: shifted wavelength vector
    :rtype: np.ndarray
    """
    return wl * np.sqrt((1.+rv/3E8)/(1.-rv/3E8))


if __name__ == "__main__":
    import time
    t1 = time.time()

    # total number of photons
    N = 1000000

    # generate random XY position within slit (slit is 3 x 10 pixels in size)
    xy = generate_slit_xy(N, 3,10)

    # generate spectrum (i.e. list of wavelength following a certain distribution)
    # TODO: implement inverse transform sampling for e.g. stellar spectra

    # wl = np.random.uniform(600, 600.2, N)  # "flat"
    wl = np.random.choice(np.linspace(600, 600.2, 10), N) # etalon lines
    # for applying an RV shift to the wavelength uncomment the next line
    # wl = apply_rv(wl, 1000)

    # do tracing... Note that sx and sy are 1 here, therefore the slit coordinates more or less directly translate in pixel coordinates
    # slit rotation is 8 deg
    transformed = trace(wl, xy[0], xy[1], 1., 1., 8. / 180. * 3.14159265359, 0.)

    # apply PSF as gaussian distortion
    noise = generate_gaussian_distortion(N, 0.5, 0.5)
    transformed += noise
    # bin to pixels
    img = bin_2d(*transformed, ymax=30, xmax=200)

    t2 = time.time()

    print(t2-t1)
    plt.figure()
    plt.imshow(img, origin='lower')
    plt.show()
