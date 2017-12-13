import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
import pyximport; pyximport.install()
from tracing import trace
from scipy.sparse import coo_matrix, lil_matrix
import h5sparse


def bin_2d(x, y, xmin=0, xmax=4096, ymin=0, ymax=4096):
    """  Bin XY position into 2d grid and throw away data outside the limits.

    Args:
        x (np.ndarray): X positions
        y (np.ndarray): Y positions
        xmin (int): minimum X value of the grid
        xmax (int): maximum X value of the grid
        ymin (int): minimum Y value of the grid
        ymax (int): maximum Y value of the grid

    Returns:
        np.ndarray: binned XY positions
    """
    valid_idx = np.logical_and.reduce((x>=xmin, y>=ymin, y<ymax, x<xmax))
    x = x[valid_idx]
    y = y[valid_idx]
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
    """  Generate uniform distributed XY position within a box

    Args:
        N (int):  number of random numbers
        width (float): width of the box
        height (float): height of the box

    Returns:
        np.ndarray: random XY position
    """
    x = np.random.random(N) * width
    y = np.random.random(N) * height
    return np.array([x,y])


def generate_round_slit_xy(N, diam=1.):
    """ Generate uniform distributed XY position within a disk

    Args:
        N (int): number of random numbers
        diam (float): diameter of the disk/fiber

    Returns:
        np.ndarray: random XY positions
    """
    r = np.sqrt(np.random.random(N)) * diam/2.
    phi = np.random.random(N) * np.pi * 2.
    return np.array([r*np.cos(phi), r * np.sin(phi)])


def generate_gaussian_distortion(N, sig_x=0.5, sig_y=0.5):
    """ Generate gaussian distributed random numbers to emulate gaussian PSF

    Args:
        N (int): number of random numbers
        sig_x (float): sigma in x direction
        sig_y (float): sigma in y direction

    Returns:
        np.ndarray gaussian distributed XY distortions
    """
    return np.random.multivariate_normal([0,0],[[sig_x, 0],[0,sig_y]], size=(N,)).T


def apply_rv(wl, rv=100):
    """ Apply radial velocity shift to wavelength vector.

    Args:
        wl (np.ndarray): original wavelength
        rv (float): radial velocity shift in [m/s]

    Returns:
        np.ndarray: shifted wavelength
    """
    return wl * np.sqrt((1.+rv/3E8)/(1.-rv/3E8))


class Spectrograph():
    """ An echelle spectrograph.

    This class should be used to subclass actual spectrograph models.

    Overwrite generate_slit() and generate_psf_distortion() to alter their slit shape and PSF.

    To further adjust the 2D spectrum generation adjust the following attributes...

    Attributes:
        sx (float): slit scaling in x-direction
        sy (float): slit scaling in y-direction
        rot (float): slit rotation in radians
        shear (float): slit shear

        offset_x (float): translation offset in X-direction (tx of affine matrix)
        offset_y (float): translation offset in Y-direction (ty of affine matrix)

        disp_x (float): dispersion in [px/nm] in X-direciton
        disp_y (float): dispersion in [px/nm] in Y-direciton

        wl_center (float): value is substracted from any wavelength_vector before multiplying disp_x (see tracing for
        details)

    """
    def __init__(self, name=''):
        self.name = name

        #: float: slit scaling in x-direction
        self.sx = 1.
        #: float: slit scaling in y-direction
        self.sy = 1.
        self.rot = 0. / 180. * 3.14159265359
        self.shear = 0.
        self.offset_x = 0.
        self.offset_y = 10.
        self.disp_x = 500.
        self.disp_y = 10.
        self.wl_center = 600.

        self.img_width = 200
        self.img_height = 30

    def generate_slit(self, N):
        return np.zeros(N), np.zeros(N)

    def generate_psf_distortion(self, N):
        return np.zeros(N), np.zeros(N)

    def generate_2d_spectrum(self, wl_vector):
        N = len(wl_vector)
        xy = self.generate_slit(N)
        transformed = trace(wl_vector, xy[0], xy[1],self.sx, self.sy, self.rot, self.shear, self.offset_x, self.offset_y,
                            self.disp_x, self.disp_y, self.wl_center)

        transformed += self.generate_psf_distortion(N)
        img = bin_2d(*transformed, ymax=self.img_height, xmax=self.img_width)
        return img

    def calc_A(self, wl_vector, photons_per_step=1E6, path=None):
        """
        Calculates the monochromatic response for each wavelength step.
        Calculates calibration matrix.
        To save matrix, specify path to an HDF5 file.

        Args:
            wl_vector (np.ndarray): wavelength vector that is used to calculate calibration matrix
            photons_per_step (int): number of photons per wavelength step
            path (str): string to hdf file where to save HDF file

        Returns:
            scipy.sparse.coo_matrix: calibration matrix

        """
        N = int(photons_per_step)
        wl_steps = len(wl_vector)
        A = coo_matrix((self.img_width*self.img_height, wl_steps), dtype=np.float)
        for i, w in enumerate(wl_vector):
            data = self.generate_2d_spectrum(np.repeat(w, N)).flatten()
            A += coo_matrix((data / np.sum(data),(np.arange(6000), np.repeat(i,6000))), shape=(6000,wl_steps))
        if path is not None:
            with h5sparse.File(path, "w") as h5f:
                h5f.create_dataset("A", data=A.tocsc(), format='csc', compression='gzip', compression_opts=6)
        return A


class MaroonX(Spectrograph):
    """ MaroonX-like spectrograph.

    This spectrograph has a rectangular slit/fiber and a PSF < slit.
    """
    def generate_slit(self, N):
        return generate_slit_xy(N, 3, 10)

    def generate_psf_distortion(self, N):
        return generate_gaussian_distortion(N, 0.5, 0.5)


class HARPS(Spectrograph):
    """ HARPS-like spectrograph.

    This spectrograph has a circular slit/fiber and a PSF < slit.

    """
    def generate_slit(self, N):
        return generate_round_slit_xy(N, 3.)

    def generate_psf_distortion(self, N):
        return generate_gaussian_distortion(N, 0.5, 0.5)


class iLocater(Spectrograph):
    """ iLocater-like spectrograph.

    This spectrograph is a SM spectrograph.
    The slit has basically no extent. Therefore, all slit related attributes such as rotation, scaling in X- and Y-
    direction have no effect.
    All 'spread' comes from the PSF distortion.
    """
    def generate_psf_distortion(self, N):
        return generate_gaussian_distortion(N, 1., 1.)


if __name__ == "__main__":
    import time
    t1 = time.time()

    # total number of photons
    N = int(1E7)

    spec = MaroonX()
    img = spec.generate_2d_spectrum(np.random.choice(np.linspace(600, 600.4, 25), N))
    t2 = time.time()
    print(t2 - t1)
    plt.figure()
    plt.imshow(img, origin='lower')


    # now lets tilt the slit !
    spec.rot = 8./180. * np.pi
    img = spec.generate_2d_spectrum(np.random.choice(np.linspace(600, 600.4, 25), N))

    plt.figure()
    plt.imshow(img, origin='lower')

    # lets calculate mighty A on a fine wavelength grid (that takes a while...)
    A = spec.calc_A(np.linspace(600, 600.4, 1000), int(1E5), 'maroonx.hdf')

    plt.figure()
    plt.imshow(A.todense(), origin='lower')
    plt.show()
