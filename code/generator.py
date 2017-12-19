import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
import pyximport; pyximport.install()
from tracing import trace
from scipy.sparse import coo_matrix, lil_matrix
import h5sparse
from scipy.stats import rv_continuous
from scipy.stats import wrapcauchy
import os
import urllib
from astropy.io import fits
import logging
import h5py


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
    return np.random.multivariate_normal([0, 0], [[sig_x, 0], [0, sig_y]], size=(N,)).T


def apply_rv(wl_vector, rv=100):
    """ Apply radial velocity shift to wavelength vector.

    Args:
        wl_vector (np.ndarray): original wavelength
        rv (float): radial velocity shift in [m/s]

    Returns:
        np.ndarray: shifted wavelength
    """
    return wl_vector * np.sqrt((1. + rv / 3E8) / (1. - rv / 3E8))


class Spectrograph:
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
        raise NotImplementedError()

    def generate_psf_distortion(self, N):
        raise NotImplementedError()

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
            A += coo_matrix((data / np.sum(data),(np.arange(6000), np.repeat(i, 6000))), shape=(6000, wl_steps))
        if path is not None:
            with h5sparse.File(path, "w") as h5f:
                h5f.create_dataset("A", data=A.tocsc(), format='csc', compression='gzip', compression_opts=6)
        return A


class MaroonX(Spectrograph):
    """ MaroonX-like spectrograph.

    This spectrograph has a rectangular slit/fiber and a PSF < slit.
    """
    def __init__(self):
        super().__init__(name='MaroonX')
        self.psf_sig_x = 0.5
        self.psf_sig_y = 0.5

    def generate_slit(self, N):
        return generate_slit_xy(N, 3, 10)

    def generate_psf_distortion(self, N):
        return generate_gaussian_distortion(N, self.psf_sig_x, self.psf_sig_y)


class HARPS(Spectrograph):
    """ HARPS-like spectrograph.

    This spectrograph has a circular slit/fiber and a PSF < slit.

    """
    def __init__(self):
        super().__init__(name='HARPS')
        self.psf_sig_x = 0.5
        self.psf_sig_y = 0.5

    def generate_slit(self, N):
        return generate_round_slit_xy(N, 3.)

    def generate_psf_distortion(self, N):
        return generate_gaussian_distortion(N, self.psf_sig_x, self.psf_sig_y)


class iLocater(Spectrograph):
    """ iLocater-like spectrograph.

    This spectrograph is a SM spectrograph.
    The slit has basically no extent. Therefore, all slit related attributes such as rotation, scaling in X- and Y-
    direction have no effect.
    All 'spread' comes from the PSF distortion.
    """
    def __init__(self):
        super().__init__(name='iLocator')
        self.psf_sig_x = 0.5
        self.psf_sig_y = 0.5

    def generate_psf_distortion(self, N):
        return generate_gaussian_distortion(N, self.psf_sig_x, self.psf_sig_y)

    def generate_slit(self, N):
        return np.zeros(N), np.zeros(N)


class Spectrum:
    """ A spectral source.

    This class should be subclassed to implement different spectral sources.

    Attributes:
        wavelength (np.ndarray): randomly drawn wavelength constituting the spectrum
        min_wl (float): lower wavelength limit [nm] (for normalization purposes)
        max_wl (float): upper wavelength limit [nm] (for normalization purposes)

    """
    def __init__(self, min_wl=600., max_wl=600.4, name=''):
        self.name = name
        self.wavelength = None
        self.min_wl = min_wl
        self.max_wl = max_wl

    def draw_wavelength(self, N):
        """
        Overwrite this function in child class !
        Args:
            N (int): number of wavelength to randomly draw

        Returns:

        """
        raise NotImplementedError()

    def apply_rv(self, rv):
        """ Apply radial velocity shift.

        Applies an RV shift to the formerly drawn wavelength.
        Args:
            rv (float): radial velocity shift [m/s]

        Returns:
            np.ndarray: shifted wavelength

        """
        self.wavelength = apply_rv(self.wavelength, rv)
        return self.wavelength

    def bin_to_wavelength(self, wl_vector):
        """ Bins random wavelength into wavelength vector.

        Args:
            wl_vector (np.ndarray): wavelength bin edges

        Returns:
            see np.histogram for details
        """
        return np.histogram(self.wavelength, wl_vector)


class Flat(Spectrum):
    def __init__(self, **kwargs):
        super().__init__(**kwargs, name='Flat')

    def draw_wavelength(self, N):
        self.wavelength = (self.max_wl-self.min_wl)* np.random.random(N) + self.min_wl
        return self.wavelength


class Etalon(Spectrum):
    def __init__(self, d=10., n=1., theta=0., **kwargs):
        super().__init__(**kwargs, name='Etalon')
        self.d = d
        self.n = n
        self.theta = theta
        self.min_m = np.floor(2E6 * d * np.cos(theta) / self.max_wl)
        self.max_m = np.ceil(2E6 * d * np.cos(theta) / self.min_wl)

    @staticmethod
    def peak_wavelength_etalon(m, d=10., n=1., theta=0.):
        return 2E6 * d * n * np.cos(theta) / m

    def draw_wavelength(self, N):
        self.wavelength = np.random.choice(self.peak_wavelength_etalon(np.arange(self.min_m, self.max_m),
                                                                       self.d, self.n, self.theta), N)
        return self.wavelength


class Phoenix(Spectrum):
    """
    Phoenix M-dwarf spectra.

             .-'  |
            / M <\|
           /dwarf\'
           |_.- o-o
           / C  -._)\
          /',        |
         |   `-,_,__,'
         (,,)====[_]=|
           '.   ____/
            | -|-|_
            |____)_)

    This class provides a convenient handling of PHOENIX M-dwarf spectra.
    For a given set of effective Temperature, log g, metalicity and alpha, it downloads the spectrum from PHOENIX ftp
    server.

    TODO:
    * recalculate spectral flux of original fits files to photons !!!!!
    """
    def __init__(self, t_eff=3600, log_g=5., z=0, alpha=0., **kwargs):
        self.t_eff = t_eff
        self.log_g = log_g
        self.z = z
        self.alpha = alpha
        super().__init__(**kwargs, name='phoenix')
        valid_T = [*list(range(2300, 7000, 100)), *list((range(7000, 12200, 200)))]
        valid_g = [*list(np.arange(0, 6, 0.5))]
        valid_z = [*list(np.arange(-4, -2, 1)), *list(np.arange(-2., 1.5, 0.5))]
        valid_a = [*list(np.arange(-.2, 1.4, 0.2))]

        if t_eff in valid_T and log_g in valid_g and z in valid_z and alpha in valid_a:
            if not os.path.exists("data/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits"):
                print("Download Phoenix wavelength file...")
                url = 'ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits'
                with urllib.request.urlopen(url) as response, open('data/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits', 'wb') as out_file:
                    data = response.read()
                    out_file.write(data)

            self.wl_data = fits.getdata(str('data/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits')) / 10.

            baseurl = 'ftp://phoenix.astro.physik.uni-goettingen.de/' \
                      'HiResFITS/PHOENIX-ACES-AGSS-COND-2011/' \
                      'Z{0:{1}2.1f}{2}{3}/'.format(z,
                                                   '+' if z>0 else '-',
                                                   '' if alpha==0 else '.Alpha=',
                                                   '' if alpha==0 else '{:+2.2f}'.format(alpha))
            url = baseurl+'lte{0:05}-{1:2.2f}{2:{3}2.1f}{4}{5}.' \
                          'PHOENIX-ACES-AGSS-COND-2011-HiRes.fits'.format(t_eff,
                                                                          log_g,
                                                                          z,
                                                                          '+' if z>0 else '-',
                                                                          '' if alpha==0 else '.Alpha=',
                                                                          '' if alpha==0 else '{:+2.2f}'.format(alpha))

            filename = 'data/'+ url.split("/")[-1]

            if not os.path.exists(filename):
                print("Download Phoenix spectrum...")
                with urllib.request.urlopen(url) as response, open(filename, 'wb') as out_file:
                    data = response.read()
                    out_file.write(data)

            self.spectrum_data = fits.getdata(filename)

            lowWl = np.where(self.wl_data > self.min_wl)[0][0]
            highWl = np.where(self.wl_data > self.max_wl)[0][0]
            self.spectrum_data = self.spectrum_data[lowWl:highWl]
            self.wl_data = self.wl_data[lowWl:highWl]
        else:
            print("Valid values are:")
            print("T: ", *valid_T )
            print("log g: ", *valid_g)
            print("Z: ", *valid_z)
            print("alpha: ", *valid_a)
            raise ValueError("Invalid parameter for M-dwarf spectrum ")

    def draw_wavelength(self, N):
        self.wavelength = np.random.choice(self.wl_data, p=self.spectrum_data/np.sum(self.spectrum_data), size=N)
        self.wavelength += (np.random.random(N) * (self.wl_data[1]-self.wl_data[0]))
        return self.wavelength


def generate_rv_series(spectrograph, spectrum, radial_velocities, photons_per_spectrum=int(1E7), outfile='test.hdf', bins_for_1d=None):
    """
    Function to generate a series of spectra with different radial velocities.

    Args:
        spectrograph (Spectrograph): spectrograph used for simulation
        spectrum (Spectrum): spectrum used for simulation
        radial_velocities (iterable): radial velocities, one spectrum per entry will be generated
        photons_per_spectrum (int): number of photons per spectrum
        outfile (string): path to HDF file that contains RV series
        bins_for_1d (np.ndarray): OPTIONAL, if given, spectrum.wavelength will be binned directly onto that wavelength vector and saved in the file.

    Returns:
        None

    """
    with h5py.File(outfile, "w") as h5f:
        h5f.create_group('2d_spectra')
        if bins_for_1d is not None:
            h5f.create_group('1d_spectra')
        spectrograph_group = h5f.create_group('spectrograph')

        # save all spectrograph attributes (at least all string, int and float values...
        for attr, value in spectrograph.__dict__.items():
            if isinstance(value, float) or isinstance(value, str) or isinstance(value, int):
                spectrograph_group.attrs[attr] = value

        for rv in radial_velocities:
            logging.debug('Simulating '+str(rv) +' m/s spectrum...')
            spectrum.draw_wavelength(photons_per_spectrum)
            img = spectrograph.generate_2d_spectrum(spectrum.apply_rv(rv))
            dt = h5f['2d_spectra'].create_dataset(str(rv), data=img)

            if bins_for_1d is not None:
                spec_1d, bins = spectrum.bin_to_wavelength(bins_for_1d)
                h5f['1d_spectra'].create_dataset(str(rv)+'_wavelength', data=bins)
                h5f['1d_spectra'].create_dataset(str(rv) + '_spectrum', data=spec_1d)

            # save all spectrum attributes
            for attr, value in spectrum.__dict__.items():
                if isinstance(value, float) or isinstance(value, str) or isinstance(value, int):
                    dt.attrs[attr] = value


if __name__ == "__main__":
    import time
    t1 = time.time()

    spectrum = Etalon()
    spectrograph = MaroonX()
    generate_rv_series(spectrograph, spectrum, [0., 100., 50.], bins_for_1d=np.linspace(600, 600.4, 1000))
