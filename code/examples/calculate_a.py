# this example calculates the A matrix for the MaroonX on a fine wavelength grid (N times finer than the 'pixel' grid)
from generator import *

# oversampling for A (compared to one wavelength step per pixel - keep in mind that one resolution element is about 3px)
N = 5

spectrograph = MaroonX()

# wavelength to compute A on
wl_vector = spectrograph.calc_wavelength_vector()[1]
wl_vector = np.linspace(min(wl_vector), max(wl_vector), N*len(wl_vector))

# calculate A and save it. THIS TAKES A LONG TIME! depending on photons_per_step obviously...
spectrograph.calc_A(wl_vector, photons_per_step=int(1E8), path="maroonx.hdf")
