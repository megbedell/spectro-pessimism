# This example generates 1d and 2d Phoenix spectra for MaroonX for a list of radial velocities.
# The output is saved in an HDF file. See

from generator import *

# spectrograph model
spectrograph = MaroonX()

# artificial m-dwarf spectrum for given stellar parameters
spectrum = Phoenix(t_eff=3600, log_g=5.,z=0, alpha=0., data_folder='../data')

# generate 1d and 2d spectra and save all of them in HDF file
generate_rv_series(spectrograph, spectrum, radial_velocities=[0., 100.],
                   photons_per_spectrum=int(1E7), bins_for_1d=spectrograph.calc_wavelength_vector()[0],
                   outfile='phoenix_1e7_photons.hdf')

