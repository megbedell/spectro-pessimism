# a simple example how to read in the HDF files with the simulated spectra...

import h5py
import matplotlib.pyplot as plt

with h5py.File('test.hdf') as f:
    spec0 = f['1d_spectra/0.0_spectrum'].value
    wl0 = f['1d_spectra/0.0_wavelength'].value
    spec100 = f['1d_spectra/100.0_spectrum'].value
    wl100 = f['1d_spectra/100.0_wavelength'].value

    spec_2d_0 = f['2d_spectra/0.0'].value
    spec_2d_100 = f['2d_spectra/100.0'].value

plt.figure()
plt.imshow(spec_2d_0 - spec_2d_100)

plt.figure()
plt.plot(spec0)
plt.plot(spec100)

plt.figure()
plt.plot(spec0-spec100)
plt.show()
