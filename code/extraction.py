import h5py
import h5sparse
import matplotlib.pyplot as plt
import numpy as np

# Load spectrum
with h5py.File('example_data/etalon.hdf') as h5f:
    true_data = h5f['1d_spectra/0.0_spectrum'].value
    true_wavelength = h5f['1d_spectra/0.0_wavelength'].value
    # since truewl are the edges of the wavelength bins, the centers are calculated here:
    true_wavelength = true_wavelength[:-1] + np.diff(true_wavelength) / 2.
    data = h5f['2d_spectra/0.0'].value.flatten()

# Load A matrix - adjust n for different sampling of A
take_every_n_entry = 5
with h5sparse.File("example_data/maroonx.hdf") as h5f:
    a = h5f['A'].value.todense()[:, ::take_every_n_entry] # Do I have to normalize A?
    a_wavelength = h5f['wavelength'].value[::take_every_n_entry]

# add readout noise to data, to avoid singularities in Matrix N^(-1).
# Is this valid? What about the negative values in data then ?
data += np.random.normal(0, 3, len(data))

# Calculate N^(-1)
Ninv = np.diag(1. / (9. + data))
# Calculate a.T * N^(-1) * a
ana = np.linalg.multi_dot([a.T, Ninv, a])
# Calculate (a.T * N^(-1) * a)^(-1)
anainv = np.linalg.inv(ana)
# Calculate a.T * N^(-1)
an = np.dot(a.T, Ninv)
# Calculate (a.T * N^(-1) * a)^(-1) * a.T * N^(-1)
anaan = np.dot(anainv, an)
# Calculate deconvolved spectrum
f = np.dot(anaan, data).T

# Calculate R - doesn't work!
cinv = ana
w, v = np.linalg.eig(cinv)
eigdiag = np.sqrt(np.diag(w))
q = np.dot(v, eigdiag)
sinv = 1. / np.trace(q)
R = np.diag(sinv*np.diag(q))
data2d = np.dot(a, f)
freconvolved = np.dot(R, f)

plt.figure()
plt.title("Simulated 2D spectrum")
plt.imshow(data.reshape(30, 200))

plt.figure()
plt.title("2d reconstructed - sanity check")
plt.imshow(data2d.reshape(30, 200))

plt.figure()
plt.title("1d extracted spectra")
plt.plot(true_wavelength, true_data / np.max(true_data) * np.max(f), linewidth=3, label='True data')  # scaled so it matches f
plt.plot(a_wavelength, f, label='deconvolved spectrum')
plt.plot(a_wavelength, freconvolved, label='reconvolved spectrum')
plt.legend()
plt.show()