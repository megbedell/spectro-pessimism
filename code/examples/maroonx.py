# this example simulates two m-dwarf spectra with 100m RV shift
# It uses the mock-up MaroonX spectrograph and extracts the 2D echellogram using box extraction
from generator import *

# spectrograph model
spectrograph = MaroonX()

# artificial m-dwarf spectrum
spectrum = Phoenix(3600, data_folder='../data')
wl = spectrum.draw_wavelength(int(1E7)) # draw 1E7 photons

# simulate 2d spectrum
echellogram = spectrograph.generate_2d_spectrum(wl)
# box extract 1d spectrum
extracted = np.sum(echellogram, axis=0)


# Do the same but apply 100m/s shift
spectrum.draw_wavelength(int(1E7))
wl = spectrum.apply_rv(100)

echellogram = spectrograph.generate_2d_spectrum(wl)
extracted2 = np.sum(echellogram, axis=0)

plt.figure()
plt.title("Echellogram")
plt.imshow(echellogram)

plt.figure()
plt.plot(spectrograph.calc_wavelength_vector()[1], extracted, label='extracted')
plt.plot(spectrograph.calc_wavelength_vector()[1], extracted2, label='extracted 100m/s shift')
plt.legend()
plt.show()

