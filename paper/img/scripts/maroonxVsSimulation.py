import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from os.path import dirname
import sys
sys.path.append("../../../code/") # I know it's bad practise...

from generator import MaroonX, Etalon

N = int(5E7)

es = Etalon(11.3)
wl = es.draw_wavelength(N)
spec = MaroonX()
spec.sx=1.15
spec.psf_sig_x = 0.5
spec.psf_sig_y = 1.

simulator = spec.generate_2d_spectrum(wl)[8:25, :80]

maroonx_data = fits.getdata("maroonx.fit")

plt.figure()
plt.imshow(np.flip(maroonx_data, axis=0))
plt.tight_layout()
plt.savefig('../maroonx_data.png', dpi=150)
plt.figure()
plt.imshow(simulator)
plt.tight_layout()
plt.savefig('../simulator_data.png', dpi=150)
plt.show()
