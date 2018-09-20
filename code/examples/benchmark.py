# simple benchmark
from generator import *
import time

# number of simulated photons
N = int(5E7)

t1 = time.time()
spectrum = Etalon()
spectrograph = MaroonX()
generate_rv_series(spectrograph, spectrum, [0.], photons_per_spectrum=N)
t2 = time.time()
print("Total time for tracing: {:.2f} s".format(t2-t1))
print("Simulating {:.2E} photons per second".format(N/(t2-t1)))