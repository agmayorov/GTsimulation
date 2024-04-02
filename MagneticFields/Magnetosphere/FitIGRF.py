import numpy as np
import scipy.io

import matplotlib.pyplot as plt

coeffs = scipy.io.loadmat("HarmonicCoeffsIGRF.mat")

d = coeffs["d"][0]
g10 = coeffs["g10"][0]
g11 = coeffs["g11"][0]
h11 = coeffs["h11"][0]

t = np.linspace(np.min(d), np.max(d), 500)

plt.plot(d, g10, d, g11, d, h11)
plt.show()

g10_poly = np.polyfit(d, g10, 10)
g11_poly = np.polyfit(d, g11, 10)
h11_poly = np.polyfit(d, h11, 10)

g10sm = np.poly1d(g10_poly)
g11sm = np.poly1d(g11_poly)
h11sm = np.poly1d(h11_poly)

plt.plot(d, g10, 'o--')
plt.plot(t, g10sm(t))
plt.show()

plt.plot(d, g11, 'o--')
plt.plot(t, g11sm(t))
plt.show()

plt.plot(d, h11, 'o--')
plt.plot(t, h11sm(t))
plt.show()


Coeffs = {"d": d, "g10": g10, "g10_fit": g10_poly,
                  "g11": g11, "g11_fit": g11_poly,
                  "h11": h11, "h11_fit": h11_poly}

np.save("HarmonicCoeffsIGRF.npy", Coeffs)