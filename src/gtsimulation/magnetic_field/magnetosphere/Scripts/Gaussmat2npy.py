import scipy.io
import numpy as np

name = "../IGRF14/igrf14coeffs"

coeffs = scipy.io.loadmat(name + ".mat")["coefs"]
years = coeffs["year"][0].astype(np.float32)
g = [gg.astype(np.float32) for gg in coeffs["g"][0]]
h = [hh.astype(np.float32) for hh in coeffs["h"][0]]
gh = [ghgh.astype(np.float32) for ghgh in coeffs["gh"][0]]

if 'slope' in coeffs.dtype.names:
    slope = coeffs['slope'][0].astype(int)

    Coeffs = {"years": years, "g": g, "h": h,
              "gh": gh, "slope": slope}
else:
    Coeffs = {"years": years, "g": g, "h": h,
              "gh": gh}
np.save(name + ".npy", Coeffs)
