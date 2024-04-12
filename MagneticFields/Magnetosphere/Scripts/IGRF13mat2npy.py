import scipy.io
import numpy as np
coeffs = scipy.io.loadmat("../Data/igrfcoefs.mat")["coefs"]
years = coeffs["year"][0].astype(int)
g = [gg.astype(np.float32) for gg in coeffs["g"][0]]
h = [hh.astype(np.float32) for hh in coeffs["h"][0]]
gh = [ghgh.astype(np.float32) for ghgh in coeffs["gh"][0]]
slope = coeffs['slope'][0].astype(int)

Coeffs = {"years": years, "g": g, "h": h,
                  "gh": gh, "slope": slope}
np.save("../Data/igrfcoefs.npy", Coeffs)