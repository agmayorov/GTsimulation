import numpy as np
import scipy.io


coeffs = scipy.io.loadmat("../Data/G_nCell=250_boxSize=0.5kpc_lMin=4.0pc_lMax=500.0pc_seed=0.mat")

Gx = coeffs["Gx"]
Gy = coeffs["Gy"]
Gz = coeffs["Gz"]

x_grid = coeffs["x_grid"]
y_grid = coeffs["y_grid"]
z_grid = coeffs["z_grid"]

np.save("../Data/G_nCell=250_boxSize=0.5kpc_lMin=4.0pc_lMax=500.0pc_seed=0.npy",
        {"Gx": Gx, "Gy": Gy, "Gz": Gz,
         "x_grid": x_grid, "y_grid": y_grid, "z_grid": z_grid})