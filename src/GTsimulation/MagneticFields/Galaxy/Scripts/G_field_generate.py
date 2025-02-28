import numpy as np
import scipy.io as sio

# generation seed
seed = 0
np.random.seed(seed)

# define the length of the box
boxSize = 1

# number of grid points along one axis
n = 512

# distance between grid points
dl = boxSize / (n - 1)

lMin = 2 * dl
lMax = boxSize

kMin = dl / lMax
kMax = dl / lMin

sIndex = 5 / 3
alpha = -(sIndex + 2)

n2 = int(np.floor(n / 2)) + 1

K = np.arange(n) / n - np.arange(n) // (n / 2)

ek = np.array(np.meshgrid(K, K, K[:n2]))
k = np.linalg.norm(ek, axis=0)
k[0, 0, 0] = 1.e-8 # to avoid division by zero

n0 = np.array([1., 1., 1.]) / np.sqrt(3)
e1 = np.cross(n0, ek, axisb=0, axisc=0)
e2 = np.cross(ek, e1, axis=0)

area = (ek[0] == ek[1]) & (ek[1] == ek[2])

e1[0, area] = -1.
e1[1, area] =  1.
e1[2, area] =  0.

e2[0, area] =  1.
e2[1, area] =  1.
e2[2, area] = -2.

e1 = e1 / (np.linalg.norm(e1, axis=0) + 1.e-8)
e2 = e2 / (np.linalg.norm(e2, axis=0) + 1.e-8)

theta = np.random.uniform(0., 2 * np.pi, ek[0].shape)
phase = np.random.uniform(0., 2 * np.pi, ek[0].shape)

b = e1 * np.cos(theta[None, :]) + e2 * np.sin(theta[None, :])
b = b * np.random.randn(*theta.shape)[None, :] * k[None, :] ** (alpha / 2)
Bk = b * np.exp(1.j * phase)[None, :]

Bk[:, k < kMin] = 0
Bk[:, k > kMax] = 0

Gx = np.fft.irfftn(Bk[0])
Gy = np.fft.irfftn(Bk[1])
Gz = np.fft.irfftn(Bk[2])

rms = np.sqrt(np.sum(Gx**2 + Gy**2 + Gz**2) / n**3)
Gx = Gx / rms
Gy = Gy / rms
Gz = Gz / rms

x_grid = np.linspace(0, boxSize, n)
y_grid = np.linspace(0, boxSize, n)
z_grid = np.linspace(0, boxSize, n)

filename = f'G_n={n}_boxSize={boxSize}_lMin={lMin:.1e}_lMax={lMax:.1e}_seed={seed}.mat'

mdic = {'Gx': Gx.astype('float32'),
        'Gy': Gy.astype('float32'),
        'Gz': Gz.astype('float32'),
        'x_grid': x_grid,
        'y_grid': y_grid,
        'z_grid': z_grid,
        'boxSize': boxSize,
        'lMin': lMin,
        'lMax': lMax}

sio.savemat(filename, mdic)
