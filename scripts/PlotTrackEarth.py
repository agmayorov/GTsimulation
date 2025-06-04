import PIL
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from Global import Units

file = np.load("IGRFtest/IGRFtest_0.npy", allow_pickle=True)[0]
R = file["Track"]["Coordinates"] / Units.RE #from meters to RE
X, Y, Z = R[:, 0], R[:, 1], R[:, 2]
# X/=6378.1371
# Y/=6378.1371
# Z/=6378.1371
# print(f"Path: {file['Track']['Path'][-1]}")

print(X[0], Y[0], Z[0])
print(X[-1], Y[-1], Z[-1])

atmo_height = 500/6000
bm = PIL.Image.open('../Earth.jpg')
bm = np.array(bm.resize([int(d/1.5) for d in bm.size]))/256.

lons = np.linspace(-180, 180, bm.shape[1]) * np.pi/180
lats = np.linspace(-90, 90, bm.shape[0])[::-1] * np.pi/180

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = np.outer(np.cos(lons), np.cos(lats)).T
y = np.outer(np.sin(lons), np.cos(lats)).T
z = np.outer(np.ones(np.size(lons)), np.sin(lats)).T

lons = np.linspace(-180, 180, 30) * np.pi/180
lats = np.linspace(-90, 90, 30)[::-1] * np.pi/180

xa = (1+atmo_height)*np.outer(np.cos(lons), np.cos(lats)).T
ya = (1+atmo_height)*np.outer(np.sin(lons), np.cos(lats)).T
za = (1+atmo_height)*np.outer(np.ones(np.size(lons)), np.sin(lats)).T
ax.plot_surface(x, y, z, rstride=4, cstride=4, facecolors = bm)
ax.scatter(xa, ya, za, c="#00ffff", s=0.5)

ax.set_xlabel(f"X [RE]")
ax.set_ylabel(f"Y [RE]")
ax.set_zlabel(f"Z [RE]")

ax.plot(X, Y, Z, label="Trajectory", color='black', linewidth=1)
ax.scatter(X[::15], Y[::15], Z[::15], color='black', s=0.01)
ax.scatter(X[0], Y[0], Z[0], label='Starting point', color='red')
ax.scatter(X[-1], Y[-1], Z[-1], label="End point", color='blue')

ax.axis("equal")
# ax.set_xlim([-3, 3])
# ax.set_ylim([-3, 3])


ax.legend()

plt.show()

