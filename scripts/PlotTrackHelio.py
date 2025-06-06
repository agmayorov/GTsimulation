import numpy as np
import matplotlib.pyplot as plt
from gtsimulation.Global import Units

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_xlabel(f"X [AU]")
ax.set_ylabel(f"Y [AU]")
ax.set_zlabel(f"Z [AU]")

ax.scatter(0, 0, 0, marker="*", color='orange', label="The Sun")
file = np.load("test_Parker10GeV_0.npy", allow_pickle=True)
for i, event in enumerate(file):
    R = event["Track"]["Coordinates"] / Units.AU #from meters to AU
    X, Y, Z = R[:, 0], R[:, 1], R[:, 2]

    if i < len(file)-1:
        ax.plot(X, Y, Z, color='black', linewidth=1)
        ax.scatter(X[0], Y[0], Z[0], color='red')
        ax.scatter(X[-1], Y[-1], Z[-1], color='blue')
    else:
        ax.plot(X, Y, Z, label="Trajectory", color='black', linewidth=1)
        ax.scatter(X[0], Y[0], Z[0], label='Starting point', color='red')
        ax.scatter(X[-1], Y[-1], Z[-1], label="End point", color='blue')

lons = np.linspace(-180, 180, 30) * np.pi/180
lats = np.linspace(-90, 90, 30)[::-1] * np.pi/180

x = np.outer(np.cos(lons), np.cos(lats)).T
y = np.outer(np.sin(lons), np.cos(lats)).T
z = np.outer(np.ones(np.size(lons)), np.sin(lats)).T

ax.scatter(x, y, z, color='#00ffff', s=1)

ax.legend()
plt.show()