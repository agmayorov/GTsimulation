import numpy as np
import matplotlib.pyplot as plt
from gtsimulation.Global import Units

plt.rcParams.update({'font.size': 15})

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel(f"X [kpc]")
ax.set_ylabel(f"Y [kpc]")
ax.set_zlabel(f"Z [kpc]")
events = np.load("tests/GalaxySingle/tracks.npy", allow_pickle=True)
for event in events:
    R = event["Track"]["Coordinates"] / Units.kpc #from meters to kpc
    X, Y, Z = R[:, 0], R[:, 1], R[:, 2]
    T = event["Particle"]["T0"] / 1000000000000
    Clock = event["Track"]["Clock"][-1] * 3.16887646e-8
    ax.plot(X, Y, Z, label=f"E = {np.round(T, 2)} EeV, Time = {np.round(Clock,2)} years", linewidth=1)
    ax.scatter(X[-1], Y[-1], Z[-1], label="End point", s=30)

ax.scatter(X[0], Y[0], Z[0], label='Starting point', s=30)
ax.scatter(0, 0, 0, s=50, label = "Center of the galaxy", color='black')
phi = np.linspace(-180, 180, 30) * np.pi/180
r = np.linspace(0, 28.5, 30)[::-1]

x = np.outer(np.cos(phi), r).T-8.5
y = np.outer(np.sin(phi), r).T
z = x*0

ax.scatter(x, y, z, s=0.5, label='28.5 kpc disk')

lons = np.linspace(-180, 180, 30) * np.pi/180
lats = np.linspace(-90, 90, 30)[::-1] * np.pi/180

x = 28.5*np.outer(np.cos(lons), np.cos(lats)).T-8.5
y = 28.5*np.outer(np.sin(lons), np.cos(lats)).T
z = 28.5*np.outer(np.ones(np.size(lons)), np.sin(lats)).T
ax.scatter(x, y, z, s=0.5, label='28.5 kpc sphere')

ax.axis('equal')
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax.legend()
plt.show()