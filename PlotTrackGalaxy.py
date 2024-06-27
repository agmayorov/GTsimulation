import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel(f"X [kpc]")
ax.set_ylabel(f"Y [kpc]")
ax.set_zlabel(f"Z [kpc]")
events = np.load("Galaxy.npy", allow_pickle=True)
for event in events:
    R = event["Track"]["Coordinates"]
    X, Y, Z = R[:, 0], R[:, 1], R[:, 2]
    T = event["Particle"]["T0"] / 1000000000000
    Clock = event["Track"]["Clock"][-1] / 100000000
    ax.plot(X, Y, Z, label=f"E = {np.round(T, 2)} EeV, Time = {Clock}$\\cdot 10^8$ sec", linewidth=1)
    ax.scatter(X[-1], Y[-1], Z[-1], label="End point")

ax.scatter(X[0], Y[0], Z[0], label='Starting point')
ax.scatter(0, 0, 0, s=50, label = "Center", color='black')
phi = np.linspace(-180, 180, 30) * np.pi/180
r = np.linspace(0, 20, 30)[::-1]

x = np.outer(np.cos(phi), r).T
y = np.outer(np.sin(phi), r).T
z = x*0

ax.scatter(x, y, z, s=1, label='20 kpc disk')

lons = np.linspace(-180, 180, 30) * np.pi/180
lats = np.linspace(-90, 90, 30)[::-1] * np.pi/180

x = 20*np.outer(np.cos(lons), np.cos(lats)).T
y = 20*np.outer(np.sin(lons), np.cos(lats)).T
z = 20*np.outer(np.ones(np.size(lons)), np.sin(lats)).T
ax.scatter(x, y, z, s=1, label='20 kpc sphere')

ax.axis('equal')
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()