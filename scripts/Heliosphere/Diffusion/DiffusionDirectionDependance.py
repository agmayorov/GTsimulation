import os
from datetime import datetime

import matplotlib.animation as animation

import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 15})

import numpy as np

from scipy.optimize import curve_fit
from tqdm import tqdm

from gtsimulation.MagneticFields.Heliosphere import Parker
from scripts.draw_tools import *

au2cm = 1.5e13

Date = datetime(2008, 1, 1)
parker = Parker(date=Date, use_reg=True, use_noise=False)

c_cm = 3e10
c_au = c_cm/au2cm

dt = 0.01*100
Ns = np.logspace(1, 5.8, 30, dtype=int)

def update_lines(num, walks, lines):
    for line, walk in zip(lines, walks):
        line.set_data_3d(walk[:num, :].T)
    return lines


def basis(Bx, By, Bz):
    B = np.sqrt(Bx**2 + By**2 + Bz**2)
    ez = np.array([Bx, By, Bz])/B

    n = np.array([0, 1, 0])
    if np.linalg.norm(np.cross(n, ez)) < 1e-3:
        n = np.array([0, 1, 0])

    ex = np.cross(n, ez)/np.linalg.norm(np.cross(n, ez))
    ey = np.cross(ez, ex)

    return ex, ey, ez

paths = [f"../../../tests/DiffUniform{i}" for i in range(20, 24)]
Bx, By, Bz = None, None, None
ex, ey, ez = None, None, None
time = None


Kzz = dict()
N = 5
phi_vals = np.linspace(-np.pi, np.pi, N, endpoint=True)
cos_theta_vals = np.linspace(-1, 1, N, endpoint=True)

# phi_vals, cos_theta_vals = np.meshgrid(phi_vals, cos_theta_vals)

for path in paths:
    files = os.listdir(path)
    for file in files:
        if not file.endswith(".npy"):
            continue
        particles = np.load(path + os.sep + file, allow_pickle=True)
        for event in particles:
            T = event["Particle"]["T0"]
            M = event["Particle"]["M"]
            V_norm = np.sqrt(1 - (M/(T+M))**2) * c_au

            R = event["Track"]["Coordinates"]
            X, Y, Z = R[:, 0], R[:, 1], R[:, 2]

            V = event["Track"]["Velocities"][0] * V_norm
            Vx, Vy, Vz = V[0], V[1], V[2]
            theta = np.arccos(Vz/V_norm)
            phi = np.arctan2(Vy, Vx)

            if Bx is None:
                Bx, By, Bz = parker.CalcBfield(X[0], Y[0], Z[0])
                ex, ey, ez = basis(Bx, By, Bz)
                time = np.arange(len(X))[1:]*dt

            delta_r = R[1:] - R[0, :]

            delta_r_z = delta_r @ ez
            delta_r_x = delta_r @ ex
            delta_r_y = delta_r @ ey

            Vx = V @ ex
            Vy = V @ ey

            kzz = delta_r_z**2 / (2*time)

            for i in range(len(phi_vals)):
                if phi_vals[i] <= phi < phi_vals[i + 1]:
                    for j in range(len(cos_theta_vals)):
                        if cos_theta_vals[j] <= np.cos(theta) < cos_theta_vals[j + 1]:
                            if (phi_vals[i], cos_theta_vals[j]) in Kzz:
                                Kzz[(phi_vals[i], cos_theta_vals[j])].append(kzz)
                            else:
                                Kzz[(phi_vals[i], cos_theta_vals[j])] = [delta_r_z ** 2 / (2 * time)]
P = []
KZZ = []
dKZZ = []
plt.figure()
for phi, theta in sorted(Kzz.keys()):
    pt = plt.plot(time, np.mean(Kzz[(phi, theta)], axis=0)*au2cm**2, label=rf"$\varphi={np.round(phi, 2)}, \theta={np.round(np.arccos(theta),2)}$")

plt.legend()
plt.ylabel("K, [$cm^2/sec$]")
plt.xlabel("t, [sec]")
plt.xscale('log')
plt.yscale('log')
plt.title(rf"$\kappa_{{zz}}$ for protons $\vec{{r}} = (\frac{{1}}{{\sqrt{{2}}}}, \frac{{1}}{{\sqrt{{2}}}}, 0) au$ T={T} MeV")


plt.show()

