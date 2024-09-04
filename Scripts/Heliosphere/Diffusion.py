import os
from datetime import datetime

import matplotlib.animation as animation

import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import curve_fit
from tqdm import tqdm

from MagneticFields.Heliosphere import Parker
from Scripts.draw_tools import *

au2cm = 1.5e13

Date = datetime(2008, 1, 1)
parker = Parker(date=Date, use_reg=True, use_noise=False)

c_cm = 3e10
c_au = c_cm/au2cm

dt = 0.01
Ns = np.logspace(1, 5.8, 30, dtype=int)

def update_lines(num, walks, lines):
    for line, walk in zip(lines, walks):
        line.set_data_3d(walk[:num, :].T)
    return lines


def basis(Bx, By, Bz):
    B = np.sqrt(Bx**2 + By**2 + Bz**2)
    ez = np.array([Bx, By, Bz])/B

    n = np.array([1, 0, 0])
    if np.linalg.norm(np.cross(n, ez)) < 1e-3:
        n = np.array([0, 1, 0])

    ex = np.cross(n, ez)/np.linalg.norm(np.cross(n, ez))
    ey = np.cross(ez, ex)

    return ex, ey, ez

paths = ["../../tests/DiffNew1"]
Bx, By, Bz = None, None, None
ex, ey, ez = None, None, None
time = None

for path in paths:
    files = os.listdir(path)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')


    walks = []

    for file in files:
        if not file.endswith(".npy"):
            continue
        particles = np.load(path + os.sep + file, allow_pickle=True)
        i = 0
        for event in particles:
            R = event["Track"]["Coordinates"]
            X, Y, Z = R[:, 0], R[:, 1], R[:, 2]
            if Bx is None:
                Bx, By, Bz = parker.CalcBfield(X[0], Y[0], Z[0])
                B = np.sqrt(Bx**2 + By**2 + Bz**2)
                ex, ey, ez = basis(Bx, By, Bz)
                time = np.arange(len(X))[1:]*dt
            walks.append(R[:Ns[-1]:50])
            # ax.set_xlabel(f"X [AU]")
            # ax.set_ylabel(f"Y [AU]")
            # ax.set_zlabel(f"Z [AU]")
            # plt.plot(X[:Ns[-1]], Y[:Ns[-1]], Z[:Ns[-1]])
            i+=1
            if i == 20:
                break
        break

    lines = [ax.plot([], [], [])[0] for _ in range(len(walks))]
    ani = animation.FuncAnimation(
        fig, update_lines, Ns[-1], fargs=(walks, lines), interval=1e-4)

    plt.axis("equal")
    ax.set(xlim3d=(0.55, 0.85), xlabel='X')
    ax.set(ylim3d=(0.55, 0.85), ylabel='Y')
    ax.set(zlim3d=(-0.15, 0.15), zlabel='Z')
    ax.arrow3D(X[0], Y[0], 0,
               0.2*ez[0], 0.2*ez[1], 0.2*ez[2],
               mutation_scale=10, label="ez")
    ax.arrow3D(X[0], Y[0], 0,
               0.2*ex[0], 0.2*ex[1], 0.2*ex[2],
               mutation_scale=5, label="ex")
    ax.arrow3D(X[0], Y[0], 0,
               0.2*ey[0], 0.2*ey[1], 0.2*ey[2],
               mutation_scale=5, label="ey")
    plt.legend()
    plt.show()

    Kzz = []
    Kxx = []
    Kyy = []
    Kxy = []
    Kyx = []


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

            Kzz.append(delta_r_z**2 / (2*time))
            Kxx.append(delta_r_x**2 / (2*time))
            Kyy.append(delta_r_y**2 / (2*time))

            Kxy.append(delta_r_x*Vy)
            Kyx.append(delta_r_y*Vx)

    Kzz = np.array(Kzz)
    Kxx = np.array(Kxx)
    Kyy = np.array(Kyy)

    Kxy = np.array(Kxy)
    Kyx = np.array(Kyx)

    plt.figure()
    plt.plot(time, np.mean(Kzz, axis=0)*au2cm**2)
    plt.legend()
    plt.ylabel("K, [$cm^2/sec$]")
    plt.xlabel("t, [sec]")
    plt.xscale('log')
    plt.yscale('log')
    plt.title("$K_\parallel $")

    plt.figure()
    plt.plot(time, np.mean(Kxx, axis=0)*au2cm**2, label="Kxx")
    plt.plot(time, np.mean(Kyy, axis=0)*au2cm**2, label="Kyy")
    plt.legend()
    plt.ylabel("K, [$cm^2/sec$]")
    plt.xlabel("t, [sec]")
    plt.xscale('log')
    plt.yscale('log')
    plt.title("$K_\perp$")

    plt.figure()
    plt.plot(time, np.mean(Kxy, axis=0)*au2cm**2, label="Kxy")
    plt.ylabel("K, [$cm^2/sec$]")
    plt.xlabel("t, [sec]")
    plt.ylabel("K, [$cm^2/sec$]")
    plt.plot(time, np.mean(Kyx, axis=0)*au2cm**2, label="Kyx")
    plt.legend()
    plt.xscale('log')
    plt.title("$K_A$")

    plt.show()

