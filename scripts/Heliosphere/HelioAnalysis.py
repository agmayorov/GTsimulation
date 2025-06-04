import os
import sys
import inspect
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

angles = []

T_start = []
T_mod = []

V_start = []
V_mod = []

R_start = []
R_mod = []


def draw_hist(arr, label="Label", bins=50, area=1, div_bins=True, **kwargs):
    arr = np.array(arr)
    # arr = arr[arr<7.5]
    values, bins = np.histogram(arr, bins=bins)
    if area is None:
        area = np.sum(np.diff(bins) * values)
    centers = 0.5 * (bins[1:] + bins[:-1])
    div = np.diff(bins) if div_bins else 1
    plt.bar(centers, values / area / div, width=np.diff(bins), label=label, edgecolor="black",
            yerr=values ** 0.5 / area / div,
            capsize=4, **kwargs)

    return centers, values / area / div


def plot_traj():
    for i, j in prs:
        event = np.load(path + os.sep + files[i], allow_pickle=True)[j]
        R = event["Track"]["Coordinates"]
        T = event["Particle"]["T0"] / 1000
        X, Y, Z = R[:, 0], R[:, 1], R[:, 2]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.set_xlabel(f"X [AU]")
        ax.set_ylabel(f"Y [AU]")
        ax.set_zlabel(f"Z [AU]")

        ax.plot(X, Y, Z, label="Trajectory", color='black', linewidth=1)
        ax.scatter(X[0], Y[0], Z[0], label='Starting point', color='red')
        ax.scatter(X[-1], Y[-1], Z[-1], label="End point", color='blue')

        lons = np.linspace(-180, 180, 30) * np.pi / 180
        lats = np.linspace(-90, 90, 30)[::-1] * np.pi / 180

        x = 70 * np.outer(np.cos(lons), np.cos(lats)).T
        y = 70 * np.outer(np.sin(lons), np.cos(lats)).T
        z = 70 * np.outer(np.ones(np.size(lons)), np.sin(lats)).T

        ax.scatter(x, y, z, color='#00ffff', s=1, label="70 AU surface")
        ax.axis("equal")
        ax.set_title(f"Kin energy = {T} GeV")

        ax.legend()
        plt.show()
        angles = event["Track"]["Angles"] * 180 / np.pi


def plot_spectrums():
    bins = np.logspace(np.log10(0.1), np.log10(20), 13, endpoint=True)
    cen_s, val_s = draw_hist(T_start, label="Initial distribution", bins=bins, ecolor="black")
    cen_m, val_m = draw_hist(T_mod, label="Modulated distribution", alpha=0.5, bins=bins, ecolor="red", area=(7/10)**2)
    # plt.hlines(0.05, xmin=0, xmax=20, linestyles="--", linewidth=2, colors='#BB00BB')
    plt.xlabel("T [GeV]")
    plt.ylabel("Spectrum [num/GeV]")
    plt.xscale("log")
    plt.yscale("log")


    def func(T, a, b):
        return b*T**a

    def mod(T, b, phi):
        m = 0.938
        E = m + T
        return func(T+phi, -2.7, b) * (E**2 - m**2)/((E+phi)**2 - m**2)

    popt_s, pcov_s = curve_fit(mod, cen_s, val_s)
    print(popt_s)
    print(np.diag(pcov_s))


    popt_m, pcov_m = curve_fit(func, cen_m, val_m)
    print(popt_m)
    print(np.diag(pcov_m))

    x = np.logspace(np.log10(0.1), np.log10(20), 50)
    plt.plot(x, func(x, *popt_s), label="Fit of initial", color="black", linewidth=2)
    plt.plot(x, mod(x, *popt_m), label="Fit of modulated", color="red", linewidth=2)


    plt.legend()
    plt.show()


def Cart2Sphere(x, y, z):
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return r, theta, phi


def Cart2SpherVec(R, V):
    r, theta, phi = Cart2Sphere(R[:, 0], R[:, 1], R[:, 2])
    Vr = np.sin(theta) * np.cos(phi) * V[:, 0] + np.sin(theta) * np.sin(phi) * V[:, 1] + np.cos(theta) * V[:, 2]
    Vtheta = np.cos(theta) * np.cos(phi) * V[:, 0] + np.cos(theta) * np.sin(phi) * V[:, 1] - np.sin(theta) * V[:, 2]
    Vphi = -np.sin(phi) * V[:, 0] + np.cos(phi) * V[:, 1]

    theta = np.arccos(-np.sum(V * R, axis=1) / r)
    phi = np.arctan2(Vphi, Vtheta)

    return Vr, Vtheta, Vphi, theta, phi


def plot_vel():
    Vr_start, Vtheta_start, Vphi_start, theta_start, phi_start = Cart2SpherVec(R_start, V_start)
    Vr_mod, Vtheta_mod, Vphi_mod, theta_mod, phi_mod = Cart2SpherVec(R_mod, V_mod)
    draw_hist(phi_start, bins=20, label="Azimuthal angle of initial particles")
    draw_hist(phi_mod, bins=10, label="Azimuthal angle of endpoint", alpha=0.5, ecolor="red")
    plt.legend()
    plt.xlabel("$\\varphi$")
    plt.show()
    draw_hist(np.sin(theta_start) ** 2, bins=20, label="Polar angle of initial particles", div_bins=False)
    draw_hist(np.sin(theta_mod) ** 2, bins=10, label="Polar angle of endpoint", alpha=0.5, ecolor="red", div_bins=False)
    plt.legend()
    plt.xlabel("$\sin^2\\theta$")
    plt.show()


def plot_coords():
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    ax.set_xlabel(f"X [AU]")
    ax.set_ylabel(f"Y [AU]")
    ax.set_zlabel(f"Z [AU]")
    ax.scatter(R_start[:, 0] / 100, R_start[:, 1] / 100, R_start[:, 2] / 100, label="Normalized Initial points", s=3)
    ax.legend()
    ax = fig.add_subplot(122, projection='3d')
    ax.set_xlabel(f"X [AU]")
    ax.set_ylabel(f"Y [AU]")
    ax.set_zlabel(f"Z [AU]")
    ax.scatter(R_mod[:, 0] / 100, R_mod[:, 1] / 100, R_mod[:, 2] / 100, label="Normalized ending points", s=3)
    ax.legend()
    plt.show()

    r_s, theta_s, phi_s = Cart2Sphere(R_start[:, 0], R_start[:, 1], R_start[:, 2])
    r_m, theta_m, phi_m = Cart2Sphere(R_mod[:, 0], R_mod[:, 1], R_mod[:, 2])

    draw_hist(np.cos(theta_s), bins=20, label="Polar angle of initial particles", div_bins=False)
    draw_hist(np.cos(theta_m), bins=10, label="Polar angle of endpoint", alpha=0.5, ecolor="red", div_bins=False)
    plt.legend()
    plt.xlabel("$\cos\\theta$")
    plt.show()

    T = np.array(T_mod)
    draw_hist(T[np.abs(np.cos(theta_m)) < 0.5], bins = np.logspace(np.log10(0.1), np.log10(10), 15, endpoint=True),
              div_bins=True, label="Energies of particles with $|\cos\\theta<0.5|$")
    draw_hist(T[np.abs(np.cos(theta_m)) > 0.5], div_bins=True, label="Energies of particles with $|\cos\\theta>0.5|$",
              alpha=0.5, bins = np.logspace(np.log10(0.1), np.log10(10), 15, endpoint=True))
    plt.legend()
    plt.xlabel("T [Gev]")
    plt.yscale("log")
    plt.xscale("log")
    plt.show()


for num in range(0, 10):
    path = f"..{os.sep}tests{os.sep}ParkerPR100Noise{num}"

    files = os.listdir(path)
    prs = []

    for i in range(len(files)):
        try:
            if (not "Upd" in files[i]) or ('txt' in files[i]):
                continue

            file = np.load(path + os.sep + files[i], allow_pickle=True)
            for j in range(len(file)):
                T_start.append(file[j]["Particle"]["T0"] / 1000)
                V_start.append(file[j]["Track"]["Velocities"][0])
                R_start.append(file[j]["Track"]["Coordinates"][0])
                # angles.extend(file[j]["Track"]["Angles"] * 180 / np.pi)
                if file[j]["WOut"] == 3:
                    T_mod.append(file[j]["Particle"]["T0"] / 1000)
                    V_mod.append(file[j]["Track"]["Velocities"][-1])
                    R_mod.append(file[j]["Track"]["Coordinates"][-1])
                    prs.append((i, j))
        except:
            continue

    # plot_traj()

R_start = np.array(R_start)
R_mod = np.array(R_mod)

V_start = np.array(V_start)
V_mod = np.array(V_mod)

print(len(T_mod) / (len(T_start)) * 100)

print(len(T_mod))
print(len(T_start))

plot_spectrums()

# plot_coords()

# plot_vel()

# plt.hist(angles, 100, edgecolor="black")
# plt.show()
