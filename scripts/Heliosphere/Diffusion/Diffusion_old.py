import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import curve_fit
from tqdm import tqdm

from gtsimulation.MagneticFields.Heliosphere import Parker
from scripts.draw_tools import *

au2cm = 1.5e13

Date = datetime(2008, 1, 1)
parker = Parker(date=Date, use_reg=True, use_noise=False)

c = 3e10

dt = 0.01
radis = []
Nmax = 2000 / 0.01
Nmin = 10
mx = 0
Ns = np.linspace(Nmin, Nmax, 50, dtype=int)

# paths = ["../../tests/DiffusionPR0"]
# paths.extend([f"../../tests/Diffusion{i}" for i in range(1, 6)])
# Radius = [1, 2, 3, 5, 8, 10]
paths = ["../../tests/DDiffusionPRReg0"]
Radius = [1]

D_perp_arr = []
D_parallel_arr = []

for path in paths:

    D_perp = []
    D_parallel = []
    Vxx = []
    Vyy = []
    Vzz = []

    files = os.listdir(path)
    for N in tqdm(Ns):
        tau = dt * N
        delta_r_perp = []
        delta_r_parallel = []

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        for file in files:
            if not file.endswith(".npy"):
                continue

            radi = 1

            particles = np.load(path + os.sep + file, allow_pickle=True)
            for event in particles:
                T = event["Particle"]["T0"]

                R = event["Track"]["Coordinates"]
                X, Y, Z = R[:, 0], R[:, 1], R[:, 2]
                #
                # ax.set_xlabel(f"X [AU]")
                # ax.set_ylabel(f"Y [AU]")
                # ax.set_zlabel(f"Z [AU]")
                # # plt.plot(X, Y, Z)
                # plt.plot(X[:int(Nmax)], Y[:int(Nmax)], Z[:int(Nmax)])
                # lons = np.linspace(-180, 180, 30) * np.pi / 180
                # lats = np.linspace(-90, 90, 30)[::-1] * np.pi / 180
                #
                # x = np.outer(np.cos(lons), np.cos(lats)).T
                # y = np.outer(np.sin(lons), np.cos(lats)).T
                # z = np.outer(np.ones(np.size(lons)), np.sin(lats)).T

                # ax.scatter(x, y, z, color='#00ffff', s=1)
                # plt.axis("equal")
                # ax.scatter(X[0], X[0], 0, s=50, color="black")
                # ax.arrow3D(X[0], X[0], 0,
                #            0.2 * Bx / B, 0.2 * By / B, 0.2 * Bz / B,
                #            mutation_scale=10)

                dr = np.sqrt((X[N] - X[0]) ** 2 + (Y[N] - Y[0]) ** 2 + (Y[N] - Y[0]) ** 2)
                Bx, By, Bz = parker.CalcBfield(X[0], Y[0], Z[0])
                B = np.sqrt(Bx ** 2 + By ** 2 + Bz ** 2)
                cos = np.abs((Bx * (X[N] - X[0]) + By * (Y[N] - Y[0]) + Bz * (Z[N] - Z[0]))) / (dr * B)
                delta_r_perp.append(dr * np.sqrt(1 - cos ** 2))
                delta_r_parallel.append(dr * cos)
            mx = max(mx, np.max(delta_r_perp))

        # plt.axis("equal")
        # ax.scatter(1 / np.sqrt(2), 1 / np.sqrt(2), 0, s=50, color="black")
        # ax.arrow3D(1 / np.sqrt(2), 1 / np.sqrt(2), 0,
        #            0.1 * Bx / B, 0.1 * By / B, 0.1 * Bz / B,
        #            mutation_scale=10)
        # plt.show()

        D_parallel.append(np.mean(np.array(delta_r_parallel) ** 2) / (2 * tau))

    plt.plot(Ns*dt, D_parallel)
    plt.show()

#         def func_perp(r, D, C):
#             return C * r * np.exp(-r ** 2 / (4 * D * tau))
#
#         def func_parallel(r, D, C):
#             return C * np.exp(-r ** 2 / (4 * D * tau))
#
#
#         delta_r_perp = np.array(delta_r_perp)
#         delta_r_parallel = np.array(delta_r_parallel)
#         values, bins = np.histogram(delta_r_perp, bins=15)
#         area = np.sum(np.diff(bins) * values)
#         values = values / area
#         centers = 0.5 * (bins[1:] + bins[:-1])
#         mode = centers[np.max(values) == values][0]
#
#         # Di = mode**2 / (8*tau)
#         # Di = np.pi**2/16*(np.mean(delta_r))**2/tau
#
#         try:
#             popt, pcov = curve_fit(func_perp, centers, values, p0=[5e-7, (5e-7 * tau) ** (-3 / 2)],
#                                    bounds=([1e16/au2cm ** 2, (tau) ** (-3 / 2)], [1e23/au2cm ** 2, (1e-13 * tau) ** (-3 / 2)]))
#             Di = popt[0]
#             C = popt[1]
#             D_perp.append(Di * au2cm ** 2)
#
#             # r = np.linspace(0, np.max(delta_r_perp), 50)
#             # plt.hist(delta_r_perp, 15, edgecolor="black", density=True)
#             # # plt.bar(centers, values, width=np.diff(bins), edgecolor="black")
#             # # plt.plot(r, func_perp(r, *popt))
#             # plt.title(f"$|\\vec{{r}}_{{\perp}}(t={tau} sec) - \\vec{{r}}(t=0)|$; $D={Di*au2cm**2}\\frac{{cm^2}}{{sec}}$")
#             # plt.xlabel("$\Delta r \quad [au]$")
#             # plt.show()
#
#
#             values, bins = np.histogram(delta_r_parallel, bins=15)
#             area = np.sum(np.diff(bins) * values)
#             values = values / area
#             centers = 0.5 * (bins[1:] + bins[:-1])
#             mode = centers[np.max(values) == values][0]
#
#             # Di = mode**2 / (8*tau)
#             # Di = np.pi**2/16*(np.mean(delta_r))**2/tau
#
#             popt, pcov = curve_fit(func_parallel, centers, values, p0=[5e-7, (5e-7 * tau) ** (-3 / 2)],
#                                    bounds=([1e-13, (tau) ** (-3 / 2)], [1, (1e-13 * tau) ** (-3 / 2)]))
#             Di = popt[0]
#             C = popt[1]
#             D_parallel.append(Di * au2cm ** 2)
#
#             r = np.linspace(0, np.max(delta_r_parallel), 50)
#             plt.hist(delta_r_parallel, 15, edgecolor="black", density=True)
#             # plt.bar(centers, values, width=np.diff(bins), edgecolor="black")
#             # plt.plot(r, func_parallel(r, *popt))
#             plt.title(f"$|\\vec{{r}}_{{||}}(t={tau} sec) - \\vec{{r}}(t=0)|$; $D={Di*au2cm**2}\\frac{{cm^2}}{{sec}}$")
#             plt.xlabel("$\Delta r \quad [au]$")
#             plt.show()
#
#         except:
#             D_perp.append(0)
#             D_parallel.append(0)
#
#
#
#     D_parallel_arr.append(D_parallel)
#     D_perp_arr.append(D_perp)
#
# plt.figure()
#
# for i in range(len(Radius)):
#     plt.plot(Ns * dt, D_perp_arr[i], label=f"R = {Radius[i]}au")
#
# plt.legend()
# plt.xlabel("$\\tau\\quad$[sec]", fontsize=12)
# plt.ylabel("$D_{{\perp}}\\quad[\\frac{{cm^2}}{{sec}}]$", fontsize=12)
# # plt.show()
# plt.figure()
#
# for i in range(len(Radius)):
#     plt.plot(Ns * dt, D_parallel_arr[i], label=f"R = {Radius[i]}au")
# plt.legend()
# plt.xlabel("$\\tau\\quad$[sec]", fontsize=12)
# plt.ylabel("$D_{{\||}}\\quad[\\frac{{cm^2}}{{sec}}]$", fontsize=12)
# # plt.show()
#
# T = event["Particle"]["T0"]
# M = event["Particle"]["M"]
# V = c * np.sqrt(1 - (M / (T + M)) ** 2)
#
# plt.figure()
# l_per = []
# for i in range(len(Radius)):
#     lambda_perp = D_perp_arr[i] / (3 * V)
#     if Radius[i] != 8:
#         l_per.append(lambda_perp[-1] / au2cm)
#     plt.plot(Ns * dt, lambda_perp / au2cm, label=f"R = {Radius[i]}au")
# plt.legend()
# plt.xlabel("$\\tau\\quad$[sec]", fontsize=12)
# plt.ylabel("$\lambda_{{\perp}}\\quad[au]$", fontsize=12)
# # plt.show()
#
# plt.figure()
# l_par = []
# for i in range(len(Radius)):
#     lambda_parallel = D_parallel_arr[i] / (3 * V)
#     if Radius[i] != 8 and Radius[i] != 10:
#         l_par.append(lambda_parallel[-1] / au2cm)
#     plt.plot(Ns * dt, lambda_parallel / au2cm, label=f"R = {Radius[i]}au")
# plt.xlabel("$\\tau\\quad$[sec]", fontsize=12)
# plt.ylabel("$\lambda_{{\||}}\\quad[au]$", fontsize=12)
# plt.legend()


# def func2(x, a, b):
#     return a * x ** b
#
#
# plt.figure()
# plt.scatter([1, 2, 3, 5], l_par, label="$\lambda_{{\||}}$")
# popt, pcov = curve_fit(func2, np.array([1, 2, 3, 5]), l_par)
# print(popt)
# print(np.diag(pcov))
# plt.plot([1, 2, 3, 5, 10], func2(np.array([1, 2, 3, 5, 10]), *popt), label="$\lambda_{{\||}}$")
# plt.scatter([1, 2, 3, 5, 10], l_per, label="$\lambda_{{\perp}}}$")
# popt, pcov = curve_fit(func2, np.array([1, 2, 3, 5, 10]), l_per)
# print()
# print(popt)
# print(np.diag(pcov))
# plt.plot([1, 2, 3, 5, 10], func2(np.array([1, 2, 3, 5, 10]), *popt), label="$\lambda_{{\perp}}}$")
# plt.xlabel("R [au]", fontsize=12)
# plt.ylabel("$\lambda\\quad[au]$", fontsize=12)
# plt.legend()

# plt.show()

# plt.legend()
# plt.xlabel("$\\tau$ [sec]", fontsize=12)
# plt.ylabel("$\\frac{{\langle\Delta r^2\\rangle}}{{\\tau}}$ [cm$\cdot$sec$^{{-1}}$]", fontsize=12)
# plt.show()
#
# plt.title(f'R=1au, T=0.2 GeV')
# plt.hist(K, 20, label="Distribution of diffusion coefficient", edgecolor="black")
# plt.legend()
# plt.show()

# files = np.load("../../tests/ParkerApr/noise_apr_out_100_20r_10_7.npy", allow_pickle=True)
# for event in files:
#     T = event["Particle"]["T0"]
#
#     R = event["Track"]["Coordinates"]
#     X, Y, Z = R[:, 0], R[:, 1], R[:, 2]
#
#     Angle = event["Track"]["Angles"]
#     r = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
#
#     plt.scatter(r[::20], Angle[::20], s=1)
#     plt.yscale("log")
#     plt.xscale("log")
#     plt.xlabel("r [au]")
#     plt.ylabel(r"$\Delta\alpha$")
#     plt.grid(True, which="both")
#     plt.show()
#
#     D_av = []
#     r_av = []
#
#     batch_size = 2000
#
#     for i in range(0, len(r), batch_size):
#         D_av.append(np.var(Angle[i: i + batch_size])/np.mean(Angle[i: i + batch_size])**2)
#         r_av.append(np.mean(r[i: i + batch_size]))
#
#     D_av = np.array(D_av)
#     r_av = np.array(r_av)
#
#     idx = np.argsort(r_av)
#     r_av = r_av[idx]
#     D_av = D_av[idx]
#
#
#
#     def func(rad, a, b):
#         return b * rad ** a
#
#
#     # try:
#     #     popt_s, pcov_s = curve_fit(func, r_av[idx], D_av[idx])
#     #     print(popt_s)
#     #     print(np.diag(pcov_s))
#     #     plt.plot(np.linspace(np.min(r_av), np.max(r_av)), func(np.linspace(np.min(r_av), np.max(r_av)), *popt_s), linewidth=2,
#     #              zorder=2, label=f"$\propto R^{{{np.round(popt_s[0], 2)}}}$")
#     # except:
#     #     pass
#
#     plt.plot(r_av, D_av, zorder=1, label=f"Energy {T/1000} GeV")
#     plt.yscale("log")
#     plt.xscale("log")
#     plt.xlabel("r [au]")
#     plt.ylabel(r"$\frac{\sigma^
