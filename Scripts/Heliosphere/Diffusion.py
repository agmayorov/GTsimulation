import os

import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import curve_fit

path = "../../tests/ParkerApr2"

au2cm = 1.5e13

dt = 0.01
radis = []
files = os.listdir(path)

Nmax = 2000 / 0.01
Nmin = 10

Ns = np.linspace(Nmin, Nmax, 50, dtype=int)
# Ns = np.arange(Nmin, Nmax, 500, dtype=int)
# files = ["noise_apr_out_1r.npy"]

for file in files:
    if not file.endswith("r.npy"):
        continue

    radi = float(file[:-5].split("_")[-1])
    radis.append(radi)

    particles = np.load(path + os.sep + file, allow_pickle=True)
    for event in particles:
        T = event["Particle"]["T0"]
        # if T != 200:
        #     continue

        R = event["Track"]["Coordinates"]
        X, Y, Z = R[:, 0], R[:, 1], R[:, 2]
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        #
        # ax.set_xlabel(f"X [AU]")
        # ax.set_ylabel(f"Y [AU]")
        # ax.set_zlabel(f"Z [AU]")
        # ax.plot(X, Y, Z, label="Trajectory", color='black', linewidth=1)
        # ax.scatter(X[0], Y[0], Z[0], label='Starting point', color='red')
        # ax.scatter(X[-1], Y[-1], Z[-1], label="End point", color='blue')
        # ax.set_title(f"$R_0 = {radi}$ au, $T = {T / 1000}$ GeV")
        #
        # lons = np.linspace(-180, 180, 30) * np.pi / 180
        # lats = np.linspace(-90, 90, 30)[::-1] * np.pi / 180
        #
        # x = radi * np.outer(np.cos(lons), np.cos(lats)).T
        # y = radi * np.outer(np.sin(lons), np.cos(lats)).T
        # z = radi * np.outer(np.ones(np.size(lons)), np.sin(lats)).T
        #
        # ax.scatter(x, y, z, color='#00ffff', s=1, label=f'{radi} AU surface')
        # ax.axis('equal')
        #
        # plt.show()

        # Angle = event["Track"]["Angles"]
        # plt.hist(Angle, bins=200, edgecolor="black")
        # plt.title(f"$R_0 = {radi}$ au, $T = {T / 1000}$ GeV")
        # plt.xlabel("$\\alpha$", fontsize=12)
        # plt.show()
        D = []
        for N in Ns:
            tau = N * dt
            # angle_n = np.arctan2(np.linalg.norm(np.cross(R[:-N, :], R[N:, :]), axis=1),
            #                      np.sum(R[:-N, :]*R[N:, :], axis=1))
            lag_n = np.sqrt(np.sum(((R[:-N, :] - R[N:, :])*au2cm) ** 2, axis=1))
            # plt.hist(lag_n / np.sqrt(2 * tau) / 2, bins=200, edgecolor="black")
            # # plt.hist(angle_n, bins=200, edgecolor="black")
            # plt.xlabel("$\\frac{\Delta r}{2\sqrt{2\\tau}}$ [au/sec$^{1/2}$]", fontsize=12)
            # # plt.xlabel("$\\alpha_n$ [rad]", fontsize=12)
            # plt.title(f"$R_0 = {radi}$ au, $T = {T / 1000}$ GeV, $\\tau = {tau}$ sec")
            # # plt.yscale("log")
            # plt.show()
            values, bins = np.histogram(lag_n, bins=200)
            centers = 0.5 * (bins[1:] + bins[:-1])

            mode = centers[np.max(values) == values]

            D.append(np.mean((lag_n / np.sqrt(tau))**2))

        D = np.array(D)
        # plt.hist(D, bins=15, edgecolor="black")
        plt.plot(Ns*dt, D, label=f"{T/1000} GeV")
        # plt.title(f"$T = {T / 1000}$ GeV")
        plt.title(f"$T = {radi}$ au")

    break
plt.legend()
plt.xlabel("$\\tau$ [sec]", fontsize=12)
plt.ylabel("$\\frac{{\langle\Delta r^2\\rangle}}{{\\tau}}$ [cm$\cdot$sec$^{{-1}}$]", fontsize=12)
plt.show()

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
#     plt.ylabel(r"$\frac{\sigma^2_{\alpha}}{\Delta t}$")
#     plt.grid(True, which="both")
#     plt.legend()
#     plt.show()
