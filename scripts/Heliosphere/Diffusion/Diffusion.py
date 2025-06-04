import os
from datetime import datetime

import matplotlib.animation as animation

import matplotlib.pyplot as plt

from Global import Units
from Scripts.Heliosphere import misc

plt.rcParams.update({'font.size': 20})

import numpy as np

from scipy.optimize import curve_fit
from tqdm import tqdm

from MagneticFields.Heliosphere import ParkerUniform, Parker
# from Scripts.draw_tools import *

au2cm = 1.5e13
m2cm = 1e2

Date = datetime(2008, 1, 1)
parker = ParkerUniform(x=1/np.sqrt(2), y=1/np.sqrt(2), z= 0, date=Date, use_reg=True, use_noise=False)

c_cm = 3e10
c_m = 3e8
c_au = c_cm/au2cm

# Ns = np.logspace(1, 5.8, 30, dtype=int)
Ns = -1

def update_lines(num, walks, lines):
    for line, walk in zip(lines, walks):
        line.set_data_3d(walk[:num, :].T)
    return lines


def basis(Bx, By, Bz):
    # B = np.sqrt(Bx**2 + By**2 + Bz**2)
    # ez = np.array([Bx, By, Bz])/B
    #
    # n = np.array([0, 1, 0])
    # if np.linalg.norm(np.cross(n, ez)) < 1e-3:
    #     n = np.array([0, 1, 0])
    #
    # ex = np.cross(n, ez)/np.linalg.norm(np.cross(n, ez))
    # ey = np.cross(ez, ex)
    ez = np.array([0, 0, 1])
    ex = np.array([1, 0, 0])
    ey = np.array([0, 1, 0])

    return ex, ey, ez

# paths = [f"../../../tests/DiffUniformUpdNew{i}" for i in range(1, 24)]
paths = [f"../../../tests/DiffUniform8.{i}" for i in range(1, 19)]
Bx, By, Bz = None, None, None
ex, ey, ez = None, None, None
time = None


Kzz = dict()
lzz = dict()
lzz_v = dict()
Kxx = dict()
lxx = dict()
Kyy = dict()
lyy = dict()
Kxy = dict()
Kyx = dict()

# #
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# #
# # walks = []
for path in paths:
    if not os.path.exists(path):
        continue
    files = os.listdir(path)
    for file in files:
        if not file.endswith(".npy"):
            continue
        try:
            particles = np.load(path + os.sep + file, allow_pickle=True)
        except:
            continue
        i = 0
        for event in particles:
            R = event["Track"]["Coordinates"] / Units.AU
            X, Y, Z = R[:, 0], R[:, 1], R[:, 2]
            # if Bx is None:
            #     Bx, By, Bz = parker.CalcBfield(X[0] / Units.AU, Y[0] / Units.AU, Z[0] / Units.AU)
            #     ex, ey, ez = basis(Bx, By, Bz)
# #                 ex, ey, ez = basis(Bx, By, Bz)
#             # walks.append(R[:Ns:50])
#             ax.set_xlabel(f"X [AU]")
#             ax.set_ylabel(f"Y [AU]")
#             ax.set_zlabel(f"Z [AU]")
#             plt.plot(X, Y, Z)
            break
#
#     # lines = [ax.plot([], [], [])[0] for _ in range(len(walks))]
#     # ani = animation.FuncAnimation(
#     #     fig, update_lines, Ns, fargs=(walks, lines), interval=1e-4)
#
# plt.axis("equal")
# ax.set(xlabel='X [au]')
# ax.set(ylabel='Y [au]')
# ax.set(zlabel='Z [au]')
# ax.arrow3D(X[0], Y[0], 0,
#            ez[0], ez[1], ez[2],
#            mutation_scale=10, label="ez")
# # ax.arrow3D(X[0], Y[0], 0,
# #            0.2*ex[0], 0.2*ex[1], 0.2*ex[2],
# #            mutation_scale=5, label="ex")
# # ax.arrow3D(X[0], Y[0], 0,
# #            0.2*ey[0], 0.2*ey[1], 0.2*ey[2],
# # #            mutation_scale=5, label="ey")
# plt.legend()
# plt.show()

for path in paths:
    if not os.path.exists(path):
        continue
    files = os.listdir(path)
    i = 0
    for file in files:
        if i == 5:
            break
        i+=1
        if not file.endswith(".npy"):
            continue
        try:
            particles = np.load(path + os.sep + file, allow_pickle=True)
        except:
            continue
        for event in particles:
            T = event["Particle"]["T0"]
            M = event["Particle"]["M"]
            V_norm = np.sqrt(1 - (M/(T+M))**2) * c_m
            time = event["Track"]["Clock"][1:]

            R = event["Track"]["Coordinates"]
            X, Y, Z = R[:, 0], R[:, 1], R[:, 2]

            V = event["Track"]["Velocities"][0] * V_norm

            if Bx is None:
                Bx, By, Bz = parker.CalcBfield(X[0]/Units.AU, Y[0]/Units.AU, Z[0]/Units.AU)
                ex, ey, ez = basis(Bx, By, Bz)
                # ex = np.array([1, 0, 0])
                # ey = np.array([0, 1, 0])
                # ez = np.array([0, 0, 1])
                # time = (np.arange(len(X))[1:]*dt)[::10]

            delta_r = R[1:] - R[0, :]

            # eig_vec = np.linalg.eig(np.cov(delta_r.T))[1]
            # ez, ex, ey = eig_vec[:, 0], eig_vec[:, 1], eig_vec[:, 2]

            delta_r_z = delta_r @ ez
            delta_r_x = delta_r @ ex
            delta_r_y = delta_r @ ey

            Vx = V @ ex
            Vy = V @ ey

            if T in Kzz:
                Kzz[T].append((delta_r_z**2 ) / (2*time))
                lzz[T].append((delta_r_z**2 ) / (2*time) * 3 / V_norm)
                Kxx[T].append(delta_r_x**2 / (2*time))
                lxx[T].append(delta_r_x**2 / (2*time) * 3 / V_norm)
                Kyy[T].append(delta_r_y**2 / (2*time))
                lyy[T].append(delta_r_y**2 / (2*time) * 3 / V_norm)
            else:
                Kzz[T] = [(delta_r_z**2)/ (2*time)]
                lzz[T] = [(delta_r_z**2) / (2*time) * 3 / V_norm]
                Kxx[T] = [delta_r_x**2 / (2*time)]
                lxx[T] = [delta_r_x**2 / (2*time) * 3 / V_norm]
                Kyy[T] = [delta_r_y**2 / (2*time)]
                lyy[T] = [delta_r_y**2 / (2*time) * 3 / V_norm]

            # Kzz.append(delta_r_z**2 / (2*time))
            # Kxx.append(delta_r_x**2 / (2*time))
            # Kyy.append(delta_r_y**2 / (2*time))
            #
            # Kxy.append(delta_r_x*Vy)
            # Kyx.append(delta_r_y*Vx)

# Kzz = np.array(Kzz)
# Kxx = np.array(Kxx)
# Kyy = np.array(Kyy)
#
# Kxy = np.array(Kxy)
# Kyx = np.array(Kyx)

P = []
KZZ = []
dKZZ = []

KXX = []
dKXX = []

KYY = []
dKYY = []
plt.figure()
for T in sorted(Kzz.keys()):
    idx = time>1e4
    p = np.sqrt((T+M)**2-M**2)
    P.append(p)
    pt = plt.plot(time, np.median(Kzz[T], axis=0)*m2cm**2, label=rf"$\kappa_{{zz}}$, P={np.round(p,2)} MV")
    k_mean = np.mean((np.median(Kzz[T], axis=0)*m2cm**2)[idx])
    k_std = np.sqrt(np.std((np.median(Kzz[T], axis=0)*m2cm**2)[idx])**2)
    KZZ.append(k_mean)
    dKZZ.append(k_std)
    plt.hlines([k_mean], xmin=time[0], xmax=time[-1], linestyles="--", colors=pt[0]._color)
    #
    #
#     pt = plt.plot(time, np.mean(Kxx[T], axis=0)*m2cm**2, label=rf"$\kappa_{{xx}}$, P={np.round(p,2)} MV")
#     k_mean = np.mean((np.mean(Kxx[T], axis=0)*m2cm**2)[idx])
#     k_std = np.std((np.mean(Kxx[T], axis=0)*m2cm**2)[idx])
#     KXX.append(k_mean)
#     dKXX.append(k_std)
#     plt.plot(time, np.mean(Kyy, axis=0)*au2cm**2, label=r"$\kappa_{yy}$")
#     plt.plot(time, 0.5*(np.mean(Kyy, axis=0)+np.mean(Kxx, axis=0))*au2cm**2, label=r"$\kappa_{\perp}$")
#     plt.plot(time, np.mean(Kxx, axis=0)*au2cm**2, label=r"$\kappa_{xx}$")
# # plt.legend()
# plt.ylabel("$\kappa$, [$cm^2/sec$]")
# plt.xlabel("t, [sec]")
# plt.xscale('log')
# plt.yscale('log')
# plt.title(r"protons $\vec{r} = (\frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}}, 0) au$")
# plt.show()
# plt.figure()
# plt.errorbar(P, KZZ, yerr=dKZZ, capsize=4, fmt='o')
plt.ylabel("$\kappa_{zz}$, [$cm^2/sec$]")
plt.xlabel("t, [sec]")
plt.xscale('log')
plt.yscale('log')
plt.grid(which='both')
# plt.title(r"protons $\vec{r} = (\frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}}, 0) au$")
plt.legend()


plt.figure()
for T in sorted(Kxx.keys()):
    idx = time>1e4
    p = np.sqrt((T + M) ** 2 - M ** 2)
    pt = plt.plot(time, np.median(Kxx[T], axis=0)*m2cm**2, label=rf"$\kappa_{{xx}}$, P={np.round(p,2)} MV")
    k_mean = np.mean((np.median(Kxx[T], axis=0)*m2cm**2)[idx])
    k_std = np.sqrt(np.std((np.median(Kxx[T], axis=0)*m2cm**2)[idx])**2)
    KXX.append(k_mean)
    dKXX.append(k_std)
    plt.hlines([k_mean], xmin=time[0], xmax=time[-1], linestyles="--", colors=pt[0]._color)


    pt = plt.plot(time, np.median(Kyy[T], axis=0)*m2cm**2, label=rf"$\kappa_{{yy}}$, P={np.round(p,2)} MV")
    k_mean = np.mean((np.median(Kyy[T], axis=0)*m2cm**2)[idx])
    k_std = np.sqrt(np.std((np.median(Kyy[T], axis=0)*m2cm**2)[idx])**2)
    KYY.append(k_mean)
    dKYY.append(k_std)
    plt.hlines([k_mean], xmin=time[0], xmax=time[-1], linestyles="--", colors=pt[0]._color)

plt.ylabel("$\kappa_{\perp}$, [$cm^2/sec$]")
plt.xlabel("t, [sec]")
plt.xscale('log')
plt.yscale('log')
plt.grid(which='both')
# plt.title(r"protons $\vec{r} = (\frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}}, 0) au$")
plt.legend()


lZZ = []
lZZ_v = []
dlZZ = []
plt.figure()
for T in sorted(lzz.keys()):
    idx = time>1e4
    p = np.sqrt((T + M) ** 2 - M ** 2)
    pt = plt.plot(time, np.median(lzz[T], axis=0)/Units.AU, label=rf"$\lambda_{{zz}}$, P={np.round(p,2)} MV")
    l_mean = np.mean((np.median(lzz[T], axis=0)/Units.AU)[idx])
    l_std = np.std((np.median(lzz[T], axis=0)/Units.AU)[idx])
    lZZ.append(l_mean)
    # lZZ_v.append(np.mean((np.median(lzz_v[T], axis=0)*m2cm/au2cm)[idx]))
    dlZZ.append(l_std)
    plt.hlines([l_mean], xmin=time[0], xmax=time[-1], linestyles="--", colors=pt[0]._color)
# plt.plot(time, np.mean(Kyy, axis=0)*au2cm**2, label=r"$\kappa_{yy}$")
# plt.plot(time, 0.5*(np.mean(Kyy, axis=0)+np.mean(Kxx, axis=0))*au2cm**2, label=r"$\kappa_{\perp}$")
# plt.plot(time, np.mean(Kxx, axis=0)*au2cm**2, label=r"$\kappa_{xx}$")
plt.legend()
plt.ylabel("$\lambda_{zz}$, [au]")
plt.xlabel("t, [sec]")
plt.xscale('log')
plt.yscale('log')
plt.grid(which='both')
plt.title(r"protons $\vec{r} = (\frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}}, 0) au$")


lXX = []
lYY = []
dlXX = []
dlYY = []
plt.figure()
for T in sorted(lxx.keys()):
    idx = time>1e4
    p = np.sqrt((T + M) ** 2 - M ** 2)
    l = np.median(lxx[T], axis=0)/Units.AU
    # _, time_smooth, l_smooth = misc.smoothing_function(time, l, window=1.2)
    pt = plt.plot(time, l, label=rf"$\lambda_{{xx}}$, P={np.round(p,2)} MV")
    l_mean = np.mean((np.median(lxx[T], axis=0)/Units.AU)[idx])
    l_std = np.std((np.median(lxx[T], axis=0)/Units.AU)[idx])
    lXX.append(l_mean)
    dlXX.append(l_std)
    plt.hlines([l_mean], xmin=time[0], xmax=time[-1], linestyles="--", colors=pt[0]._color)

    l = np.median(lyy[T], axis=0)/Units.AU
    # _, time_smooth, l_smooth = misc.smoothing_function(time, l, window=1.2)
    pt = plt.plot(time, l, label=rf"$\lambda_{{yy}}$, P={np.round(p,2)} MV")
    l_mean = np.mean((np.median(lyy[T], axis=0)/Units.AU)[idx])
    l_std = np.std((np.median(lyy[T], axis=0)/Units.AU)[idx])
    lYY.append(l_mean)
    dlYY.append(l_std)
    plt.hlines([l_mean], xmin=time[0], xmax=time[-1], linestyles="--", colors=pt[0]._color)
# plt.plot(time, np.mean(Kyy, axis=0)*au2cm**2, label=r"$\kappa_{yy}$")
# plt.plot(time, 0.5*(np.mean(Kyy, axis=0)+np.mean(Kxx, axis=0))*au2cm**2, label=r"$\kappa_{\perp}$")
# plt.plot(time, np.mean(Kxx, axis=0)*au2cm**2, label=r"$\kappa_{xx}$")
plt.legend()
plt.ylabel("$\lambda_{\perp}$, [au]")
plt.xlabel("t, [sec]")
plt.xscale('log')
plt.grid(which='both')
plt.yscale('log')
plt.title(r"protons $\vec{r} = (\frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}}, 0) au$")

plt.figure()
plt.errorbar(P, KZZ, yerr=dKZZ, capsize=4, fmt='o', label="$\kappa_{zz}$")
plt.errorbar(P, KXX, yerr=dKXX, capsize=4, fmt='o', label="$\kappa_{xx}$")
plt.errorbar(P, KYY, yerr=dKYY, capsize=4, fmt='o', label="$\kappa_{yy}$")
plt.ylabel("$\kappa$, [$cm^2/sec$]")
plt.xlabel("P, [MV]")
plt.xscale('log')
plt.yscale('log')
plt.grid(which='both')

plt.figure()
plt.errorbar(np.array(P)/1e3, lZZ, yerr=dlZZ, capsize=4, fmt='o', label="$\lambda_{zz}$")
plt.errorbar(np.array(P)/1e3, lXX, yerr=dlXX, capsize=4, fmt='o', label="$\lambda_{xx}$")
plt.errorbar(np.array(P)/1e3, lYY, yerr=dlYY, capsize=4, fmt='o', label="$\lambda_{yy}$")
plt.ylabel("$\lambda$, [au]")
plt.xlabel("P, [GV]")
plt.xscale('log')
plt.yscale('log')
plt.grid(which='both')
plt.legend()

# plt.figure()
# plt.plot(time, np.mean(Kxy, axis=0)*au2cm**2, label=r"$\kappa_{xy}$")
# plt.ylabel("K, [$cm^2/sec$]")
# plt.xlabel("t, [sec]")
# plt.ylabel("K, [$cm^2/sec$]")
# plt.plot(time, np.mean(Kyx, axis=0)*au2cm**2, label=r"$\kappa_{yx}$")
# plt.legend()
# plt.xscale('log')
# plt.title("Antisymmetric part")

# plt.show()
P = np.array(P)
lZZ= np.array(lZZ)
dlZZ = np.array(dlZZ)


# def func(x, a, b):
#     return a*x + b
#
# popt, pcov = curve_fit(func, np.log(P), np.log(lZZ))#, sigma=dlZZ/lZZ)
# def chi_squared(func, y_data, x_data, y_errors, params):
#     expected = np.exp(func(np.log(x_data), *params))
#     residuals = (y_data - expected) / y_errors
#     return np.sum(residuals**2)
#
# chi2_min = chi_squared(func, lZZ, P, dlZZ, popt)
#
# dof = len(P) - len(popt)
# reduced_chi2 = chi2_min / dof
#
# print(popt)
# print(pcov)
#
# print(reduced_chi2)
#
# p = np.linspace(np.min(P), np.max(P), 10000)
# plt.plot(p, np.exp(func(np.log(p), *popt)))


plt.show()

