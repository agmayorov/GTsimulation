import os
import numpy as np
import matplotlib.pyplot as plt
from MagneticFields.Heliosphere import Parker

path = rf"..{os.sep}..{os.sep}..{os.sep}data"
parker_reg = Parker(use_noise=False)
parker_noise = Parker(use_noise=True, use_reg=True, noise_num=1024)


def compare_ace():
    ace_path = path + os.sep + rf"ace_full{os.sep}data.txt"
    ace_data = np.loadtxt(ace_path)
    ace_data = ace_data[ace_data[:, 3] < 100, :]
    t = (ace_data[:, 0] + (ace_data[:, 1] - 1) / 365.25 + ace_data[:, 2] / (365.25 * 24))
    idx = (t < 2009) * (t > 2007.8)
    t = t[idx]
    Br_ace = ace_data[idx, 4]

    june_4 = (31 + 28 + 31 + 30 + 31 + 4) / 365.25
    phi_y = -7.25 * np.pi / 180
    t0 = june_4 + 1 / 4

    x = np.cos(2 * np.pi * (t - t0)) * np.cos(phi_y)
    y = np.sin(2 * np.pi * (t - t0))
    z = -np.cos(2 * np.pi * (t - t0)) * np.sin(phi_y)

    Bx, By, Bz = parker_noise.CalcBfield(x, y, z, t=t*365.25*24*3600+2488320)
    Br = -Bx*x - By*y - Bz*z

    plt.figure()
    plt.plot(t, Br_ace, label="ACE")
    plt.plot(t, Br, label="Regular parker")

    plt.xlabel("t, years")
    plt.ylabel("$B_x$ (gse), nT")
    plt.legend()
    plt.show()


def compare_ulysses():
    ulysses_path = path + os.sep + rf"ulysses{os.sep}data.txt"
    ulysses_data = np.loadtxt(ulysses_path)
    ulysses_data = ulysses_data[(ulysses_data[:, 0]<2010)*(ulysses_data[:, 0]>2007), :]
    ulysses_data = ulysses_data[(ulysses_data[:, 9]<100)]

    t = (ulysses_data[:, 0] + (ulysses_data[:, 1] - 1) / 365.25 + ulysses_data[:, 2] / (365.25 * 24))

    R = ulysses_data[:, 3]
    lat = ulysses_data[:, 4] * np.pi/180
    long = ulysses_data[:, 5] * np.pi/180

    x = R*np.cos(lat)*np.cos(long)
    y = R*np.cos(lat)*np.sin(long)
    z = R*np.sin(lat)

    BR = ulysses_data[:, 6]

    plt.figure().add_subplot(projection='3d')
    plt.plot(x, y, z)

    plt.figure()
    Bx, By, Bz = parker_noise.CalcBfield(x, y, z, t=t*365.25*24*3600+2488320)
    Br = Bx*x/R + By*y/R + Bz*z/R
    plt.plot(t, BR, label="Ulysses observations")
    plt.plot(t, Br, label="Parker (with turb)")

    plt.xlabel("t, years")
    plt.ylabel("$B_r$, nT")
    plt.legend()

    plt.show()

if __name__ == "__main__":
    compare_ulysses()
