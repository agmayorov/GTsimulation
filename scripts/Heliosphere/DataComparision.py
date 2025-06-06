import os
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from gtsimulation.MagneticFields.Heliosphere import Parker, ParkerUniform

from scipy.optimize import curve_fit, minimize

from misc import moving_average

path = rf"D:\mephi_data\data"
parker_reg = Parker(use_noise=False)
parker_noise = Parker(use_noise=True, use_reg=False, noise_num=1024)



def compare_ace():
    ace_path = path + os.sep + rf"ace_full{os.sep}data_hourly.txt"
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

    Bx, By, Bz = parker_reg.CalcBfield(x, y, z, t=t*365.25*24*3600+2488320)
    Bx_n = np.zeros_like(Bx)
    By_n = np.zeros_like(Bx)
    Bz_n = np.zeros_like(Bx)
    # Bx_n, By_n, Bz_n = parker_noise.CalcBfield(x, y, z, t=t* 365.25 * 24 * 3600 + 2488320)
    for i in range(len(x)):
        parker_uniform = ParkerUniform(use_noise=True, use_reg=False, noise_num=1024, x=x[i], y=y[i], z=z[i], coeff_noise=0.95)
        Bx_n[i], By_n[i], Bz_n[i] = parker_uniform.CalcBfield(x[i], y[i], z[i], t = t[i]*365.25*24*3600+2488320)
    Br = -(Bx+Bx_n)*x - (By+By_n)*y - (Bz+Bz_n)*z

    plt.figure()
    plt.plot(t, Br, label="Regular model + Noise from tangent space")
    plt.plot(t, Br_ace, label="ACE")

    plt.xlabel("t, years")
    plt.ylabel("$B_x$ (gse), nT")
    plt.legend()

    plt.figure()
    plt.plot(t, Br_ace, label="ACE")
    plt.plot(t, Br, label="Regular model + Noise from tangent space")

    plt.xlabel("t, years")
    plt.ylabel("$B_x$ (gse), nT")
    plt.legend()

    plt.figure()
    plt.hist(Bx_n*x + By_n*y+Bz_n, label="model", alpha=0.8, bins=10, range=(-10, 10))
    plt.hist(Br_ace+Bx*x+By*y+Bz*z, label='ace', alpha=0.8, bins=10, range=(-10, 10))
    plt.legend()
    plt.show()

def calibrate_on_ace():
    ace_path = path + os.sep + rf"ace_full{os.sep}data_daily.txt"
    ace_data = np.loadtxt(ace_path)
    ace_data = ace_data[ace_data[:, 3] < 100, :]
    t = (ace_data[:, 0] + (ace_data[:, 1] - 1) / 365.25 + ace_data[:, 2] / (365.25 * 24))
    magnitudes1 = []
    phase1 = []
    phase2 = []
    magnitudes2 = []
    magnitudes3 = []
    t_min = np.min(t)
    t_max = np.max(t)
    delta_t = 0.3
    taus = np.arange(t_min, t_max, delta_t)
    for tau in taus:
        idx = (t < tau+delta_t) * (t > tau)
        t_idx_ = t[idx]
        Br_ace_ = ace_data[idx, 4]

        # t_idx_, Br_ace_ = moving_average(t_idx, (1/365), Br_ace)
        # plt.plot(t_idx, Br_ace)
        # plt.plot(t_idx_, Br_ace_)
        # plt.show()

        june_4 = (31 + 28 + 31 + 30 + 31 + 4) / 365.25
        phi_y = -7.25 * np.pi / 180
        t0 = june_4 + 1 / 4

        x = np.cos(2 * np.pi * (t_idx_ - t0)) * np.cos(phi_y)
        y = np.sin(2 * np.pi * (t_idx_ - t0))
        z = -np.cos(2 * np.pi * (t_idx_ - t0)) * np.sin(phi_y)

        # Bx, By, Bz = parker_reg.CalcBfield(x, y, z, t=t_idx*365.25*24*3600+2488320)
        # Br = -Bx*x - By*y - Bz*z

        def func(t_idx_, magnitude):
            parker_reg = Parker(use_noise=False, magnitude=magnitude)
            Bx, By, Bz = parker_reg.CalcBfield(x, y, z, t=t_idx_ * 365.25 * 24 * 3600 + 2488320)
            Br = -Bx * x - By * y - Bz * z
            return Br

        def error_func1(params):
            magnitude, dt0 = params
            parker_reg = Parker(use_noise=False, magnitude=magnitude)
            dt0 = 28.88
            Bx, By, Bz = parker_reg.CalcBfield(x, y, z, t=(t_idx_ * 365.25 + dt0) * 24*3600)
            Br = -Bx * x - By * y - Bz * z

            return np.mean(np.abs(Br - Br_ace_))
        def error_func2(params):
            magnitude, dt0 = params
            parker_reg = Parker(use_noise=False, magnitude=magnitude)
            dt0 = 28.88
            Bx, By, Bz = parker_reg.CalcBfield(x, y, z, t=(t_idx_ * 365.25 + dt0) * 24*3600)
            Br = -Bx * x - By * y - Bz * z

            return np.mean((Br - Br_ace_)**2) + 2*sp.stats.skew((Br-Br_ace_))**2

        result1 = minimize(error_func1, np.array([2, 28.8]))
        result2 = minimize(error_func2, np.array([2, 28.8]))

        parker_reg = Parker(use_noise=False, magnitude=1)
        Bx, By, Bz = parker_reg.CalcBfield(x, y, z, t=(t_idx_ * 365.25 +28.88)*3600*24)
        Br = -Bx * x - By * y - Bz * z
        m = np.nanmean(Br*Br_ace_)

        m1 = np.abs(result1.x[0])
        m2 = np.abs(result2.x[0])
        m3 = np.abs(m)

        parker_reg = Parker(use_noise=False, magnitude=result1.x[0])
        # Bx, By, Bz = parker_reg.CalcBfield(x, y, z, t=t_idx_ * 365.25 * 24 * 3600 + 2488320)
        Bx, By, Bz = parker_reg.CalcBfield(x, y, z, t=(t_idx_ * 365.25 + 28.88)*24*3600 )
        Br = -Bx * x - By * y - Bz * z
        magnitudes1.append(m1 if np.std(Br-Br_ace_)/m1 <4 else np.nan)

        parker_reg = Parker(use_noise=False, magnitude=result2.x[0])
        # Bx, By, Bz = parker_reg.CalcBfield(x, y, z, t=t_idx_ * 365.25 * 24 * 3600 + 2488320)
        Bx, By, Bz = parker_reg.CalcBfield(x, y, z, t=(t_idx_ * 365.25 + 28.88)*24*3600 )
        Br = -Bx * x - By * y - Bz * z

        magnitudes2.append(m2 if np.std(Br-Br_ace_)/m2 <4 else np.nan)

        parker_reg = Parker(use_noise=False, magnitude=m)
        # Bx, By, Bz = parker_reg.CalcBfield(x, y, z, t=t_idx_ * 365.25 * 24 * 3600 + 2488320)
        Bx, By, Bz = parker_reg.CalcBfield(x, y, z, t=(t_idx_ * 365.25 + 28.88)*24*3600 )
        Br = -Bx * x - By * y - Bz * z

        magnitudes3.append(m3 if np.std(Br-Br_ace_)/m3 <4 else np.nan)

        # plt.figure()
        # plt.plot(t_idx_, Br_ace_, label="ACE")
        # plt.plot(t_idx_, Br, label="Regular parker")
        # # plt.plot(t, Br_opt, label="Regular parker")
        #
        # plt.xlabel("t, years")
        # plt.ylabel("$B_x$ (gse), nT")
        # plt.legend()
        # plt.show()
    # print(magnitudes)
    # plt.plot(taus, np.abs(magnitudes1), label="L1")
    # plt.plot(taus, np.abs(magnitudes2), label="L2")
    # plt.plot(taus, np.abs(magnitudes3), label="Mean")

    plt.scatter(taus, 1/3*(np.array(np.abs(magnitudes3)) + np.array(np.abs(magnitudes2)) + np.array(np.abs(magnitudes1))), label="Average")

    # a0 = 1.209
    # a1, b1 = 0.3978, 0.4558
    # a2, b2 = -0.3019, 0.5982
    # a3, b3 = -0.1757, 0.256
    # a4, b4 = -0.02529, -0.2915
    # a5, b5 = 0.1289, -0.116
    # a6, b6 = 0.2045, 0.02604
    # a7, b7 = 0.1077, -0.1076
    # a8, b8 = 0.193, 0.1602
    # w = 0.2828
    #
    # # Fourier series as a lambda function
    # fourier_series = lambda x: (
    #         a0
    #         + a1 * np.cos(1 * w * x) + b1 * np.sin(1 * w * x)
    #         + a2 * np.cos(2 * w * x) + b2 * np.sin(2 * w * x)
    #         + a3 * np.cos(3 * w * x) + b3 * np.sin(3 * w * x)
    #         + a4 * np.cos(4 * w * x) + b4 * np.sin(4 * w * x)
    #         + a5 * np.cos(5 * w * x) + b5 * np.sin(5 * w * x)
    #         + a6 * np.cos(6 * w * x) + b6 * np.sin(6 * w * x)
    #         + a7 * np.cos(7 * w * x) + b7 * np.sin(7 * w * x)
    #         + a8 * np.cos(8 * w * x) + b8 * np.sin(8 * w * x)
    # )
    a0 = 1.304
    a1, b1 = 0.3011, -0.3554
    a2, b2 = -0.08633, -0.5805
    a3, b3 = -0.1178, 0.04987
    a4, b4 = -0.3004, -0.2365
    w = 0.2568

    # Fourier series as a lambda function
    fourier_series = lambda x: (
            a0
            + a1 * np.cos(1 * w * x) + b1 * np.sin(1 * w * x)
            + a2 * np.cos(2 * w * x) + b2 * np.sin(2 * w * x)
            + a3 * np.cos(3 * w * x) + b3 * np.sin(3 * w * x)
            + a4 * np.cos(4 * w * x) + b4 * np.sin(4 * w * x)
    )

    time = np.linspace(np.min(taus), np.max(taus), 1000)
    plt.plot(time, fourier_series(time), label='Fit')

    plt.xlabel("t, year")
    plt.ylabel("Br, nT")
    plt.legend()
    plt.xticks(np.arange(1998, 2021,2))
    plt.show()

    # plt.plot(taus, phase1, label="L1")
    # # plt.plot(taus, phase2, label="L2")
    # # plt.plot(taus, np.abs(magnitudes3), label="Mean")
    # plt.legend()
    # plt.xticks(np.arange(1998, 2021,2))
    # plt.show()
    #
    # print(np.mean(phase1))

def calibrate_on_ace_speed():
    ace_path = path + os.sep + rf"ace_plasma_full" + os.sep + "data_daily.txt"
    ace_data = np.loadtxt(ace_path)
    ace_data = ace_data[ace_data[:, 3] < 2000, :]
    t = (ace_data[:, 0] + (ace_data[:, 1] - 1) / 365.25 + ace_data[:, 2] / (365.25 * 24))
    V = ace_data[:, 3]

    t, V = moving_average(t, 0.05, V)

    a0 = 414.9
    a1, b1 = 17.28, 17.98
    a2, b2 = -13.54, 24.15
    a3, b3 = 1.775, 7.98
    a4, b4 = 2.891, -9.709
    a5, b5 = -12.02, -18.19
    a6, b6 = 17.1, -12.89
    a7, b7 = 5.636, 1.996
    w = 0.2733

    # Fourier series as a lambda function
    fourier_series = lambda x: (
            a0
            + a1 * np.cos(1 * w * x) + b1 * np.sin(1 * w * x)
            + a2 * np.cos(2 * w * x) + b2 * np.sin(2 * w * x)
            + a3 * np.cos(3 * w * x) + b3 * np.sin(3 * w * x)
            + a4 * np.cos(4 * w * x) + b4 * np.sin(4 * w * x)
            + a5 * np.cos(5 * w * x) + b5 * np.sin(5 * w * x)
            + a6 * np.cos(6 * w * x) + b6 * np.sin(6 * w * x)
            + a7 * np.cos(7 * w * x) + b7 * np.sin(7 * w * x)
    )
    time = np.linspace(np.min(t), np.max(t), 1000)
    plt.plot(t, V, label='Data')
    plt.plot(time, fourier_series(time), label='Fit')
    plt.ylabel("V, km/s")
    plt.xlabel("t, year")
    plt.xticks(np.arange(1998, 2021, 2))
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
    compare_ace()
    # calibrate_on_ace()
    # calibrate_on_ace_speed()
