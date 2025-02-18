import matplotlib.pyplot as plt
import numpy as np
from numba import jit, prange

from scipy.interpolate import interp2d
from scipy.optimize import curve_fit
from tqdm import tqdm

from MagneticFields.Heliosphere import ParkerUniform, Parker
from MagneticFields.Heliosphere.Functions import transformations
from Scripts.Heliosphere import misc


@jit(fastmath=True, nopython=True)
def calc_noise_uni(r_param, theta_param, phi_param, a,
                   r, theta, phi,
                   A_rad, alpha_rad, delta_rad,
                   A_azimuth, alpha_azimuth, delta_azimuth,
                   A_2d, alpha_2d, delta_2d,
                   rs, k, dk, use_slab, use_2d):
    """
    doi.org/10.3847/1538-4357/aca892/meta
    """
    q_slab = 5 / 3
    q_2d = 8 / 3
    p = 0
    gamma = 3

    cospsi = 1. / np.sqrt(1 + ((r_param - rs) * np.sin(theta_param) / a) ** 2)
    sinpsi = ((r_param - rs) * np.sin(theta_param) / a) / np.sqrt(1 + ((r_param - rs) * np.sin(theta_param) / a) ** 2)

    cospsi_ = 1. / np.sqrt(1 + ((r_param - rs) / a) ** 2)
    sinpsi_ = ((r_param - rs) / a) / np.sqrt(1 + ((r_param - rs) / a) ** 2)

    lam_2d = 0.04 * (r_param / (rs / 5)) ** 0.8 * (rs / 5)
    dlamd_2d = 0.032 * (rs / (5 * r_param)) ** 0.2
    lam_slab = 2 * lam_2d
    dlam_slab = 2 * dlamd_2d

    Br, Btheta, Bphi = 0., 0., 0.

    for mod in prange(len(k)):
        numer_slab = dk[mod, 0] * k[mod, 0] ** p

        # Radial spectrum

        B_rad = A_rad[mod, 0] * r_param ** (-gamma / 2)
        brk_rad = lam_slab * k[mod, 0] / np.sqrt(a * r_param)
        denom_rad = (1 + brk_rad ** (p + q_slab))

        spectrum_rad = np.sqrt(numer_slab / denom_rad)
        deltaB_rad = 2 * B_rad * spectrum_rad * cospsi_ * r_param * np.sqrt(r_param * a)

        # Azimuthal spectrum

        B_azimuth = A_azimuth[mod, 0] * r_param ** (-gamma / 2)
        brk_azimuth = lam_slab * k[mod, 0] / r_param
        denom_azimuth = (1 + brk_azimuth ** (p + q_slab))
        spectrum_azimuth = np.sqrt(numer_slab / denom_azimuth)

        deltaB_azimuth = B_azimuth * spectrum_azimuth
        dspectrum_azimuth = -spectrum_azimuth * (p + q_2d) * (denom_azimuth - 1) * (r_param * dlam_slab - lam_slab) / (
                denom_azimuth * 2 * r_param * lam_slab)
        ddeltaB_azimtuth = B_azimuth * dspectrum_azimuth + spectrum_azimuth * B_azimuth * (-gamma / (2 * r_param))

        # 2d spectrum

        B_2d = A_2d[mod, 0] * r_param ** (-gamma / 2)
        brk_2d = lam_2d * k[mod, 0] / r_param
        denom_2d = (1 + brk_2d ** (p + q_2d))
        numer_2d = dk[mod, 0] * k[mod, 0] ** (p + 1)
        spectrum_2d = np.sqrt(2 * np.pi * numer_2d / denom_2d)
        deltaB_2d = B_2d * spectrum_2d

        dspectrum_2d = -spectrum_2d * (p + q_2d) * (denom_2d - 1) * (r_param * dlamd_2d - lam_2d) / (
                denom_2d * 2 * r_param * lam_2d)
        ddeltaB_2d = B_2d * dspectrum_2d + spectrum_2d * B_2d * (-gamma / (2 * r_param))

        # Radial polarization and phase
        phase_rad = k[mod, 0] * np.sqrt(r / a) + delta_rad[mod, 0]

        # Azimuthal polarization and phase
        phase_azimuth = k[mod, 0] * phi + delta_azimuth[mod, 0]

        # 2d polarization and phase
        phase_2d = k[mod, 0] * ((r / a + phi) * np.sin(alpha_2d[mod, 0]) + theta * np.cos(alpha_2d[mod, 0])) + \
                   delta_2d[mod, 0]

        # Radial field
        Br_rad = 0
        Btheta_rad = -deltaB_rad * a * np.sin(alpha_rad[mod, 0]) * np.cos(phase_rad) / (
                2 * r_param * np.sin(theta_param) * np.sqrt(a * r_param))
        Bphi_rad = deltaB_rad * a * np.cos(alpha_rad[mod, 0]) * np.cos(phase_rad) / (
                2 * r_param * np.sin(theta_param) * np.sqrt(a * r_param))

        # Azimuthal field

        Br_az = -deltaB_azimuth * sinpsi_ * np.cos(alpha_azimuth[mod, 0]) * np.cos(phase_azimuth)
        Btheta_az = deltaB_azimuth * sinpsi_ * np.sin(alpha_azimuth[mod, 0]) * np.cos(phase_azimuth)
        Bphi_az = 1 / k[mod, 0] * (np.sin(theta_param) * np.sin(phase_azimuth) * np.cos(alpha_azimuth[mod, 0]) *
                                   (
                                           2 * deltaB_azimuth * sinpsi_ + r_param / a * deltaB_azimuth * cospsi_ + r_param * sinpsi_ * ddeltaB_azimtuth) -
                                   np.cos(theta_param) * deltaB_azimuth * np.sin(phase_azimuth) * sinpsi_ * np.sin(
                    alpha_azimuth[mod, 0]))

        # 2d field
        Br_2d = -deltaB_2d / (r_param * k[mod, 0]) * (np.sin(phase_2d) * sinpsi * np.tan(theta_param) ** (-1) +
                                                      k[mod, 0] * np.cos(alpha_2d[mod, 0]) * np.cos(phase_2d) * sinpsi +
                                                      np.sin(phase_2d) * sinpsi * cospsi ** 2 * np.tan(theta_param) ** (
                                                          -1))
        Btheta_2d = deltaB_2d / (r_param * np.sin(theta_param)) * cospsi * np.sin(alpha_2d[mod, 0] * np.cos(phase_2d)) \
                    - np.sin(theta_param) * cospsi / (a * r_param * k[mod, 0]) * (
                            ddeltaB_2d * r_param * (r_param - rs) * np.sin(phase_2d) +
                            deltaB_2d * np.sin(phase_2d) * (
                                    2 * r_param - rs - r_param * sinpsi ** 2) +
                            k[mod, 0] * r_param * (r_param - rs) / a * np.sin(
                        alpha_2d[mod, 0] * np.cos(phase_2d) * deltaB_2d))

        Bphi_2d = -deltaB_2d / (r_param * k[mod, 0]) * (
                cospsi * k[mod, 0] * np.cos(alpha_2d[mod, 0]) * np.cos(phase_2d) -
                (np.tan(theta_param)) ** (-1) * np.sin(phase_2d) * cospsi * sinpsi ** 2)

        # Total field
        coeff_slab = 0
        coeff_2d = 0
        if use_slab:
            coeff_slab = 1 / 2

        if use_2d:
            coeff_2d = 1

        Br += coeff_2d * Br_2d + coeff_slab * (Br_az + Br_rad)
        Btheta += coeff_2d * Btheta_2d + coeff_slab * (Btheta_az + Btheta_rad)
        Bphi += coeff_2d * Bphi_2d + coeff_slab * (Bphi_az + Bphi_rad)

    Bx, By, Bz = transformations.Sphere2Cart(Br, Btheta, Bphi, theta, phi)
    return Bx * (r > rs), By * (r > rs), Bz * (r > rs)


@jit(fastmath=True, nopython=True)
def calc_noise(r, theta, phi, a,
               A_rad, alpha_rad, delta_rad,
               A_azimuth, alpha_azimuth, delta_azimuth,
               A_2d, alpha_2d, delta_2d,
               rs, k, dk, use_slab, use_2d):
    q_slab = 5 / 3
    q_2d = 8 / 3
    p = 0
    gamma = 3

    cospsi = 1. / np.sqrt(1 + ((r - rs) * np.sin(theta) / a) ** 2)
    sinpsi = ((r - rs) * np.sin(theta) / a) / np.sqrt(1 + ((r - rs) * np.sin(theta) / a) ** 2)

    cospsi_ = 1. / np.sqrt(1 + ((r - rs) / a) ** 2)
    sinpsi_ = ((r - rs) / a) / np.sqrt(1 + ((r - rs) / a) ** 2)

    lam_2d = 0.04 * (r / (rs / 5)) ** 0.8 * (rs / 5)
    dlamd_2d = 0.032 * (rs / (5 * r)) ** 0.2
    lam_slab = 2 * lam_2d
    dlam_slab = 2 * dlamd_2d

    Br, Btheta, Bphi = 0., 0., 0.

    for mod in prange(len(k)):
        numer_slab = dk[mod, 0] * k[mod, 0] ** p

        # Radial spectrum

        B_rad = A_rad[mod, 0] * r ** (-gamma / 2)
        brk_rad = lam_slab * k[mod, 0] / np.sqrt(a * r)
        denom_rad = (1 + brk_rad ** (p + q_slab))

        spectrum_rad = np.sqrt(numer_slab / denom_rad)
        deltaB_rad = 2 * B_rad * spectrum_rad * cospsi_ * r * np.sqrt(r * a)

        # Azimuthal spectrum

        B_azimuth = A_azimuth[mod, 0] * r ** (-gamma / 2)
        brk_azimuth = lam_slab * k[mod, 0] / r
        denom_azimuth = (1 + brk_azimuth ** (p + q_slab))
        spectrum_azimuth = np.sqrt(numer_slab / denom_azimuth)

        deltaB_azimuth = B_azimuth * spectrum_azimuth
        dspectrum_azimuth = -spectrum_azimuth * (p + q_2d) * (denom_azimuth - 1) * (r * dlam_slab - lam_slab) / (
                denom_azimuth * 2 * r * lam_slab)
        ddeltaB_azimtuth = B_azimuth * dspectrum_azimuth + spectrum_azimuth * B_azimuth * (-gamma / (2 * r))

        # 2d spectrum

        B_2d = A_2d[mod, 0] * r ** (-gamma / 2)
        brk_2d = lam_2d * k[mod, 0] / r
        denom_2d = (1 + brk_2d ** (p + q_2d))
        numer_2d = dk[mod, 0] * k[mod, 0] ** (p + 1)
        spectrum_2d = np.sqrt(2 * np.pi * numer_2d / denom_2d)
        deltaB_2d = B_2d * spectrum_2d

        dspectrum_2d = -spectrum_2d * (p + q_2d) * (denom_2d - 1) * (r * dlamd_2d - lam_2d) / (
                denom_2d * 2 * r * lam_2d)
        ddeltaB_2d = B_2d * dspectrum_2d + spectrum_2d * B_2d * (-gamma / (2 * r))

        # Radial polarization and phase
        phase_rad = k[mod, 0] * np.sqrt(r / a) + delta_rad[mod, 0]

        # Azimuthal polarization and phase
        phase_azimuth = k[mod, 0] * phi + delta_azimuth[mod, 0]

        # 2d polarization and phase
        phase_2d = k[mod, 0] * ((r / a + phi) * np.sin(alpha_2d[mod, 0]) + theta * np.cos(alpha_2d[mod, 0])) + \
                   delta_2d[mod, 0]

        # Radial field
        Br_rad = 0
        Btheta_rad = -deltaB_rad * a * np.sin(alpha_rad[mod, 0]) * np.cos(phase_rad) / (
                2 * r * np.sin(theta) * np.sqrt(a * r))
        Bphi_rad = deltaB_rad * a * np.cos(alpha_rad[mod, 0]) * np.cos(phase_rad) / (
                2 * r * np.sin(theta) * np.sqrt(a * r))

        # Azimuthal field

        Br_az = -deltaB_azimuth * sinpsi_ * np.cos(alpha_azimuth[mod, 0]) * np.cos(phase_azimuth)
        Btheta_az = deltaB_azimuth * sinpsi_ * np.sin(alpha_azimuth[mod, 0]) * np.cos(phase_azimuth)
        Bphi_az = 1 / k[mod, 0] * (np.sin(theta) * np.sin(phase_azimuth) * np.cos(alpha_azimuth[mod, 0]) *
                                   (
                                           2 * deltaB_azimuth * sinpsi_ + r / a * deltaB_azimuth * cospsi_ + r * sinpsi_ * ddeltaB_azimtuth) -
                                   np.cos(theta) * deltaB_azimuth * np.sin(phase_azimuth) * sinpsi_ * np.sin(
                    alpha_azimuth[mod, 0]))

        # 2d field
        Br_2d = -deltaB_2d / (r * k[mod, 0]) * (np.sin(phase_2d) * sinpsi * np.tan(theta) ** (-1) +
                                                k[mod, 0] * np.cos(alpha_2d[mod, 0]) * np.cos(phase_2d) * sinpsi +
                                                np.sin(phase_2d) * sinpsi * cospsi ** 2 * np.tan(theta) ** (-1))
        Btheta_2d = deltaB_2d / (r * np.sin(theta)) * cospsi * np.sin(alpha_2d[mod, 0] * np.cos(phase_2d)) \
                    - np.sin(theta) * cospsi / (a * r * k[mod, 0]) * (ddeltaB_2d * r * (r - rs) * np.sin(phase_2d) +
                                                                      deltaB_2d * np.sin(phase_2d) * (
                                                                              2 * r - rs - r * sinpsi ** 2) +
                                                                      k[mod, 0] * r * (r - rs) / a *
                                                                      np.sin(alpha_2d[mod, 0]) * np.cos(phase_2d) *
                                                                      deltaB_2d)

        Bphi_2d = -deltaB_2d / (r * k[mod, 0]) * (cospsi * k[mod, 0] * np.cos(alpha_2d[mod, 0]) * np.cos(phase_2d) -
                                                  (np.tan(theta)) ** (-1) * np.sin(phase_2d) * cospsi * sinpsi ** 2)

        # Total field
        coeff_slab = 0
        coeff_2d = 0
        if use_slab:
            coeff_slab = 1 / 2

        if use_2d:
            coeff_2d = 1

        Br += coeff_2d * Br_2d + coeff_slab * (Br_az + Br_rad)
        Btheta += coeff_2d * Btheta_2d + coeff_slab * (Btheta_az + Btheta_rad)
        Bphi += coeff_2d * Bphi_2d + coeff_slab * (Bphi_az + Bphi_rad)

    Bx, By, Bz = transformations.Sphere2Cart(Br, Btheta, Bphi, theta, phi)
    return Bx * (r > rs), By * (r > rs), Bz * (r > rs)


@jit(nopython=True, fastmath=True)
def calc_path_2d_slab(A, alpha, delta, x, y, z, k, dk, ls):
    N = len(x)
    Bx, By, Bz = np.zeros(N), np.zeros(N), np.zeros(N)
    for ind in prange(N):
        x_i, y_i, z_i = x[ind], y[ind], z[ind]

        Bx[ind], By[ind], Bz[ind] = gen_2d(x_i, y_i, z_i, k, dk, A, alpha, delta, ls)

    return Bx, By, Bz


@jit(nopython=True, fastmath=True)
def calc_path(a, A_rad, alpha_rad, delta_rad,
              A_azimuth, alpha_azimuth, delta_azimuth,
              A_2d, alpha_2d, delta_2d,
              rs, k, dk, use_slab, use_2d, ex, ey, ez, x, y, z):
    N = len(x)
    Bx, By, Bz = np.zeros(N), np.zeros(N), np.zeros(N)
    for ind in prange(N):
        x_i, y_i, z_i = x[ind], y[ind], z[ind]
        r, _, theta, phi = transformations.Cart2Sphere(x_i, y_i, z_i)
        ex_i, ey_i, ez_i = ex[:, ind], ey[:, ind], ez[:, ind]
        Bx_helio, By_helio, Bz_helio = calc_noise(r, theta, phi, a,
                                                  A_rad, alpha_rad, delta_rad,
                                                  A_azimuth, alpha_azimuth, delta_azimuth,
                                                  A_2d, alpha_2d, delta_2d,
                                                  rs, k, dk, use_slab, use_2d)
        B = np.array([Bx_helio, By_helio, Bz_helio])
        Bx[ind] = B @ ex_i
        By[ind] = B @ ey_i
        Bz[ind] = B @ ez_i

    return Bx, By, Bz


@jit(nopython=True, fastmath=True)
def calc_grid(r_param, theta_param, phi_param, a,
              A_rad, alpha_rad, delta_rad,
              A_azimuth, alpha_azimuth, delta_azimuth,
              A_2d, alpha_2d, delta_2d,
              rs, k, dk, use_slab, use_2d, ls, ex, ey, ez):
    Nz = 20
    Nx = 100
    Ny = 100

    T = np.vstack((ex, ey, ez))

    x = np.linspace(-5, 5, Nx) * ls
    y = np.linspace(-5, 5, Ny) * ls
    z = np.linspace(-5, 5, Nz) * ls
    #
    Bx, By, Bz = np.zeros((Nx, Ny, Nz)), np.zeros((Nx, Ny, Nz)), np.zeros((Nx, Ny, Nz))

    for i in prange(Nx):
        for j in prange(Ny):
            for l in prange(Nz):
                x_helio, y_helio, z_helio = np.linalg.inv(T) @ np.array([x[i], y[j], z[l]])
                x_helio += 1
                y_helio += 1
                r, _, theta, phi = transformations.Cart2Sphere(x_helio, y_helio, z_helio)
                Bx_helio, By_helio, Bz_helio = calc_noise_uni(r_param, theta_param, phi_param, a,
                                                              r, theta, phi,
                                                              A_rad, alpha_rad, delta_rad,
                                                              A_azimuth, alpha_azimuth, delta_azimuth,
                                                              A_2d, alpha_2d, delta_2d,
                                                              rs, k, dk, use_slab, use_2d,
                                                              )
                Bx[i, j, l], By[i, j, l], Bz[i, j, l] = T @ np.array([Bx_helio, By_helio, Bz_helio])

    return Bx, By, Bz, x, y, z


@jit(nopython=True, fastmath=True)
def gen_slab(x, y, z, k, dk, A, alpha, delta, ls):
    Bx, By, Bz = 0, 0, 0
    spec = dk / (1 + (ls * k)**(5/3))
    for mod in prange(len(k)):
        Bx += A[mod, 0] * np.sqrt(spec[mod, 0]) * np.cos(k[mod, 0] * z + delta[mod, 0]) * np.cos(alpha[mod, 0])
        By += A[mod, 0] * np.sqrt(spec[mod, 0]) * np.cos(k[mod, 0] * z + delta[mod, 0]) * np.sin(alpha[mod, 0])
        Bz += 0

    return Bx, By, Bz


@jit(nopython=True, fastmath=True)
def gen_2d(x, y, z, k, dk, A, alpha, delta, ls):
    Bx, By, Bz = 0, 0, 0
    spec = dk * k / (1 + (ls * k)**(8/3))
    for mod in prange(len(k)):
        Bx += A[mod, 0] * np.sqrt(spec[mod, 0]) * np.cos(
            k[mod, 0] * (np.cos(alpha[mod, 0]) * x + np.sin(alpha[mod, 0]) * y) +
            delta[mod, 0]) * np.cos(alpha[mod, 0])
        By += A[mod, 0] * np.sqrt(spec[mod, 0]) * np.cos(
            k[mod, 0] * (np.cos(alpha[mod, 0]) * x + np.sin(alpha[mod, 0]) * y) +
            delta[mod, 0]) * np.sin(alpha[mod, 0])
        Bz += 0

    return Bx, By, Bz


def run_with_grid():
    Bx, By, Bz, x, y, z = calc_grid(p.r, p.theta, p.phi, a,
                                    A_rad, alpha_rad, delta_rad,
                                    A_azimuth, alpha_azimuth, delta_azimuth,
                                    A_2D, alpha_2D, delta_2D,
                                    rs, k, dk, p.use_slab, p.use_2d,
                                    ls, ex, ey, ez)
    Nz = len(z)
    Nx = len(x)
    Ny = len(y)

    for i in range(Nz):
        B = np.sqrt(Bx ** 2 + By ** 2 + Bz ** 2)[:, :, i]
        plt.pcolormesh(B)
        plt.show()

    Bx_fft, By_fft, Bz_fft = np.fft.fft(Bx), np.fft.fft(By), np.fft.fft(Bz)
    kx, ky, kz = np.fft.fftfreq(Nx, x[1] - x[0]), np.fft.fftfreq(Ny, y[1] - y[0]), np.fft.fftfreq(Nz, z[1] - z[0])

    P = np.abs(Bx_fft) ** 2 + np.abs(By_fft) ** 2 + np.abs(Bz_fft) ** 2

    dkx = np.abs(kx[1] - kx[0])
    dky = np.abs(ky[1] - ky[0])
    dkz = np.abs(kz[1] - kz[0])

    Pz = np.sum(np.sum(P, axis=0), axis=0) * dkx * dky
    Px = np.sum(np.sum(P, axis=1), axis=1) * dkz * dky
    Py = np.sum(np.sum(P, axis=0), axis=1) * dkx * dkz
    Pxy = np.sum(P, axis=-1) * dkz

    idxx = np.argsort(kx)
    idxy = np.argsort(ky)

    plt.figure()
    # plt.pcolormesh(kx[idxx], ky[idxy], np.log(Pxy))
    plt.imshow(Pxy)
    plt.colorbar()
    # plt.show()

    plt.figure()
    plt.plot(kx[kx > 0], Px[kx > 0])
    plt.ylabel(r"$\int P(\mathbf{k})dk_ydk_z$")
    plt.xlabel("$k_x, au^{-1}$")
    plt.xscale('log')
    plt.yscale('log')
    # plt.show()

    plt.figure()
    plt.plot(ky[ky > 0], Py[ky > 0])
    plt.ylabel(r"$\int P(\mathbf{k})dk_xdk_z$")
    plt.xlabel("$k_y, au^{-1}$")
    plt.xscale('log')
    plt.yscale('log')
    # plt.show()

    plt.figure()
    plt.plot(kz[kz > 0], Pz[kz > 0])
    plt.ylabel(r"$\int P(\mathbf{k})dk_xdk_y$")
    plt.xlabel("$k_z, au^{-1}$")
    plt.xscale('log')
    plt.yscale('log')
    plt.show()


def run_with_path_helio():
    global ls, ez, ex, ey
    l = np.linspace(0, 0.25, 10000)
    dl = l[1] - l[0]
    phi = np.zeros_like(l) + 0
    theta = np.pi / 2 + 0 * phi
    r = np.zeros_like(phi)
    r = 1 + 0 * phi
    # r[0] = 1/(1 - omega*np.sin(theta[0])/v * phi[0])
    # r[0] = 50
    for ind in range(1, len(l)):
        cospsi = 1/np.sqrt(1 + ((r[ind-1])*omega*np.sin(theta[ind-1])/v)**2)
        sinpsi = ((r[ind-1])*omega*np.sin(theta[ind-1])/v)/np.sqrt(1 + ((r[ind-1])*omega*np.sin(theta[ind-1])/v)**2)


        dphi = -sinpsi/(r[ind-1]*np.sin(theta[ind-1]))*dl
        dr = cospsi*dl
        #
        # dphi = cospsi/(r[ind-1]*np.sin(theta[ind-1]))*dl
        # dr = sinpsi*dl

        r[ind] = r[ind-1]+dr
        phi[ind] = phi[ind-1]+dphi
        # dphi = -np.abs(dl/np.sqrt((v/omega)**2*(1+np.sin(theta[ind-1])**2 * phi[ind-1]**2)))
        # phi[ind] = phi[ind-1] + dphi
        # dtheta = dl / r[ind - 1]
        # theta[ind] = theta[ind - 1] + dtheta
        # dphi = dl / np.sqrt(r[ind-1]**2 * np.sin(theta[ind-1])**2 * (1 + omega**2/v**2))
        # phi[ind] = phi[ind-1] - dphi
        # r[ind] = 1 / (1 - omega * np.sin(theta[ind]) / v * phi[ind])
    #
    # r = -v / omega * phi
    # plt.polar(phi, r)
    # plt.show()
    rs = p.rs
    ls = 0.08 * (r / (rs / 5)) ** 0.8 * (rs / 5)
    z = r * np.cos(theta)
    x = r * np.cos(phi) * np.sin(theta)
    y = r * np.sin(phi) * np.sin(theta)
    i = np.array([[1], [0], [0]])
    j = np.array([[0], [1], [0]])
    kk = np.array([[0], [0], [1]])
    e_r = np.sin(theta) * np.cos(phi) * i + np.sin(theta) * np.sin(phi) * j + np.cos(theta) * kk
    e_phi = -np.sin(phi) * i + np.cos(phi) * j
    e_theta = np.cos(theta) * np.cos(phi) * i + np.cos(theta) * np.sin(phi) * j - np.sin(theta) * kk
    cos_qsi = 1 / np.sqrt(1 + (omega * (r - p.rs) * np.sin(theta) / v) ** 2)
    sin_qsi = omega * (r - p.rs) * np.sin(theta) / v / np.sqrt(1 + (omega * (r - p.rs) * np.sin(theta) / v) ** 2)
    ez = cos_qsi * e_r - sin_qsi * e_phi
    ex = e_theta
    ey = sin_qsi * e_r + cos_qsi * e_phi
    Bx, By, Bz = calc_path(a, A_rad, alpha_rad, delta_rad,
                           A_azimuth, alpha_azimuth, delta_azimuth,
                           A_2D, alpha_2D, delta_2D,
                           rs, k, dk, p.use_slab, p.use_2d,
                           ex, ey, ez, x, y, z)
    # ax = plt.figure().add_subplot(projection='3d')
    ax = plt.figure().add_subplot(projection='polar')
    # plt.title("Trajectory")
    # plt.plot(x, y, z)
    plt.polar(phi, r, label='The path')
    # plt.plot(0, 0, 0, "*")
    plt.plot(0, 0, "*", label='The Sun')
    plt.legend()
    # ax.set_xlabel("x, au")
    # ax.set_ylabel("y, au")
    # ax.set_zlabel("z, au")

    plt.figure()
    plt.title("Field (slab)")
    plt.plot(l, Bx, label=r"$B_{\theta}$")
    plt.plot(l, By, label="$B_{\perp}$")
    plt.plot(l, Bz, label="$B_{\parallel}$")
    plt.xlabel("Path, au")
    plt.ylabel("Magnetic field, nT")
    plt.legend()

    P = np.abs(np.fft.fft(Bx)) ** 2 + np.abs(np.fft.fft(By)) ** 2 + np.abs(np.fft.fft(Bz)) ** 2
    wave_number = np.fft.fftfreq(len(l), dl)
    idx = wave_number > 0
    wave_number = wave_number[idx]
    P = P[idx]
    _, wave_number_mean, P_mean = misc.smoothing_function(wave_number, P, window=2)

    s, e = 3e1, 1e3

    fitx = np.log(wave_number_mean[(wave_number_mean>s)*(wave_number_mean<e)])
    fity = np.log(P_mean[(wave_number_mean>s)*(wave_number_mean<e)])
    def func(param, a, b):
        return a*param+b
    param, cov = curve_fit(func, fitx, fity)

    plt.figure()
    plt.title("Power spectrum")
    plt.plot(wave_number, P)
    plt.plot(wave_number_mean, P_mean)
    plt.plot(np.linspace(s, e), np.exp(func(np.log(np.linspace(s, e)), *param)), linestyle="--", linewidth=2, color='k',
             label=fr"$\propto k^{{{np.round(param[0], 2)}\pm{np.round(np.sqrt(cov[0][0]), 2)}}}$")

    # s, e = 1e3, 1e4
    #
    # fitx = np.log(wave_number_mean[(wave_number_mean>s)*(wave_number_mean<e)])
    # fity = np.log(P_mean[(wave_number_mean>s)*(wave_number_mean<e)])
    # param, cov = curve_fit(func, fitx, fity)
    # plt.plot(np.linspace(s, e), np.exp(func(np.log(np.linspace(s, e)), *param)), linestyle="--", linewidth=2, color='brown',
    #          label=fr"$\propto k^{{{np.round(param[0], 2)}\pm{np.round(np.sqrt(cov[0][0]), 2)}}}$")

    plt.yscale("log")
    plt.xscale("log")
    plt.ylabel(r"$P, nT^2 au^2$")
    plt.xlabel("$k, au^{-1}$")
    plt.legend()
    plt.show()


def run_with_path_uni():
    global ls
    ls = 0.08 * (1 / (p.rs / 5)) ** 0.8 * (p.rs / 5)
    y = np.linspace(-20*ls, 20*ls, 100000) + 1/np.sqrt(2)
    x = np.zeros_like(y) + 1/np.sqrt(2)
    z = np.zeros_like(y) + 0*1/np.sqrt(2)
    l = y
    dl = l[1] - l[0]
    Bx, By, Bz = calc_path_2d_slab(A_2D, alpha_2D, delta_2D, x, y, z, k, dk, ls)
    # plt.title("2d field")
    # plt.plot(l, Bx, label="Bx")
    # plt.plot(l, By, label="By")
    # plt.plot(l, Bz, label="Bz")
    # plt.xlabel("x', au")
    # plt.ylabel("Magnetic field")
    # plt.legend()
    # plt.show()
    P = np.abs(np.fft.fft(Bx)) ** 2 + np.abs(np.fft.fft(By)) ** 2 + np.abs(np.fft.fft(Bz)) ** 2
    wave_number = np.fft.fftfreq(len(l), dl)
    idx = wave_number > 0
    wave_number = wave_number[idx]
    P = P[idx]
    _, wave_number_mean, P_mean = misc.smoothing_function(wave_number, P, window=2)

    s, e = 2e2, 1e4

    fitx = np.log(wave_number_mean[(wave_number_mean > s) * (wave_number_mean < e)])
    fity = np.log(P_mean[(wave_number_mean > s) * (wave_number_mean < e)])

    def func(param, a, b):
        return a * param + b

    param, cov = curve_fit(func, fitx, fity)

    # plt.figure()
    # plt.title("Power spectrum")
    # plt.plot(wave_number, P)
    # plt.plot(wave_number_mean, P_mean)
    # plt.plot(np.linspace(s, e), np.exp(func(np.log(np.linspace(s, e)), *param)), linestyle="--", linewidth=2, color='k',
    #          label=fr"$\propto k^{{{np.round(param[0], 2)}\pm{np.round(np.sqrt(cov[0][0]), 2)}}}$")
    #
    # plt.yscale("log")
    # plt.xscale("log")
    # plt.ylabel(r"$P, nT^2 au^2$")
    # plt.xlabel("$k, au^{-1}$")
    # plt.legend()
    # plt.title("Power spectrum")
    # plt.show()


    return np.round(param[0], 2)

if __name__ == "__main__":
    np.random.seed()
    # gamma = []
    # for _ in tqdm(range(30)):
    p = ParkerUniform(1/np.sqrt(2), 1/np.sqrt(2), 0, use_reg=False, use_noise=True, noise_num=1024)
    p_slab = ParkerUniform(1/np.sqrt(2), 1/np.sqrt(2), 0, use_reg=False, use_noise=True, use_2d=False, use_slab=True, noise_num=1024)
    p_2d = ParkerUniform(1/np.sqrt(2), 1/np.sqrt(2), 0, use_reg=False, use_noise=True, use_2d=True, use_slab=False, noise_num=1024)
    p_reg = ParkerUniform(1/np.sqrt(2), 1/np.sqrt(2), 0, use_reg=True, use_noise=False)
    parker_reg = Parker(use_noise=False)
    Bx_reg, By_reg, Bz_reg = p_reg.CalcBfield()
    ez = np.array([Bx_reg, By_reg, Bz_reg])
    ez = ez/np.linalg.norm(ez)
    n = np.array([1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)])
    ex = np.cross(ez, n)
    ex = ex/np.linalg.norm(ex)
    ey = np.cross(ez, ex)

    l = np.linspace(0, 1, 10000)
    dl = l[1] - l[0]
    k = np.fft.rfftfreq(len(l), dl)
    dk = k[1] - k[0]

    dr = l[np.newaxis].T*ez
    x = dr[:, 0]
    y = dr[:, 1]
    z = dr[:, 2]
    Bx, By, Bz = [], [], []
    for (xx, yy, zz) in zip(x, y, z):
        bx, by, bz = p_slab.CalcBfield(xx, yy, zz)
        Bx.append(bx)
        By.append(by)
        Bz.append(bz)

    P_slab = np.abs(np.fft.rfftn(Bx)) ** 2 + np.abs(np.fft.rfftn(By)) ** 2 + np.abs(np.fft.rfftn(Bz)) ** 2
    E_slab = np.sum(P_slab*dk)
    print("Slab:", E_slab)

    l = np.linspace(0, 0.1, 1000)
    dl = l[1] - l[0]
    k = np.fft.rfftfreq(len(l), dl)
    dk = k[1] - k[0]

    dr1 = l[np.newaxis].T * ex
    x1 = dr1[:, 0]
    y1 = dr1[:, 1]
    z1 = dr1[:, 2]

    dr2 = l[np.newaxis].T * ey
    x2 = dr2[:, 0]
    y2 = dr2[:, 1]
    z2 = dr2[:, 2]
    Bx, By, Bz = [], [], []
    for (xx1, yy1, zz1) in zip(x1, y1, z1):
        Bx1, By1, Bz1 =[], [], []
        for (xx2, yy2, zz2) in zip(x2, y2, z2):
            bx, by, bz = p_2d.CalcBfield(xx1 + xx2, yy1 + yy2, zz1 + zz2)
            Bx1.append(bx)
            By1.append(by)
            Bz1.append(bz)
        Bx.append(Bx1)
        By.append(By1)
        Bz.append(Bz1)
    Bx = np.array(Bx)
    By = np.array(By)
    Bz = np.array(Bz)

    P_2d = np.abs(np.fft.rfftn(Bx)) ** 2 + np.abs(np.fft.rfftn(By)) ** 2 + np.abs(np.fft.rfftn(Bz)) ** 2
    E_2d = np.sum(P_2d * dk*dk)
    print("2D:", E_2d)

    print("2d/slab:", E_2d/E_slab)

    ls = 0.04 * (1 / (p.rs / 5)) ** 0.8 * (p.rs / 5)

    omega = p.omega
    rs = p.rs
    v_wind = p.wind
    a = v_wind / omega

    A_rad = p.A_rad
    alpha_rad = p.alpha_rad
    delta_rad = p.delta_rad

    A_azimuth = p.A_azimuth
    alpha_azimuth = p.alpha_azimuth
    delta_azimuth = p.delta_azimuth

    A_2D = p.A_2D
    alpha_2D = p.alpha_2D
    delta_2D = p.delta_2D

    k = p.k
    dk = p.dk
    v = p.v_wind(np.pi / 2, p.km2AU)

    #     gamma.append(run_with_path_uni())
    #
    # print(np.mean(gamma))
    # print(np.std(gamma))

    # run_with_path_helio()

    # run_with_grid()
