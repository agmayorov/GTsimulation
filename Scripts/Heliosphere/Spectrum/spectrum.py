import matplotlib.pyplot as plt
import numpy as np
from numba import jit, prange

from scipy.interpolate import interp2d

from MagneticFields.Heliosphere import ParkerUniform
from MagneticFields.Heliosphere.Functions import transformations


@jit(fastmath=True, nopython=True)
def calc_noise(r_param, theta_param, phi_param, a,
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
                Bx_helio, By_helio, Bz_helio = calc_noise(r_param, theta_param, phi_param, a,
                                                                   r, theta, phi,
                                                                   A_rad, alpha_rad, delta_rad,
                                                                   A_azimuth, alpha_azimuth, delta_azimuth,
                                                                   A_2d, alpha_2d, delta_2d,
                                                                   rs, k, dk, use_slab, use_2d,
                                                                   )
                Bx[i, j, l], By[i, j, l], Bz[i, j, l] = T @ np.array([Bx_helio, By_helio, Bz_helio])

    return Bx, By, Bz, x, y, z


if __name__ == "__main__":
    p = ParkerUniform(5, 5, 0, use_reg=False, use_noise=True, use_2d=True, use_slab=False)
    p_reg = ParkerUniform(5, 5, 0, use_reg=True, use_noise=False)
    Bx_reg, By_reg, Bz_reg = p_reg.CalcBfield()
    ez = np.array([Bx_reg, By_reg, Bz_reg])
    ez = ez/np.linalg.norm(ez)
    n = np.array([1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)])
    ex = np.cross(ez, n)
    ex = ex/np.linalg.norm(ex)
    ey = np.cross(ez, ex)

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
        B = np.sqrt(Bx**2 + By**2 + Bz**2)[:, :, i]
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
