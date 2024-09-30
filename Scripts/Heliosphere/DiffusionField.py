import datetime

import numpy as np
import scipy.special as sc
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.optimize import fsolve

from MagneticFields.Heliosphere import Parker
from Global.consts import Units, Constants


date = datetime.datetime(2008, 1, 1)

M = 938.2720813
c = Constants.c
e = Constants.e

p_reg = Parker(date=date, use_reg=True, use_noise=False)

r0 = 0.0232523/5
T = 200
v = c * np.sqrt(1 - M/(T+M))

D_Par = []
D_Perp = []

for r in tqdm(np.logspace(-1, 2, 30)):
    Bx, By, Bz = np.array(p_reg.CalcBfield(r*1/np.sqrt(2), r*1/np.sqrt(2), 0)) * 1e-9
    B = np.sqrt(Bx**2 + By ** 2 + Bz**2)

    ez = np.array([Bx, By, Bz])/B
    n = np.array([0, 1/np.sqrt(2), 1/np.sqrt(2)])

    ex = np.cross(n, ez) / np.linalg.norm(np.cross(n, ez))
    ey = np.cross(ez, ex)
    P = np.sqrt((T+M)**2 - M**2) * 1e6
    Rl = P/(B*c)

    l_slab = (lambda r: (0.8 * 0.08 * (r / r0) ** 0.8 * r0)*Units.AU2m)(r)
    l_2 = (lambda r: (0.8 * 0.04 * (r / r0) ** 0.8 * r0)*Units.AU2m)(r)
    l_0 = 12.5 * l_2

    B_slab = []
    B_Z = []
    B_2d = []
    B_n = []

    for _ in range(1000):
        p_slab = Parker(date=date, use_reg=False, use_noise=True, use_2d=False)
        Bx_slab, By_slab, Bz_slab = np.array(p_slab.CalcBfield(r*1/np.sqrt(2), r*1/np.sqrt(2), 0))

        Bx_slab_ = np.array([Bx_slab, By_slab, Bz_slab])@ex
        By_slab_ = np.array([Bx_slab, By_slab, Bz_slab])@ey
        Bz_slab_ = np.array([Bx_slab, By_slab, Bz_slab])@ez

        B_slab.append(By_slab_**2 + Bx_slab_**2 + Bz_slab_**2)

        p_2d = Parker(date=date, use_reg=False, use_noise=True, use_slab=False)
        Bx_2d, By_2d, Bz_2d = np.array(p_2d.CalcBfield(r * 1 / np.sqrt(2), r * 1 / np.sqrt(2), 0))
        Bx_2d_ = np.array([Bx_2d, By_2d, Bz_2d])@ex
        By_2d_ = np.array([Bx_2d, By_2d, Bz_2d])@ey
        Bz_2d_ = np.array([Bx_2d, By_2d, Bz_2d])@ez

        B_2d.append(By_2d**2 + Bx_2d**2 + Bz_2d**2)
        B_Z.append(Bz_2d_)

        pn = Parker(date=date, use_reg=False, use_noise=True, use_slab=True, use_2d=True)
        Bx_n, By_n, Bz_n = np.array(pn.CalcBfield(r * 1 / np.sqrt(2), r * 1 / np.sqrt(2), 0))
        B_n.append(By_n**2 + Bx_n**2 + Bz_n**2)

    bs2 = np.mean(B_slab)*1e-18
    b2d2 = np.mean(B_2d)*1e-18
    bz2 = np.var(B_Z)*1e-18
    bn2 = np.mean(B_n)*1e-18
    b22 = bs2/0.2497

    s = 0.746834 * (Rl/l_slab)
    A = (1+s**2)**(5/6) - 1
    q = (5*s**2/3)/(1+s**2 - (1+s**2)**(1/6))

    k = 3.1371 * (1 + (7/9 * A)/((1/3 + q)*(q+7/3)))

    l_par = k * B**(5/3)/bs2 * (P/c)**(1/3) * l_slab**(2/3)

    D_par = l_par * v/3
    D_Par.append(D_par)
    a = np.sqrt(3)
    Cs = 1 / 5
    C2 = (5 / 3 - 1) * (5 / 3 + 1) / np.pi
    C0 = (np.log10(l_0/l_2) + 1/4 + 1/(5/3-1))**(-1)

    xx = np.sqrt(3)*B/(a*l_par*np.sqrt(bn2))

    #l_perp = np.sqrt(3*np.pi)*a*C0*b2d2/(B*np.sqrt(bn2)) * (a*np.sqrt(bn2)*l_par/(np.sqrt(3*np.pi)*B)*(
     #   np.exp(-xx**2 * l_2**2) - np.exp(-xx**2 * l_0**2)) + l_0 * sc.erfc(xx*l_0) - l_2*sc.erfc(xx*l_2))

    l_perp = (a**2 * np.sqrt(3*np.pi) * (5/3-1)/(5/6) * sc.gamma(5/6)/sc.gamma(5/6-1/2) * l_2 * b2d2/B**2)**(2/3) * l_par**(1/3)

    D_perp = l_perp * v / 3
    D_Perp.append(D_perp)


plt.figure()
plt.plot(np.logspace(-1, 2, 30), D_Par, label="$\kappa_{\parallel}$")
plt.plot(np.logspace(-1, 2, 30), D_Perp, label="$\kappa_{\perp}$")
plt.yscale("log")
plt.xscale("log")
plt.xlabel("R [au]")
plt.ylabel(r"$\kappa$, [$\frac{m^2}{sec}$]")
plt.title("200 MeV proton")
plt.legend()
plt.show()



