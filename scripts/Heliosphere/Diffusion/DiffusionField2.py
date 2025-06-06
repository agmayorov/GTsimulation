import numpy as np
from gtsimulation.MagneticFields.Heliosphere import ParkerUniform
from gtsimulation.Global import Units, Constants
import matplotlib.pyplot as plt

x = 1 / np.sqrt(2)
y = 1 / np.sqrt(2)
z = 0

rs = 0.0232523

l_slab = 2 * 0.8 * 0.04 * (1 / (rs / 5)) ** 0.8 * (rs / 5)  # in au

c = Constants.c
M = 938.2720813

# P = np.sqrt((T + M) ** 2 - M ** 2) * 1e6  # to eV

P = np.array([  43.33063785,  137.35152625,  310.36624948, 1090.07893666,
       1696.03778741, 5863.67810181])
P_arr = np.logspace(np.log10(np.min(P)), np.log10(np.max(P)), 200)
Pev = P_arr * 1e6

T = np.sqrt(P_arr**2 + M**2) - M
beta = np.sqrt(1 - (M/(T+M))**2)
V_norm = beta*c

lZZ = np.array([0.00376044, 0.00794305, 0.00926166, 0.01451331, 0.02049321,
       0.10315032])
dlZZ = np.array([0.00101295, 0.00262258, 0.002737  , 0.00457201, 0.00841759,
       0.0772281 ])

parker_reg = ParkerUniform(x=x, y=y, z=z, use_noise=False)
# Bx, By, Bz = parker_reg.CalcBfield(x, y, z)
B = 6
# B = np.linalg.norm([Bx, By, Bz])
Rl = Pev / (c * B * 1e-9) / Units.AU  # Larmor radius in au

# ez = np.array([Bx, By, Bz])/B
# n = np.array([0, 0, 1])
# ex = np.cross(ez, n)
# ex /= np.linalg.norm(ex)
# ey = np.cross(ez, ex)


# Bx_slab = []
# for _ in range(5000):
#     parker_slab = ParkerUniform(x=x, y=y, z=z, use_reg=False, use_noise=True, use_2d=False, use_hcs=False,
#                                 log_kmax= 6, log_kmin= 1)
#     Bx_helio, By_helio, Bz_helio = parker_slab.CalcBfield(x, y, z)
#     Bx_slab.append(ex @ np.array([Bx_helio, By_helio, Bz_helio]))
#
# Bx_slab = np.array(Bx_slab)
# bx2 = np.mean(Bx_slab**2)
bx2 = 5.81

s = 0.746834*Rl/l_slab
A = (1+s)**(5/6) - 1
q = (5/3*s**2)/(1+s**2 - (1+s**2)**(1/6))

k = 3.1371 * (1 + 7/9 * A/((1/3+q)*(q+7/3)))

l_par = k * B**2/bx2 * Rl**(1/3) * l_slab**(2/3)
# doi:10.1088/0004-637X/810/2/141
l_par_pot1 = (1e22*1e-4 * beta * (1/4.41)  * (((P_arr/1e3)**3.5 + 0.34**3.5)/(1+0.34**3.5))**(1.55/3.5) * 3/V_norm)/Units.AU
l_par_pot2 = (1e22*1e-4 * beta * (1/B)  * (((P_arr/1e3)**3.5 + 0.34**3.5)/(1+0.34**3.5))**(1.55/3.5) * 3/V_norm)/Units.AU
plt.plot(P_arr/1e3, l_par, label="Bieber 1994")
plt.plot(P_arr/1e3, l_par_pot1, label="Potgieter 2015 (with B=4.41nT as reported in the paper)")
plt.plot(P_arr/1e3, l_par_pot2, label="Potgieter 2015 (with B=6nT as used in the simulation)")
plt.errorbar(P/1e3, lZZ, yerr=dlZZ, capsize=4, fmt='o', label="$\lambda_{zz}$")
plt.ylabel("$\lambda_{zz}$, [au]")
plt.xlabel("P, [GV]")
plt.xscale('log')
plt.yscale('log')
plt.grid(which='both')
plt.legend()
plt.show()