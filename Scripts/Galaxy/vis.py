import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from MagneticFields.Galaxy import JF12mod

field = JF12mod(use_noise=False)


ax_const = 'X'
ax_pass = 'YZ'

L = 20
n = 801

base_1 = np.linspace(-L, L, n)
base_2 = np.linspace(-L, L, n)
base_const = np.zeros(n)

X, Y = np.meshgrid(base_1, base_2)
Z = np.meshgrid(base_const, base_const)[0]


Bx_tot = np.zeros((n, n))
By_tot = np.zeros((n, n))
Bz_tot = np.zeros((n, n))

B_tot = np.zeros((n, n))
B_sign = np.ones((n, n))


for i1 in range(n):
    for i2 in range(n):
        Bx_tot[i1, i2], By_tot[i1, i2], Bz_tot[i1, i2] = field.CalcBfield(X[i1, i2], Y[i1, i2], Z[i1, i2])
        B_sign[i1,i2] = np.sign(np.dot([-Y[i1,i2], X[i1,i2]], [Bx_tot[i1,i2], By_tot[i1,i2]]))

B_tot = B_sign * np.sqrt(Bx_tot**2 + By_tot**2 + Bz_tot**2)
B_tot/=np.max(Bx_tot)

lim = 1

fig, ax = plt.subplots()
blue_white_red = LinearSegmentedColormap.from_list('blue_white_red', ['blue', 'white', 'red'])

s = ax.pcolormesh(base_1, base_2, B_tot, cmap=blue_white_red, shading='auto')
ax.set_xlim([-25, 25])
ax.set_ylim([-25, 25])
ax.set_xlabel(f'{ax_pass[0]}, kpc')
ax.set_ylabel(f'{ax_pass[1]}, kpc')
s.set_clim(-lim, lim)

c = fig.colorbar(s, ax=ax)
c.set_label('B, nT', fontsize=12)

ax.set_aspect('equal')


plt.show()
