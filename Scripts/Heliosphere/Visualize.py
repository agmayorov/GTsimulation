import numpy as np
import matplotlib.pyplot as plt

from MagneticFields.Heliosphere import Parker

b = Parker(use_noise=False)
bn = Parker(use_noise=True, log_kmin=1, log_kmax=6, noise_num=512)

X, Y, Z = np.meshgrid(np.linspace(1/np.sqrt(2)-0.2, 1/np.sqrt(2)+0.2, 25),
                      np.linspace(1/np.sqrt(2)-0.2, 1/np.sqrt(2)+0.2, 25),
                      np.linspace(-0.2, 0.2, 25))

Bx, By, Bz = b.CalcBfield(X, Y, Z)

Bx_n, By_n, Bz_n = np.zeros_like(X), np.zeros_like(X), np.zeros_like(X)

s=5
s_n=2

for i in range(25):
    for j in range(25):
        for k in range(25):
            Bx_n[i,j,k], By_n[i,j,k], Bz_n[i,j,k] = bn.CalcBfield(X[i,j,k], Y[i,j,k], Z[i,j,k])

ax = plt.figure().add_subplot(projection='3d')

ax.quiver(X[::s, ::s, ::s], Y[::s, ::s, ::s], Z[::s, ::s, ::s], Bx[::s, ::s, ::s], By[::s, ::s, ::s], Bz[::s, ::s, ::s], length=0.05, normalize=True)
ax.quiver(X[::s_n, ::s_n, ::s_n], Y[::s_n, ::s_n, ::s_n], Z[::s_n, ::s_n, ::s_n], Bx_n[::s_n, ::s_n, ::s_n], By_n[::s_n, ::s_n, ::s_n], Bz_n[::s_n, ::s_n, ::s_n], length=0.02, normalize=True,
          color="r")

plt.show()

plt.hist(Bx_n - Bx)
plt.show()
plt.hist(By_n - By)
plt.show()
plt.hist(Bz_n - Bz)
plt.show()
