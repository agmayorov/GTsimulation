import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size":15})

from tqdm import tqdm

from MagneticFields.Heliosphere import Parker, ParkerUniform

b = ParkerUniform(x=1/np.sqrt(2), y=1/np.sqrt(2), z=0, use_noise=False)

X, Y, Z = np.meshgrid(np.linspace(1/np.sqrt(2)-0.2, 1/np.sqrt(2)+0.2, 50),
                      np.linspace(1/np.sqrt(2)-0.2, 1/np.sqrt(2)+0.2, 50),
                      np.linspace(-0.2, 0.2, 50))
P_arr = []

for _ in range(2):
    bn = ParkerUniform(x=1/np.sqrt(2), y=1/np.sqrt(2), z=0, use_noise=True, log_kmin=1, log_kmax=6, noise_num=512, use_2d=False, use_reg=False)
    Bx, By, Bz = np.zeros_like(X), np.zeros_like(X), np.zeros_like(X)
    Bx_n, By_n, Bz_n = np.zeros_like(X), np.zeros_like(X), np.zeros_like(X)

    s=10
    s_n=10

    ex = np.array([ 4.32978028e-17,  4.32978028e-17, -1.00000000e+00])
    ey = np.array([1.69955019e-01, 9.85451821e-01, 5.00265775e-17])
    ez = -np.array([ 9.85451821e-01, -1.69955019e-01,  3.53092197e-17])

    for i in tqdm(range(50)):
        for j in range(50):
            for k in range(50):
                x = (X[i,j,k]*ex + Y[i,j,k]*ey + Z[i,j,k]*ez)[0]
                y = (X[i,j,k]*ex + Y[i,j,k]*ey + Z[i,j,k]*ez)[1]
                z = (X[i,j,k]*ex + Y[i,j,k]*ey + Z[i,j,k]*ez)[2]
                Bx[i,j,k], By[i,j,k], Bz[i,j,k] = (np.array(b.CalcBfield(x, y, z))@ex,
                np.array(b.CalcBfield(x, y, z)) @ ey,
                np.array(b.CalcBfield(x, y, z)) @ ez )
                Bx_n[i,j,k], By_n[i,j,k], Bz_n[i,j,k] = (np.array(bn.CalcBfield(x, y, z))@ex,
                np.array(bn.CalcBfield(x, y, z)) @ ey,
                np.array(bn.CalcBfield(x, y, z)) @ ez)


    ax = plt.figure().add_subplot(projection='3d')

    ax.quiver(X[::s, ::s, ::s], Y[::s, ::s, ::s], Z[::s, ::s, ::s],
              Bx[::s, ::s, ::s], By[::s, ::s, ::s], Bz[::s, ::s, ::s],
              length=0.07, normalize=True)
    ax.quiver(X[::s_n, ::s_n, ::s_n], Y[::s_n, ::s_n, ::s_n], Z[::s_n, ::s_n, ::s_n],
              Bx_n[::s_n, ::s_n, ::s_n], By_n[::s_n, ::s_n, ::s_n], Bz_n[::s_n, ::s_n, ::s_n],
              length=0.04, normalize=True, color="r")
    ax.set_xlabel("x [au]")
    ax.set_ylabel("y [au]")
    ax.set_zlabel("z [au]")
    ax.set_title("Slab field")
    plt.show()


    P = np.sum(np.abs(np.fft.rfftn(Bx_n))**2 + np.abs(np.fft.rfftn(By_n))**2 + np.abs(np.fft.rfftn(Bz_n))**2)/np.prod(X.shape)
    P_arr.append(P)
print(P_arr)
# plt.show()

# P_2d = 227374025.38299724
# P_slab = 870689755.5457932