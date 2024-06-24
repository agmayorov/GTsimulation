import matplotlib.pyplot as plt
import numpy as np

K = np.load("Diffusion200.npy", allow_pickle=True).item(0)
R = []
D = []

for k in K.keys():
    D.append(np.mean(K[k]))
    R.append(k)


plt.scatter(R, D)
plt.show()