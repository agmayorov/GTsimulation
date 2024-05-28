import os
import numpy as np
import matplotlib.pyplot as plt

path = f"..{os.sep}ParkerPR2"

files = os.listdir(path)
prs = []

angles = []

T_start = []
T_mod = []

for i in range(len(files)):
    try:
        file = np.load(path + os.sep + files[i], allow_pickle=True)
        for j in range(len(file)):
            T_start.append(file[j]["Particle"]["T0"] / 1000)
            if file[j]["WOut"] != 8:
                T_mod.append(file[j]["Particle"]["T0"] / 1000)
                prs.append((i, j))
                angles.extend(file[j]["Track"]["Angles"] * 180 / np.pi)
    except:
        continue
print(prs)

print(len(prs) / (100 * len(files)) * 100)


def draw_hist(arr, label="Label", bins=50, area=None, **kwargs):
    arr = np.array(arr)
    # arr = arr[arr<7.5]
    values, bins = np.histogram(arr, bins=bins)
    if area is None:
        area = sum(np.diff(bins) * values)
    centers = 0.5 * (bins[1:] + bins[:-1])
    plt.bar(centers, values / area / np.diff(bins), width=np.diff(bins), label=label, edgecolor="black",
            yerr=values ** 0.5 / area / np.diff(bins),
            capsize=4, **kwargs)

bins = np.logspace(np.log10(np.min(T_start)), np.log10(np.max(T_start)), 20, endpoint=True)
draw_hist(T_start, label="Initial distribution", bins=bins, ecolor="black", area=1)
draw_hist(T_mod, label="Modulated distribution", alpha=0.5, bins=bins, ecolor="red", area=4/9)
# plt.hlines(0.05, xmin=0, xmax=20, linestyles="--", linewidth=2, colors='#BB00BB')
plt.legend()
plt.xlabel("T [GeV]")
plt.ylabel("Spectrum [num/GeV]")
plt.xscale("log")
plt.show()
#
# for i, j in prs:
#     event = np.load(path + os.sep + files[i], allow_pickle=True)[j]
#     R = event["Track"]["Coordinates"]
#     T = event["Particle"]["T0"] / 1000
#     X, Y, Z = R[:, 0], R[:, 1], R[:, 2]
#
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#
#     ax.set_xlabel(f"X [AU]")
#     ax.set_ylabel(f"Y [AU]")
#     ax.set_zlabel(f"Z [AU]")
#
#     ax.plot(X, Y, Z, label="Trajectory", color='black', linewidth=1)
#     ax.scatter(X[0], Y[0], Z[0], label='Starting point', color='red')
#     ax.scatter(X[-1], Y[-1], Z[-1], label="End point", color='blue')
#
#     lons = np.linspace(-180, 180, 30) * np.pi / 180
#     lats = np.linspace(-90, 90, 30)[::-1] * np.pi / 180
#
#     x = 20 * np.outer(np.cos(lons), np.cos(lats)).T
#     y = 20 * np.outer(np.sin(lons), np.cos(lats)).T
#     z = 20 * np.outer(np.ones(np.size(lons)), np.sin(lats)).T
#
#     ax.scatter(x, y, z, color='#00ffff', s=1, label="20 AU surface")
#     ax.axis("equal")
#     ax.set_title(f"Kin energy = {T} GeV")
#
#     ax.legend()
#     plt.show()
#     angles = event["Track"]["Angles"] * 180 / np.pi
#
#     plt.hist(angles, bins=100, edgecolor="black")
#     plt.show()
