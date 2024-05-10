import matplotlib.pyplot as plt
import numpy as np


def PlotTracks(Track, Units):
    for track in Track:
        for event in track["Coordinates"]:
            X, Y, Z = event[:, 0], event[:, 1], event[:, 2]
            ax = plt.figure().add_subplot(projection='3d')
            ax.plot(X, Y, Z, label="Trajectory", color='black', linewidth=0.5)
            ax.scatter(X[0], Y[0], Z[0], label='Starting point', color='red')
            ax.scatter(X[-1], Y[-1], Z[-1], label="End point", color='blue')

            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            x = np.outer(np.cos(u), np.sin(v))
            y = np.outer(np.sin(u), np.sin(v))
            z = np.outer(np.ones(np.size(u)), np.cos(v))

            ax.plot_surface(x, y, z, color='g')

            ax.set_xlabel(f"X [{Units}]")
            ax.set_ylabel(f"Y [{Units}]")
            ax.set_zlabel(f"Z [{Units}]")

            ax.legend()

            plt.show()
