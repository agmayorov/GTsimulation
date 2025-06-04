import numpy as np
import scipy.io as si

file = "../test_0"

events = np.load(f"{file}.npy", allow_pickle=True)
mevents = []
for event in events:
    event["WOut"] += 1
    coords = event["Track"].pop("Coordinates")
    vels = event["Track"].pop("Velocities")
    event["Track"]["X"] = coords[:, 0]
    event["Track"]["Y"] = coords[:, 1]
    event["Track"]["Z"] = coords[:, 2]
    event["Track"]["VX"] = vels[:, 0]
    event["Track"]["VY"] = vels[:, 1]
    event["Track"]["VZ"] = vels[:, 2]
    mevents.append({"data": event})


si.savemat(f"{file}.mat", {"GTtrack": mevents})


