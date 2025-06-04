import os
import numpy  as np

paths = [f"../../../tests/ModRegular1.{i}" for i in range(1, 15)]
paths.extend([f"../../../tests/ModRegular2.{i}" for i in range(1, 15)])

T = {}

for path in paths:
    if not os.path.exists(path):
        continue
    files = os.listdir(path)
    for file in files:
        if not file.endswith(".npy"):
            continue
        try:
            particles = np.load(path + os.sep + file, allow_pickle=True)
        except:
            continue
        for event in particles:
            T0 = event['Particle']['T0']
            wout = event['BC']['WOut']

            if T0 not in T:
                T[T0] = [0, 0]

            if wout == 3:
                T[T0][0]+=1
            else:
                T[T0][1]+=1


print(T)