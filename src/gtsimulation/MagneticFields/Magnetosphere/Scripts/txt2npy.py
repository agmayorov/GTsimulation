import numpy as np

name = "../CHAOS-8.1/CHAOS-8.1_core"

with open(name + ".shc.txt", 'r') as f:
    f.readline()
    f.readline()
    f.readline()
    year = np.array(list(map(float, f.readline().split())))
    gh = []
    for line in f.readlines():
        gh.append(list(map(float, line.split()))[2:])

    gh = np.array(gh)

    save_dict = {'years': year, 'gh': [], 'g': [], 'h': []}
    for i in range(len(year)):
        save_dict['gh'].append(gh[:, i][:, np.newaxis])

        # TODO
        save_dict['h'].append(np.array([]))
        save_dict['g'].append(np.array([]))

    np.save(name + '.npy', save_dict)

