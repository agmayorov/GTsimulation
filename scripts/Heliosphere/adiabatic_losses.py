import numpy as np
import matplotlib.pyplot as plt



plt.rcParams.update({'font.size':15})

files = [rf'../../tests/Helio/AdiabaticLosses_{i}.npy' for i in range(10)]

adiabatic_losses = {}
time_energy = {}

for file in files:
    events = np.load(file, allow_pickle=True)
    for event in events:
        T0 = event["Particle"]["T0"]
        E = event["Track"]["Energy"]
        t = event["Track"]["Clock"]

        if T0 in adiabatic_losses:
            adiabatic_losses[T0].append(E)
        else:
            adiabatic_losses[T0] = [E]
        time_energy[T0] = (np.max(t), len(t)) if T0 not in time_energy else (min(np.max(t), time_energy[T0][0]),
                                                                         min(len(t), time_energy[T0][1]))

plt.figure()

for T in adiabatic_losses.keys():
    max_t, lent = time_energy[T]
    time = np.linspace(0, max_t, lent, endpoint=True)
    En = []
    for arr in adiabatic_losses[T]:
        En.append(arr[:lent])
    En = np.array(En)
    med = np.median(En, axis=0)
    low = np.quantile(En, q=0.15, axis=0)
    high = np.quantile(En, q=0.85, axis=0)
    plt.plot(time/(3600*24), T-med, label=f'Kin Energy = {np.round(T, 2)} MeV')
    plt.fill_between(time/(3600*24), y1=T-low, y2=T-high, alpha=0.5)

plt.xscale('log')
plt.ylabel("$-\Delta T$ [MeV]")
plt.xlabel('time [days]')
plt.legend()
plt.show()



