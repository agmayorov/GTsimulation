import os

import matplotlib.pyplot as plt
import numpy as np
import tqdm

from matplotlib.colors import LogNorm

from scipy.optimize import curve_fit

from gtsimulation.Particle.Particle import Particle

sec2year = 3.16887646e-8

plt.rcParams.update({'font.size': 15})


def find_cut_off(energies, times, times_std, pdg):
    nan = np.isnan(times)
    times = times[(1- nan).astype(bool)]
    times_std = times_std[(1- nan).astype(bool)]
    energies = energies[(1- nan).astype(bool)]
    def func(x, a, b, c, d):
        return a * x ** (b) + c * x ** (d)

    # t_min = np.nanmean(times[-8:])*sec2year
    t_min = 92956.5531

    def func1(x, a, b):
        return a * x ** (b) + t_min

    def func_s(x, a, b, c):
        return t_min + b / (1+(x/a)**c)

    # popt2, pcov = curve_fit(func1, energies[energies > 1e12], times[energies > 1e12] * sec2year, maxfev=int(1e8))
    # popt1, pcov = curve_fit(func_s, energies[energies<1e12], times[energies<1e12] * sec2year, maxfev=int(1e8))
    # popt1, pcov = curve_fit(func1, energies[6:], times[6:] * sec2year, maxfev=int(1e9), p0=[1e12, -1])
    popt1, pcov = curve_fit(func_s, energies[6:], times[6:] * sec2year, maxfev=int(1e9), p0=[1e9, 1e7, 1])

    # print(popt1)
    # print(popt2)

    # a, b = popt1
    a, b, c = popt1
    # c, d = popt2
    #
    # E0 = (t_min/a)**(1/b)
    # E0 = (b/(np.sqrt(t_min*(t_min+b))-t_min)-1)**(1/c)*a
    E0 = ((b+t_min)/t_min-1)**(1/c)*a

    p = plt.errorbar(energies*1e6, times * sec2year, yerr=times_std * sec2year, fmt='o', capsize=4, label=f"Modeled data {pdg}")
    # plt.plot(energies*1e6, func1(energies, *popt1), label=f"Fit $a * x^b + t_{{min}}$ {pdg}", color=p[0]._color)
    plt.plot(energies*1e6, func_s(energies, *popt1), label=f"Fit $\\frac{{b}} {{1+(x/a)^c}} + t_{{min}}$ {pdg}", color=p[0]._color)
    ## plt.plot(energies[energies<1e13]*1e6, func1(energies[energies<1e13], *popt1)-t_min)
    # plt.hlines(t_min, xmin=np.min(energies*1e6), xmax=np.max(energies*1e6), label="Minimal time")
    # plt.hlines(np.sqrt(t_min*(t_min+b)), xmin=np.min(energies*1e6), xmax=np.max(energies*1e6), label="Half height",
    #            colors='green')
    ## plt.plot(energies[energies>1e12]*1e6, func1(energies[energies>1e12], *popt2))
    ## plt.plot(energies*1e6, func(energies, *popt1, *popt2))
    plt.vlines(E0*1e6, ymin=np.min(times * sec2year) / 2, ymax=2 * np.max(times * sec2year), colors=p[0]._color, label=f"Cut-off energy {pdg}")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("E, eV")
    plt.ylabel("t, year")
    plt.legend()
    # plt.show()

    return E0*1e6

paths = [f"../../tests/Galaxy{i}" for i in range(1, 43)]
time_energy = dict()
num = 0

cut = []
A = []

for path in tqdm.tqdm(paths):
    try:
        files = os.listdir(path)
    except:
        continue
    for file in files:
        if not file.endswith(".npy"):
            continue

        events = np.load(path + os.sep + file, allow_pickle=True)
        for event in events:
            num+=1
            T = event["Particle"]["T0"]
            time = event["Track"]["Clock"][-1]
            Vx, Vy, Vz = event["Track"]["Velocities"][0]
            pdg = event["Particle"]["PDG"]
            theta = np.arccos(Vz)
            phi = np.arctan2(Vy, Vx)
            # if phi<0:
            #     phi = 2*np.pi + phi
            if pdg in time_energy.keys():
                if T in time_energy[pdg].keys():
                    time_energy[pdg][T][0].append(time)
                    time_energy[pdg][T][1].append(theta)
                    time_energy[pdg][T][2].append(phi)
                else:
                    time_energy[pdg][T] = [[time], [theta], [phi]]
            else:
                time_energy[pdg] = {T: [[time], [theta], [phi]]}

print(num)

N = 2
phi_vals = np.linspace(-np.pi, np.pi, N, endpoint=True)
cos_theta_vals = np.linspace(-1, 1, N, endpoint=True)

phi_vals, cos_theta_vals = np.meshgrid(phi_vals, cos_theta_vals)

E0s = np.zeros((N-1, N-1))

for pdg_code in sorted(time_energy):
    for i in range(N-1):
        for j in range(N-1):
            energies = []

            times = []
            times_std = []

            thetas = []
            thetas_std = []

            phis = []
            phis_std = []
            number = 0
            for energy in sorted(time_energy[pdg_code]):
                energies.append(energy)
                time_E = np.array(time_energy[pdg_code][energy][0])
                theta_E = np.array(time_energy[pdg_code][energy][1])
                phi_E = np.array(time_energy[pdg_code][energy][2])

                idx_phi = (phi_E<phi_vals[j, i+1])*(phi_E>=phi_vals[j, i])
                idx_theta = (np.cos(theta_E)<cos_theta_vals[j+1, i]) * (np.cos(theta_E)>=cos_theta_vals[j, i])

                number += np.sum(idx_phi*idx_theta)

                times.append(np.mean(time_E[idx_theta*idx_phi]))
                times_std.append(np.std(time_E[idx_theta*idx_phi]))

            energies = np.array(energies)
            times = np.array(times)
            times_std = np.array(times_std)
            # plt.title(f"$l\in[{np.round(phi_vals[j, i]*180/np.pi,2)}; {np.round(phi_vals[j, i+1]*180/np.pi, 2)}); b\in[{np.round((np.pi/2 - np.arccos(cos_theta_vals[j+1, i]))*180/np.pi, 2)}; {np.round((np.pi/2 - np.arccos(cos_theta_vals[j, i]))*180/np.pi, 2)})$")# N(E) = {number}")
            # print(f"$\\varphi\in[{phi_vals[i]}; {phi_vals[i+1]})$")
            # print(f"$\\cos\\theta\in[{cos_theta_vals[j]}; {cos_theta_vals[j+1]})$")
            # print(len(energies))
            p = Particle(PDG=pdg_code, Name=None).Name
            E0s[j, i] = find_cut_off(energies, times, times_std, p)
            print(f"{p}: {E0s[j, i]}")
            cut.append(E0s[j, i])
            A.append(Particle(PDG=pdg_code, Name=None).A)


plt.show()

plt.plot(A, cut, '-o')
plt.xlabel("Mass of particle, A [AMU]")
plt.ylabel("Cut-off energy, E [ev]")
plt.yscale("log")
plt.show()


    # phi_vals = 0.5*(phi_vals[1:, 1:] + phi_vals[:-1, :-1])
    # b_vals = np.pi/2 - np.arccos(cos_theta_vals)
    # b_vals = 0.5 * (b_vals[1:, 1:] + b_vals[:-1, :-1])
    #
    # plt.pcolormesh(phi_vals*180/np.pi,
    #                b_vals*180/np.pi,
    #                E0s,
    #                norm=LogNorm(vmin=np.nanmin(E0s), vmax=np.nanmax(E0s)))
    # plt.xlim(-180, 180)
    # plt.ylim(-90, 90)
    # plt.xlabel("$l$")
    # plt.ylabel("$b$")
    # plt.colorbar()
    # plt.show()
    #
    # plt.imshow(E0s, norm=LogNorm(vmin=np.nanmin(E0s), vmax=np.nanmax(E0s)), interpolation="bicubic",
    #            extent=[-180, 180, -90, 90], origin="lower")
    # plt.xlim(-180, 180)
    # plt.ylim(-90, 90)
    # plt.xlabel("$l$")
    # plt.ylabel("$b$")
    # plt.colorbar()
    # plt.show()

