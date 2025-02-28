import os
import datetime
import numpy as np


def LoadGaussCoeffs(npyfile: str, date: datetime.datetime):
    year = date.year
    y = year + (date - datetime.datetime(year, 1, 1)).days / (365 + float(
        (not bool(year % 4) and bool(year % 100)) or not bool(year % 4)))

    coeffs = np.load(npyfile, allow_pickle=True).item()
    years = coeffs["years"]
    g_tot = coeffs['g']
    h_tot = coeffs['h']
    gh_tot = coeffs['gh']

    if len(years) == 1:
        return g_tot[0], h_tot[0], gh_tot[0].flatten()

    assert years[0] <= y <= years[-1]

    idx = np.where(years - y < 0)[0]
    if idx.size == 0:
        lastepoch = 0
    else:
        lastepoch = idx[-1]

    nextepoch = lastepoch + 1

    lastg = g_tot[lastepoch]
    nextg = g_tot[nextepoch]

    lasth = h_tot[lastepoch]
    nexth = h_tot[nextepoch]

    lastgh = gh_tot[lastepoch]
    nextgh = gh_tot[nextepoch]

    if lastg.shape[0] > nextg.shape[0]:
        smalln = nextg.shape[0]
        nextg = np.zeros_like(lastg)
        nextg[:smalln, :smalln + 1] = g_tot[nextepoch]

        nexth = np.zeros_like(lasth)
        nexth[:smalln, :smalln + 1] = h_tot[nextepoch]

        smalln = len(nextgh)
        nextgh = np.zeros_like(lastgh)
        nextgh[:smalln] = gh_tot[nextepoch]
    elif lastg.shape[0] < nextg.shape[0]:
        smalln = lastg.shape[0]
        lastg = np.zeros_like(nextg)
        lastg[:smalln, :smalln + 1] = g_tot[lastepoch]

        lasth = np.zeros_like(nexth)
        lasth[:smalln, :smalln + 1] = h_tot[lastepoch]

        smalln = len(lastgh)
        lastgh = np.zeros_like(nextgh)
        lastgh[:smalln] = gh_tot[lastepoch]

    if coeffs.get("slope") is not None and coeffs.get("slope")[nextepoch]:
        gslope = nextg
        hslope = nexth
        ghslope = nextgh
    else:
        gslope = (nextg - lastg) / np.diff(years[[lastepoch, nextepoch]])
        hslope = (nexth - lasth) / np.diff(years[[lastepoch, nextepoch]])
        ghslope = (nextgh - lastgh) / np.diff(years[[lastepoch, nextepoch]])

    g = lastg + gslope * (y - years[lastepoch])
    h = lasth + hslope * (y - years[lastepoch])
    gh = lastgh + ghslope * (y - years[lastepoch])

    return g, h, gh.flatten()
