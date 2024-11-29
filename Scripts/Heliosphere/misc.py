import numpy as np
from numba import jit, prange

@jit(nopython=True, parallel=True)
def smoothing_function(x, y, mean=True, window=2, pad=1):
    def bisection(array, value):
        '''Given an ``array`` , and given a ``value`` , returns an index j such that ``value`` is between array[j]
        and array[j+1]. ``array`` must be monotonic increasing. j=-1 or j=len(array) is returned
        to indicate that ``value`` is out of range below and above respectively.'''
        n = len(array)
        if (value < array[0]):
            return -1
        elif (value > array[n - 1]):
            return n
        jl = 0  # Initialize lower
        ju = n - 1  # and upper limits.
        while (ju - jl > 1):  # If we are not yet done,
            jm = (ju + jl) >> 1  # compute a midpoint with a bitshift
            if (value >= array[jm]):
                jl = jm  # and replace either the lower limit
            else:
                ju = jm  # or the upper limit, as appropriate.
            # Repeat until the test condition is satisfied.
        if (value == array[0]):  # edge cases at bottom
            return 0
        elif (value == array[n - 1]):  # and top
            return n - 1
        else:
            return jl

    len_x = len(x)
    max_x = np.max(x)
    xoutmid = np.full(len_x, np.nan)
    xoutmean = np.full(len_x, np.nan)
    yout = np.full(len_x, np.nan)

    for i in prange(len_x):
        x0 = x[i]
        xf = window * x0

        if xf < max_x:
            # e = np.where(x  == x[np.abs(x - xf).argmin()])[0][0]
            e = bisection(x, xf)
            if e < len_x:
                if mean:
                    yout[i] = np.nanmean(y[i:e])
                    xoutmid[i] = x0 + np.log10(0.5) * (x0 - x[e])
                    xoutmean[i] = np.nanmean(x[i:e])
                else:
                    yout[i] = np.nanmedian(y[i:e])
                    xoutmid[i] = x0 + np.log10(0.5) * (x0 - x[e])
                    xoutmean[i] = np.nanmean(x[i:e])

    return xoutmid, xoutmean, yout