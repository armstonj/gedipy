
"""
Functions that are available to the user
"""
# This file is part of GEDIPy
# Copyright (C) 2020

import numpy
from numba import jit
from numba import prange


@jit(nopython=True)
def convolve_numba(rx, tx, result):
    """
    1D convolution for use with numba
    """
    M = rx.shape[0]
    Mf = tx.shape[0]
    Mf2 = Mf // 2
    tx_sum = numpy.sum(tx)
    for i in range(Mf2, M-Mf2):
        num = 0
        for ii in range(Mf):
            num += (tx[Mf-1-ii] * rx[i-Mf2+ii])
        result[i] = num / tx_sum


@jit(nopython=True)
def gaussian(x, sigma, offset):
    """
    Return y for a normalized Gaussian given other parameters
    """
    y = numpy.exp(-1.0 * (x - offset)**2 / (2 * sigma*2)) / (sigma * numpy.sqrt(2 * numpy.pi))
    return y


@jit(nopython=True)
def gauss_noise_thresholds(sig, res, probNoise=0.05, probMiss=0.1):
    """
    Calculate Gaussian thresholds for normally distributed noise
    """
    probN = 1.0 - probNoise / (30.0 / 0.15)
    probS = 1.0 - probMiss

    # Determine start
    x = 0.0
    tot = 0.0
    y = gaussian(x,sig,0.0)
    while y >= 0.000000001:
        if numpy.abs(x) > res:
            tot += 2.0 * y
        else:
            tot += y
        x -= res
        y = gaussian(x,sig,0.0)
    tot *= res

    foundS = 0
    foundN = 0
    cumul = 0.0
    while ((foundS == 0) or (foundN == 0)):
        y = gaussian(x,sig,0.0)
        cumul += y * res / tot
        if foundS == 0:
            if cumul >= probS:
                foundS = 1
                threshS = x
        if foundN == 0:
            if cumul >= probN:
                foundN = 1
                threshN = x
        x += res

    return threshN, threshS


@jit(nopython=True)
def get_beam_sensitivity(noise_mean, noise_std_dev, rx_sample_count, rx_sample_sum,
    nNsig, nSsig, pSigma, fSigma=5.5):
    """
    Calculate the beam sensitivity metric
    """
    totE = rx_sample_sum - rx_sample_count * noise_mean
    if totE > 0:
        gAmp = (nNsig + nSsig) * noise_std_dev
        slope = 2 * numpy.pi / 180
        tanSlope = numpy.sin(slope) / numpy.cos(slope)
        sigEff = numpy.sqrt(pSigma**2 + fSigma**2 * tanSlope**2)
        gArea = gAmp * sigEff * numpy.sqrt(2 * numpy.pi) / totE
        if gArea > 0:
            sensitivity = 1.0 - gArea
        else:
            sensitivity = 0.0
    else:
        sensitivity = 0.0

    return sensitivity


def expgaussian(p, x, y):
    """
    Return residuals from an exponentially modified Gaussian.
    expgaussian(p[amplitude, center, sigma, gamma],x,y)
    https://en.wikipedia.org/wiki/Exponentially_modified_Gaussian_distribution
    """
    gss = p[3] * p[2] * p[2]
    arg1 = p[3] * (p[1] + gss/2.0 - x)
    arg2 = (p[1] + gss - x) / max(TINY, (S2*p[2]))
    yfit = p[0] * (p[3]/2) * numpy.exp(arg1) * erfc(arg2)
    return yfit - y


def eval_expgaussian(p, x):
    """
    Evaluate an exponentially modified Gaussian fit.
    expgaussian(p[amplitude, center, sigma, gamma],x)
    https://en.wikipedia.org/wiki/Exponentially_modified_Gaussian_distribution
    """
    gss = p[3] * p[2] * p[2]
    arg1 = p[3] * (p[1] + gss/2.0 - x)
    arg2 = (p[1] + gss - x) / max(TINY, (S2*p[2]))
    yfit = p[0] * (p[3]/2) * numpy.exp(arg1) * erfc(arg2)
    return yfit

