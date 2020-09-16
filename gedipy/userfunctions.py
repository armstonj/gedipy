
"""
Functions that are available to the user
"""
# This file is part of GEDIPy
# Copyright (C) 2020

import h5py
import numpy
import datetime

from osgeo import ogr

from pygeos import box
from pygeos import points
from pygeos import contains

from numba import jit
from numba import prange

from . import GEDIPY_REFERENCE_COORDS


def get_shot_indices(shots, shot_number_dataset):
    shots_np = numpy.array([s.shot_number for s in shots], dtype=numpy.uint64)
    index = numpy.argsort(shot_number_dataset)
    sorted_y = shot_number_dataset[index]
    sorted_index = numpy.searchsorted(sorted_y, shots_np)
    xindex = numpy.take(index, sorted_index, mode="clip")

    idx_extract = []
    for i in range(len(xindex)):
        if shot_number_dataset[xindex[i]] == shots_np[i]:
            idx_extract.append(xindex[i])

    idx_extract_uniq = set(idx_extract)
    return list(idx_extract_uniq)


def get_geom_indices(beam_group, product_id, geomlist):
    xname = GEDIPY_REFERENCE_COORDS[product_id]['x']
    yname = GEDIPY_REFERENCE_COORDS[product_id]['y']
    shot_geoms = points(beam_group[xname][()], y=beam_group[yname][()])
    idx_extract = numpy.zeros(beam_group[xname].size, dtype=numpy.bool)
    for geom in geomlist:
        mask = contains(geom, shot_geoms)
        idx_extract = idx_extract | mask
    return idx_extract


def get_polygon_from_kml(kml_file):
    driver = ogr.GetDriverByName('KML')
    dataSource = driver.Open(kml_file)
    if dataSource is None:
        print('Could not open {}'.format(kml_file))
        exit(1)
    result = []
    number_of_layers = dataSource.GetLayerCount()
    for layer_id in range(number_of_layers):
        layer = dataSource.GetLayerByIndex(layer_id)
        for feat in layer:
            featgeom = feat.GetGeometryRef()
            if featgeom.GetGeometryName() != 'GEOMETRYCOLLECTION':
                geom_wkt = featgeom.ExportToWkt()
                geom = Geometry(geom_wkt)
                result.append(geom)
    return result


def get_dates_from_gedi_mission_weeks(start_mw, end_mw):
    first_mw_date = datetime.datetime.strptime('2018-12-13', '%Y-%m-%d')
    start_mw_offset = (start_mw - 1) * 7
    start_mw_date = first_mw_date + datetime.timedelta(start_mw_offset)
    end_mw_offset = (end_mw - 1) * 7 + 6
    end_mw_date = first_mw_date + datetime.timedelta(end_mw_offset)
    start_time = start_mw_date.strftime('%Y-%m-%d')
    end_time = end_mw_date.strftime('%Y-%m-%d')
    return start_time, end_time


def get_polygon_from_bbox(bbox):
    geom = box(bbox[0], bbox[3], bbox[2], bbox[1])
    result = [geom]
    return result


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


@jit(nopython=True, parallel=True)
def simple_stats(x, y, z, outimage, xmin, ymax, binsize):
    """
    Numba function to calculate running mean and variance
    outimage band 1 = Mean (M1)
    outimage band 2 = Standard deviation (M2)
    outimage band 3 = Skewness (M3)
    outimage band 4 = Kurtosis (M4)
    outimage band 5 = Number of shots (n)
    """
    for i in prange(x.shape[0]):
        col = int((x[i] - xmin) / binsize)
        row = int((ymax - y[i]) / binsize)
        if (row >= 0) & (col >= 0) & (row < outimage.shape[1]) & (col < outimage.shape[2]):
            # Intermediate variables
            n = outimage[4, row, col] + 1
            delta = z[i] - outimage[0, row, col]
            delta_n = delta / n
            delta_n2 = delta_n**2
            term1 = delta * delta_n * (n - 1)
            # M1
            outimage[0, row, col] += delta_n
            # M4
            outimage[3, row, col] += (term1 * delta_n2 * (n**2 - 3 * n + 3) + 
                6 * delta_n2 * outimage[1, row, col] - 4 * delta_n * 
                outimage[2, row, col])
            # M3
            outimage[2, row, col] += (term1 * delta_n * (n - 2) - 3 * 
                delta_n * outimage[1, row, col])
            # M2
            outimage[1, row, col] += term1
            # Number of shots
            outimage[4, row, col] = n


def finalize_simple_stats(outgrid, profile, gain=100, offset=0, dtype='int16', nodata=32767):
    """
    Retrieve the mean, standard deviation, skewness, kurtosis and scale outputs
    """
    # Mean
    tmp = numpy.full(outgrid[0].shape, nodata, dtype=outgrid.dtype)
    numpy.multiply(outgrid[0], gain, out=tmp, where=outgrid[4] > 0)
    numpy.add(tmp, offset, out=tmp, where=outgrid[4] > 0)
    outgrid[0] = tmp

    # Standard deviation
    tmp = numpy.full(outgrid[1].shape, nodata, dtype=outgrid.dtype)
    numpy.divide(outgrid[1], outgrid[2], out=tmp, where=outgrid[2] > 1)
    numpy.sqrt(tmp, out=tmp, where=outgrid[4] > 1)
    numpy.multiply(tmp, gain, out=tmp, where=outgrid[4] > 1)
    numpy.add(tmp, offset, out=tmp, where=outgrid[4] > 1)
    outgrid[1] = tmp

    # Skewness
    tmp = numpy.full(outgrid[2].shape, nodata, dtype=outgrid.dtype)
    numpy.sqrt(outgrid[4], out=tmp, where=outgrid[4] > 0)
    numpy.divide(tmp * outgrid[2], outgrid[1]**1.5, out=tmp, where=outgrid[4] > 1)
    numpy.multiply(tmp, gain, out=tmp, where=outgrid[4] > 1)
    numpy.add(tmp, offset, out=tmp, where=outgrid[4] > 1)
    outgrid[2] = tmp
    
    # Kurtosis
    tmp = numpy.full(outgrid[3].shape, nodata, dtype=outgrid.dtype)
    numpy.divide(outgrid[4] * outgrid[3], outgrid[1]**2 - 3, out=tmp, 
        where=outgrid[4] > 1)
    numpy.multiply(tmp, gain, out=tmp, where=outgrid[4] > 1)
    numpy.add(tmp, offset, out=tmp, where=outgrid[4] > 1)
    outgrid[3] = tmp

    # Number of shots
    gedimask = outgrid[4,:,0]
    idx = numpy.argwhere(gedimask == profile['nodata'])
    outgrid[:,idx,:] = nodata

    return outgrid.astype(dtype)


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
