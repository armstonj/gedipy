
"""
Functions that are available to the user
"""
# This file is part of GEDIPy
# Copyright (C) 2020

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


@jit(nopython=True, parallel=True)
def grid_moments(x, y, z, outimage, xmin, ymax, xbinsize, ybinsize):
    """
    Numba function to calculate running mean and variance using Welfords algorithm
    outimage band 1 = Mean (M1)
    outimage band 2 = Standard deviation (M2)
    outimage band 3 = Skewness (M3)
    outimage band 4 = Kurtosis (M4)
    outimage band 5 = Number of shots (n)
    """
    for i in prange(x.shape[0]):
        col = int((x[i] - xmin) / xbinsize)
        row = int((ymax - y[i]) / ybinsize)
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


def finalize_grid_moments(outgrid, profile, gain=1, offset=0):
    """
    Retrieve the mean, standard deviation, skewness, kurtosis and scale outputs
    """
    # Initialize the output
    tmpshape = (4, outgrid.shape[1], outgrid.shape[2])
    tmpgrid = numpy.empty(tmpshape, dtype=outgrid.dtype)

    # Mean
    tmpgrid[0] = numpy.where(outgrid[4] > 0, outgrid[0], profile['nodata'])

    # Standard deviation
    tmp = numpy.full(outgrid[1].shape, profile['nodata'], dtype=outgrid.dtype)
    numpy.divide(outgrid[1], outgrid[4] - 1, out=tmp, where=outgrid[4] > 1)
    numpy.sqrt(tmp, out=tmp, where=outgrid[4] > 1)
    tmpgrid[1] = tmp

    # Skewness
    tmp = numpy.full(outgrid[2].shape, profile['nodata'], dtype=outgrid.dtype)
    numpy.sqrt(outgrid[4], out=tmp, where=outgrid[4] > 2)
    numpy.multiply(tmp, outgrid[2], out=tmp, where=outgrid[4] > 2)
    numpy.divide(tmp, outgrid[1]**1.5, out=tmp, where=outgrid[4] > 2)
    tmpgrid[2] = tmp
    
    # Kurtosis
    tmp = numpy.full(outgrid[3].shape, profile['nodata'], dtype=outgrid.dtype)
    numpy.multiply(outgrid[4], outgrid[3], out=tmp, where=outgrid[4] > 3)
    numpy.divide(tmp, outgrid[1]**2, out=tmp, where=outgrid[4] > 3)
    numpy.subtract(tmp, 3, out=tmp, where=outgrid[4] > 3)
    tmpgrid[3] = tmp

    # Scale and offset
    for i in range(tmpgrid.shape[0]):
        tmp = tmpgrid[i]
        numpy.multiply(tmp, gain, out=tmp, where=outgrid[4] > i)
        numpy.add(tmp, offset, out=tmp, where=outgrid[4] > i)
        outgrid[i] = tmp

    return outgrid.astype(profile['dtype'])


@jit(nopython=True, parallel=True)
def grid_quantiles(x, y, z, outimage, xmin, ymax, xbinsize, ybinsize, quantiles, step):
    """
    Numba function to calculate running quantiles using the FAME algorithm
    This is experimental - results are an approximation
    http://www.eng.tau.ac.il/~shavitt/courses/LargeG/streaming-median.pdf
    """
    nquantiles = len(quantiles)
    for j in prange(nquantiles):
        step_val = max([x[0] / 2, step])
        step_up = 1.0 - quantiles[j]
        step_down = quantiles[j]
        for i in prange(x.shape[0]):
            col = int((x[i] - xmin) / xbinsize)
            row = int((ymax - y[i]) / ybinsize)
            if (row >= 0) & (col >= 0) & (row < outimage.shape[1]) & (col < outimage.shape[2]):
                if outimage[-1,row,col] > 0:
                    if quantiles[j] == 0:
                        if z[i] < outimage[j,row,col]:
                            outimage[j,row,col] = z[i]
                    elif quantiles[j] == 1:
                        if z[i] > outimage[j,row,col]:
                            outimage[j,row,col] = z[i]
                    else:
                        if outimage[j,row,col] > z[i]:
                            outimage[j,row,col] -= step_val * step_up
                        elif outimage[j,row,col] < z[i]:
                            outimage[j,row,col] += step_val * step_down
                        if abs(z[i] - outimage[j,row,col]) < step_val:
                            step_val /= 2.0
                else:
                    outimage[j,row,col] = z[i]


def finalize_grid_quantiles(outgrid, profile, gain=1, offset=0):
    """
    Retrieve the quantiles and scale outputs
    """
    for i in range(outgrid.shape[0] - 1):
        tmp = outgrid[i]
        numpy.multiply(tmp, gain, out=tmp, where=outgrid[-1] > 0)
        numpy.add(tmp, offset, out=tmp, where=outgrid[-1] > 0)
        outgrid[i] = tmp

    return outgrid.astype(profile['dtype'])

