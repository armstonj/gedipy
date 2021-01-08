
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

from numba import njit

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


@njit
def _add_counts(col, row, band, outimage):
    """
    Numba functions to add counts to an existing image
    """
    for i in range(col.shape[0]):
        if (row[i] >= 0) & (col[i] >= 0) & (row[i] < outimage.shape[1]) & (col[i] < outimage.shape[2]):
            outimage[band, int(row[i]), int(col[i])] += 1


def grid_counts(x, y, shotnumber, outimage, xmin, ymax, xbinsize, ybinsize):
    """
    Function to count number of shots, tracks and orbits on a grid
    outimage band 1 = Number of shots
    outimage band 2 = Number of tracks
    outimage band 3 = Number of orbits
    """
    col = numpy.int16((x - xmin) / xbinsize)
    row = numpy.int16((ymax - y) / ybinsize)
     
    _add_counts(col, row, 0, outimage)

    track = shotnumber // 100000000000
    val = numpy.unique([col,row,track], axis=1)
    _add_counts(val[0,:], val[1,:], 1, outimage)

    orbit = shotnumber // 10000000000000
    val = numpy.unique([col,row,orbit], axis=1)    
    _add_counts(val[0,:], val[1,:], 2, outimage)


@njit
def grid_moments(x, y, z, outimage, xmin, ymax, xbinsize, ybinsize):
    """
    Numba function to calculate running mean and variance using Welfords algorithm
    outimage band 1 = Mean (M1)
    outimage band 2 = Standard deviation (M2)
    outimage band 3 = Skewness (M3)
    outimage band 4 = Kurtosis (M4)
    outimage band 5 = Number of shots (n)
    """
    for i in range(x.shape[0]):
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


@njit
def grid_quantiles(x, y, z, outimage, xmin, ymax, xbinsize, ybinsize, quantiles, step):
    """
    Numba function to calculate running quantiles using the FAME algorithm
    This is experimental - results are an approximation
    http://www.eng.tau.ac.il/~shavitt/courses/LargeG/streaming-median.pdf
    """
    nquantiles = len(quantiles)
    for j in range(nquantiles):
        step_val = max([x[0] / 2, step])
        step_up = 1.0 - quantiles[j]
        step_down = quantiles[j]
        for i in range(x.shape[0]):
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
