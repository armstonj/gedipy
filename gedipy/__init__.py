
"""
Main module
"""
# This file is part of GEDIPy
# Copyright (C) 2020

from os import path
package_dir = path.abspath(path.dirname(__file__))
__all__=['kmpfit']

GEDIPY_VERSION = '0.0.1'
__version__ = GEDIPY_VERSION

GEDIPY_REFERENCE_COORDS = {'1A': {'x': 'geolocation/longitude_bin0', 'y': 'geolocation/latitude_bin0', 't': 'geolocation/delta_time'},
                           '1B': {'x': 'geolocation/longitude_bin0', 'y': 'geolocation/latitude_bin0', 't': 'geolocation/delta_time'},
                           '2A': {'x': 'lon_lowestmode', 'y': 'lat_lowestmode', 't': 'delta_time'},
                           '2B': {'x': 'geolocation/lon_lowestmode', 'y': 'geolocation/lat_lowestmode', 't': 'geolocation/delta_time'},
                           'ATL03': {'x': 'heights/lon_ph', 'y': 'heights/lat_ph', 't': 'heights/delta_time'},
                           'ATL08': {'x': 'land_segments/longitude', 'y': 'land_segments/latitude', 't': 'land_segments/delta_time'}}

GEDIPY_REFERENCE_DATASETS = {'1A': {'quality': None},
                             '1B': {'quality': None},
                             '2A': {'quality': 'quality_flag'},
                             '2B': {'quality': 'l2b_quality_flag'},
                             'ATL03': {'quality': None},
                             'ATL08': {'quality': None}}

GEDIPY_GEDI_BEAMS = {'BEAM0000':0, 'BEAM0001':1, 'BEAM0010':2, 'BEAM0011':3,
                     'BEAM0101':5, 'BEAM0110':6, 'BEAM1000':8, 'BEAM1011':11}
