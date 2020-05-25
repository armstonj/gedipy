
"""
Main module
"""
# This file is part of GEDIPy
# Copyright (C) 2020

GEDIPY_VERSION = '0.0.1'
__version__ = GEDIPY_VERSION

GEDIPY_REFERENCE_COORDS = {'1A': {'x': 'geolocation/longitude_bin0', 'y': 'geolocation/latitude_bin0', 't': 'geolocation/delta_time'},
                           '1B': {'x': 'geolocation/longitude_bin0', 'y': 'geolocation/latitude_bin0', 't': 'geolocation/delta_time'},
                           '2A': {'x': 'lon_lowestmode', 'y': 'lat_lowestmode', 't': 'delta_time'},
                           '2B': {'x': 'lon_lowestmode', 'y': 'lat_lowestmode', 't': 'delta_time'}}

GEDIPY_REFERENCE_DATASETS = {'1A': {'quality': None},
                             '1B': {'quality': None},
                             '2A': {'quality': 'quality_flag'},
                             '2B': {'quality': 'l2b_quality_flag'}}

GEDIPY_BEAMS = {'BEAM0000':0, 'BEAM0001':1, 'BEAM0010':2, 'BEAM0011':3, 
                'BEAM0101':5, 'BEAM0110':6, 'BEAM1000':8, 'BEAM1011':11}

