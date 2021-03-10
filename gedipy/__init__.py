
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
                           '4A': {'x': 'lon_lowestmode', 'y': 'lat_lowestmode', 't': 'delta_time'},
                           'ATL03': {'x': 'heights/lon_ph', 'y': 'heights/lat_ph', 't': 'heights/delta_time'},
                           'ATL08': {'x': 'land_segments/longitude', 'y': 'land_segments/latitude', 't': 'land_segments/delta_time'},
                           'LVISC1B': {'x': 'LON0', 'y': 'LAT0', 't': 'TIME'},
                           'LVISC2B': {'x': 'GLON', 'y': 'GLAT', 't': 'TIME'}}

GEDIPY_REFERENCE_DATASETS = {'1A': {'quality': None, 'solar_elevation': 'geolocation/solar_elevation', 'degrade': 'geolocation/degrade'},
                             '1B': {'quality': None, 'solar_elevation': 'geolocation/solar_elevation', 'degrade': 'geolocation/degrade'},
                             '2A': {'quality': 'quality_flag', 'solar_elevation': 'solar_elevation', 'degrade': 'degrade_flag'},
                             '2B': {'quality': 'l2b_quality_flag', 'solar_elevation': 'geolocation/solar_elevation', 'degrade': 'geolocation/degrade_flag'},
                             '4A': {'quality': 'l4_quality_flag', 'solar_elevation': 'solar_elevation', 'degrade': 'degrade_flag'},
                             'ATL03': {'quality': None, 'solar_elevation': 'geolocation/solar_elevation', 'degrade': None},
                             'ATL08': {'quality': None, 'solar_elevation': 'land_segments/solar_elevation', 'degrade': None},
                             'LVISC1B': {'quality': None, 'solar_elevation': None, 'degrade': None},
                             'LVISC2B': {'quality': 'quality', 'solar_elevation': None, 'degrade': None}}

GEDIPY_GEDI_BEAMS = {'BEAM0000':0, 'BEAM0001':1, 'BEAM0010':2, 'BEAM0011':3,
                     'BEAM0101':5, 'BEAM0110':6, 'BEAM1000':8, 'BEAM1011':11}

GEDIPY_MODE_SELECTION_OPTS = {"amp_thresh": 50.0, "energy_thresh": 50.0, "cumulative_energy_thresh": 2.5,
                              "cumulative_energy_minimum": 1.5, "pulse_sep_thresh": 8.0,
                              "botlocdist_limit1": 60.0, "botlocdist_limit2": 120.0, "botlocdist_limit3": 240.0,
                              "ampval_limit2": 250.0, "ampval_limit3": 500.0}

