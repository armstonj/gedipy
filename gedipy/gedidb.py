
"""
Classes to provide an interface to GEDI files, orbits and shot numbers
"""
# This file is part of GEDIPy
# Copyright (C) 2020

import os
import sys
import h5py
import numpy
import json
import urllib.request

from . import h5io

GEDIPY_FINDER_URL = 'https://lpdaacsvc.cr.usgs.gov/services/gedifinder'


class ShotNumber:
    """Parent class for shot numbers"""
    def __init__(self, shot_number):
        self.shot_number = shot_number


class GEDIShotNumber(ShotNumber):
    """
    Generic object for GEDI shot numbers

    Format: OOOOOBBFFFNNNNNNNN

	where
	OOOOO: Orbit number
	BB: Beam number
	FFF: Release 1 - Minor frame number (0-241)
             Release 2 - Granule ID (1-4)
	NNNNNNNN: Shot number within orbit

    If a packet is dropped (never received on the ground), NNNNNNNN will not
    save space for it; however, if a packet with a bad CRC is received it will
    be ‘skipped’ in NNNNNN. The bad CRC packet will not be in the L1A data, but
    its place is held in case it can be corrected in processing updates.
    """

    def __init__(self, shot_number):
        self.shot_number = shot_number

    def get_orbit_number(self):
        return self.shot_number // 10000000000000

    def get_beam(self):
        return (self.shot_number % 10000000000000) // 100000000000

    def get_frame(self):
        return (self.shot_number % 100000000000) // 100000000

    def get_granule(self):
        return self.get_frame()

    def get_index(self):
        return self.shot_number % 100000000

    def __repr__(self):
        return repr(self.shot_number)


class FileDatabase:
    def __init__(self):
        self.file_by_orbit = {}

    def get_reader(self, filename):
        for cls in h5io.LidarFile.__subclasses__():
            try:
                h5_file = cls(filename)
                h5_file.open_h5()
                return h5_file
            except h5io.GEDIPyDriverError:
                pass
            except:
                print('Unexpected error:', sys.exc_info()[0])
                raise

    def add_file_list(self, list_filename):
        with open(list_filename, 'r') as fid:
            for line in fid:
                fn = line.rstrip('\n')
                if len(fn) > 0:
                    self.add_file(fn)

    def add_file(self, filename):
        h5file = self.get_reader(filename)
        if h5file:
            if h5file.is_valid():
                orbit = h5file.get_orbit_number()
                if orbit not in self.file_by_orbit:
                    self.file_by_orbit[orbit] = []
                self.file_by_orbit[orbit].append(h5file)
            else:
                print('{} is not a valid H5 file'.format(filename))

    def get_file(self, orbit_number):
        if orbit_number in self.file_by_orbit:
            return self.file_by_orbit[orbit_number]
        else:
            return None

    def export_to_json(self, fn):
        def __serialize(obj):
            if isinstance(obj, h5io.GEDIH5File):
                return obj.filename
            else:
                raise TypeError
        with open(fn, 'w') as f:
            json.dump(self.file_by_orbit, f, sort_keys=True, indent=4, default=__serialize)

    def query_gedi_finder(self, bbox, product='GEDI02_B', version=1, output='json', localroot=None):
        url_template = '{}?product={}&version={:03d}&bbox=[{:f},{:f},{:f},{:f}]&output={}'
        url = url_template.format(GEDIPY_FINDER_URL, product, version,
                                  bbox[1], bbox[0], bbox[3], bbox[2], output)
        handle = urllib.request.urlopen(url)
        json_str = handle.read()
        h5_files = json.loads(json_str)
        if not h5_files['error_code']:
            for fpath in h5_files['data']:
                fn = os.path.basename(fpath)
                if localroot:
                    fn = os.path.join(localroot, fn)
                if os.path.exists(fn):
                    h5file = h5io.GEDIH5File(fn)
                    if h5file.is_valid():
                        orbit = h5file.get_orbit_number()
                        self.file_by_orbit[orbit] = h5file
                    else:
                        print('{} is not a valid GEDI H5 file'.format(fn))


class ShotNumberDatabase:
    def __init__(self):
        self.shot_numbers_by_orbit_beam = {}

    def add_shot_number_file(self, filename):
        with open(filename, 'r') as fid:
            for line in fid:
                if len(line) > 1:
                    sn = GEDIShotNumber(int(line.strip()))
                    orbit = sn.get_orbit_number()
                    beam = sn.get_beam()
                    if orbit not in self.shot_numbers_by_orbit_beam:
                        self.shot_numbers_by_orbit_beam[orbit] = {}
                    if beam not in self.shot_numbers_by_orbit_beam[orbit]:
                        self.shot_numbers_by_orbit_beam[orbit][beam] = []
                    self.shot_numbers_by_orbit_beam[orbit][beam].append(sn)

    def add_gedi_shot_numbers_from_h5(self, filename):
        f = h5py.File(filename, 'r')
        for group in f.keys():
            if group.startswith('BEAM'):
                shot_number = f[group]['shot_number'][()]
                orbit = shot_number // 10000000000000
                beam = (shot_number % 10000000000000) // 100000000000
                for i in range(shot_number.size):
                    sn = GEDIShotNumber(shot_number[i])
                    if orbit[i] not in self.shot_numbers_by_orbit_beam:
                        self.shot_numbers_by_orbit_beam[orbit[i]] = {}
                    if beam[i] not in self.shot_numbers_by_orbit_beam[orbit[i]]:
                        self.shot_numbers_by_orbit_beam[orbit[i]][beam[i]] = []
                    self.shot_numbers_by_orbit_beam[orbit[i]][beam[i]].append(sn)
