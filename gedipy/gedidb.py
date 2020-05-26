
"""
Classes to provide an interface to GEDI files, orbits and shot numbers
"""
# This file is part of GEDIPy
# Copyright (C) 2020

import os
import h5py
import numpy
import json
import urllib.request

from . import h5io

GEDIPY_FINDER_URL = 'https://lpdaacsvc.cr.usgs.gov/services/gedifinder'


class ShotNumber:
    def __init__(self, shot_number):
        self.shot_number = shot_number

    def get_orbit_number(self):
        return self.shot_number // 10000000000000

    def get_beam(self):
        return (self.shot_number % 10000000000000) // 100000000000

    def get_index(self):
        return self.shot_number % 100000000

    def __repr__(self):
        return repr(self.shot_number)


class FileDatabase:
    def __init__(self):
        self.file_by_orbit = {}

    def add_file_list(self, list_filename):
        with open(list_filename, 'r') as fid:
            for line in fid:
                fn = line.rstrip('\n')
                if len(fn) > 0:
                    h5file = h5io.GEDIH5File(fn)
                    if h5file.is_valid():
                        self.file_by_orbit[h5file.get_orbit_number()] = h5file
                    else:
                        print('{} is not a valid GEDI H5 file'.format(fn))

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
                        self.file_by_orbit[h5file.get_orbit_number()] = h5file
                    else:
                        print('{} is not a valid GEDI H5 file'.format(fn))


class ShotNumberDatabase:
    def __init__(self):
        self.shot_numbers_by_orbit_beam = {}

    def add_shot_number_file(self, filename):
        with open(filename, 'r') as fid:
            for line in fid:
                if len(line) > 1:
                    sn = ShotNumber(int(line.strip()))
                    orbit = sn.get_orbit_number()
                    beam = sn.get_beam()
                    if orbit not in self.shot_numbers_by_orbit_beam:
                        self.shot_numbers_by_orbit_beam[orbit] = {}
                    if beam not in self.shot_numbers_by_orbit_beam[orbit]:
                        self.shot_numbers_by_orbit_beam[orbit][beam] = []
                    self.shot_numbers_by_orbit_beam[orbit][beam].append(sn)

    def add_shot_numbers_from_h5(self, filename):
        f = h5py.File(filename, 'r')
        for group in f.keys():
            if group.startswith('BEAM'):
                shot_number = f[group]['shot_number'][()]
                orbit = shot_number // 10000000000000
                beam = (shot_number % 10000000000000) // 100000000000
                for i in range(shot_number.size):
                    sn = ShotNumber(shot_number[i])
                    if orbit[i] not in self.shot_numbers_by_orbit_beam:
                        self.shot_numbers_by_orbit_beam[orbit[i]] = {}
                    if beam[i] not in self.shot_numbers_by_orbit_beam[orbit[i]]:
                        self.shot_numbers_by_orbit_beam[orbit[i]][beam[i]] = []
                    self.shot_numbers_by_orbit_beam[orbit[i]][beam[i]].append(sn)

