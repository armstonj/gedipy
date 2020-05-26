
"""
I/O of GEDI H5 data products
"""
# This file is part of GEDIPy
# Copyright (C) 2020

import os
import re
import h5py
import numpy
import pandas
import datetime

from numba import jit

from pygeos import box
from pygeos import Geometry
from pygeos import contains
from pygeos import points

from . import userfunctions
from . import GEDIPY_REFERENCE_COORDS
from . import GEDIPY_REFERENCE_DATASETS


def append_to_h5_dataset(name, group, new_data, chunksize=(14200,), shot_axis=0, skip_if_existing=False):
    """
    Write/append a dataset to a GEDI H5 file
    Args:
        name (str): The dataset name
        group (obj): The h5py handle to the group with the dataset 
        new_data (array): The new data to append to the dataset
        chunksize (tuple): H5 chunksize (default (14200,))
        shot_axis: The array axis corresponding to number of shots
        skip_if_existing: Do not append if the dataset alreadyt exists
    """
    if name not in group:
        if new_data.ndim == 2:
            chunksize = (128,128)
        else:
            if not chunksize:
                chunksize=(14200,)
            chunksize = (new_data.shape[0],) * (new_data.ndim - 1) + chunksize
        maxshape = (None,) * new_data.ndim
        group.create_dataset(name, data=new_data, maxshape=maxshape,
                chunks=chunksize, compression='gzip', compression_opts=4)
    elif not skip_if_existing:
        oldsize = group[name].shape[shot_axis]
        newsize = oldsize + new_data.shape[shot_axis]
        group[name].resize(newsize, axis=shot_axis)
        if shot_axis == 0:
            if len(group[name].shape) == 1:
                group[name][oldsize:] = new_data
            elif len(group[name].shape) == 2:
                group[name][oldsize:,:] = new_data
            else:
                raise Exception('invalid number of dimensions ({})'.format(len(group[name].shape)))
        elif shot_axis == 1:
            group[name][:,oldsize:] = new_data
        else:
            raise Exception('invalid shot_axis ({})'.format(shot_axis))


def write_dict_to_h5(group, name, data):
    """
    Write a dictionary to a GEDI H5 file
    Args:
        name (str): The dataset name
        group (obj): The h5py handle to the group with the dataset 
        new_data (array): The dictionary to write to the h5 file
    """
    if name not in group:
        group.create_group(name)
    for k,i in data.items():
        if i is not None:
            if isinstance(i, (int, float, str)):
                path = '{}/{}'.format(name,k)
                group[path] = i
            elif isinstance(i, list):
                new_data = numpy.array(i)
                maxshape = (None,) * new_data.ndim
                group[name].create_dataset(k, data=new_data, maxshape=maxshape,
                    chunks=(1,), compression='gzip', compression_opts=4)
            elif isinstance(i, dict):
                write_dict_to_h5(group[name], k, i)
            else:
                raise ValueError('Cannot write {} to h5'.format(type(i)))


class GEDIH5File:
    def __init__(self, filename):
        self.filename = filename
        self.filename_pattern = re.compile(r'GEDI0(1|2)_(A|B)_(\d{13})_O(\d{5})_T(\d{5})_(\d{2})_(\d{3})_(\d{2})\.h5')

    def is_valid(self):
        return self.is_valid_filename() and h5py.is_hdf5(self.filename)

    def is_valid_filename(self):
        m = self.filename_pattern.fullmatch(os.path.basename(self.filename))
        if m:
            return True
        else:
            return False

    def get_orbit_number(self):
        m = self.filename_pattern.fullmatch(os.path.basename(self.filename))
        if m:
            return int(m.group(4))
        else:
            raise ValueError('invalid GEDI filename: "{}"'.format(self.filename))

    def get_product_id(self):
        m = self.filename_pattern.fullmatch(os.path.basename(self.filename))
        if m:
            return '{}{}'.format(m.group(1), m.group(2))
        else:
            raise ValueError('invalid GEDI filename: "{}"'.format(self.filename))

    def get_nrecords(self, beam):
        nshots = self.fid[beam]['shot_number'].shape[0]
        return nshots

    def open(self):
        self.fid = h5py.File(self.filename, 'r')
        self.beams = [beam for beam in self.fid.keys() if beam.startswith('BEAM')]

    def close(self):
        self.fid.close()
        self.beams = None

    def read_shots(self, beam, start=0, finish=None, dataset_list=[]):
        if not finish:
            finish = f[beam]['shot_number'].shape[0]

        dtype_list = []
        for name in dataset_list:
            if isinstance(f[beam][name], h5py.Dataset):
                n = os.path.basename(name)
                s = f[beam][name].dtype.str
                if f[beam][name].ndim > 1:
                    if self.get_product_id() == '2A':
                        t = f[beam][name].shape[1:]
                    else:
                        t = f[beam][name].shape[0:-1]
                    dtype_list.append((str(n), s, t))
                else:
                    dtype_list.append((str(n), s))

        num_records = finish - start
        data = numpy.empty(num_records, dtype=dtype_list)
        for i,name in enumerate(dataset_list):
            n = dtype_list[i][0]
            if isinstance(f[beam][name], h5py.Dataset):
                data[n] = f[beam][name][start:finish]
            else:
                print('{} not found'.format(name))

        return data

    @staticmethod
    @jit(nopython=True)
    def waveform_1d_to_2d(start_indices, counts, data, elev_bin0, v, out_w):
        for i in range(start_indices.shape[0]):
            for j in range(counts[i]):
                out_w[j, i] = data[start_indices[i] + j]

    def read_tx_waveform(self, beam, start=0, finish=None, minlength=None):
        if not finish:
            finish = self.fid[beam]['tx_sample_start_index'].shape[0]

        start_indices = self.fid[beam]['tx_sample_start_index'][start:finish] - 1
        counts = self.fid[beam]['tx_sample_count'][start:finish]
        waveforms = self.fid[beam]['txwaveform'][start_indices[0]:(start_indices[-1]+counts[-1])]

        max_count = numpy.max(counts)
        if minlength:
            max_count = max(minlength, max_count)
        out_shape = (max_count, counts.shape[0])

        out_waveforms = numpy.zeros(out_shape, dtype=waveforms.dtype)
        start_indices -= numpy.min(start_indices)
        waveform_1d_to_2d(start_indices, counts, waveforms, out_waveforms)

        return out_waveforms

    def read_rx_waveform(self, beam, start=0, finish=None, minlength=None, elevation=False):
        if not finish:
            finish = self.fid[beam]['rx_sample_start_index'].shape[0]

        start_indices = self.fid[beam]['rx_sample_start_index'][start:finish] - 1
        counts = self.fid[beam]['rx_sample_count'][start:finish]
        waveforms = self.fid[beam]['rxwaveform'][start_indices[0]:(start_indices[-1]+counts[-1])]

        max_count = numpy.max(counts)
        if minlength:
            max_count = max(minlength, max_count)
        out_shape = (max_count, counts.shape[0])
        
        out_waveforms = numpy.zeros(out_shape, dtype=waveforms.dtype)
        start_indices -= numpy.min(start_indices)
        waveform_1d_to_2d(start_indices, counts, waveforms, out_waveforms)
        
        if elevation:
            elev_bin0 = self.fid[beam]['geolocation']['elevation_bin0'][start:finish]
            elev_lastbin = self.fid[beam]['geolocation']['elevation_lastbin'][start_finish]
            v = (elev_bin0 - elev_lastbin) / (counts - 1)
            
            bin_dist = numpy.arange(max_count) * v
            out_elevation = elev_bin0 - numpy.repeat(bin_dist[numpy.newaxis],v.shape[0],axis=0)
            
            return out_waveforms, out_elevation
        else:
            return out_waveforms

    def copy_attrs(self, output_fid, group):
        for key in self.fid[group].attrs.keys():
            if key not in output_fid[group].attrs.keys():
                output_fid[group].attrs[key] = self.fid[group].attrs[key]

    def export_shots(self, beam, subset, dataset_list=[]):
        # Get the group information
        group = self.fid[beam]
        nshots = group['shot_number'].shape[0]

        # Find indices to extract
        product_id = self.get_product_id()
        idx_extract = userfunctions.get_geom_indices(group, product_id, subset)

        # Use h5py simple indexing - faster
        if not numpy.any(idx_extract):
            return
        tmp, = numpy.nonzero(idx_extract)
        idx_start = numpy.min(tmp)
        idx_finish = numpy.max(tmp) + 1
        idx_subset = tmp - idx_start

        # Function to extract datasets for selected shots
        def get_selected_shots(name, obj):
            if isinstance(obj, h5py.Dataset):
                try:
                    shot_axis = obj.shape.index(nshots)
                    if obj.ndim == 1:
                        arr = obj[idx_start:idx_finish]
                        colnames = [name]
                    elif obj.ndim == 2:
                        if shot_axis == 0:
                            arr = obj[idx_start:idx_finish,...]
                        else:
                            arr = numpy.transpose(obj[...,idx_start:idx_finish])
                        colnames = ['{}_{:03d}'.format(name,i) for i in range(obj.shape[shot_axis-1])]
                    else:
                        print('{} is not a 1D or 2D dataset'.format(name))
                        raise
                    df = pandas.DataFrame(data=arr[idx_subset,...], columns=colnames)
                    datasets.append(df)
                except ValueError:
                    print('{} is not a footprint level dataset'.format(name))
                    raise
                except:
                    print("Unexpected error:", sys.exc_info()[0])
                    raise

        # Extract and combine the data for extracted shots
        datasets = []
        for name in dataset_list:
            out_name = os.path.basename(name)
            get_selected_shots(out_name, group[name])
        outdata = pandas.concat(datasets, axis=1)

        return outdata

    def copy_shots(self, output_fid, beam, subset, output_2d, geom=False, dataset_list=[]):
        group = self.fid[beam]
        nshots = group['shot_number'].size
        product_id = self.get_product_id()

        # Find indices to extract
        if geom:
            idx_extract = userfunctions.get_geom_indices(group, product_id, subset)
        else:
            shot_numbers = group['shot_number'][()]
            idx_extract = userfunctions.get_shot_indices(subset, shot_numbers)

        # Use h5py simple indexing - faster
        if not numpy.any(idx_extract):
            return
        tmp, = numpy.nonzero(idx_extract)
        idx_start = numpy.min(tmp)
        idx_finish = numpy.max(tmp) + 1
        idx_subset = tmp - idx_start

        # Create beam in output file
        if beam not in output_fid:
            output_fid.create_group(beam)
        out_group = output_fid[beam]

        # Datasets not to copy in the main visititems() call because they will
        # be dealt with separately
        copy_blacklist = (
            'rx_sample_count',
            'rx_sample_start_index',
            'rxwaveform',
            'tx_sample_count',
            'tx_sample_count',
            'tx_sample_start_index',
            'txwaveform',
            'pgap_theta_z'
        )

        def copy_selected_shots(name, obj):
            if isinstance(obj, h5py.Group):
                if name not in out_group:
                    out_group.create_group(name)
            elif isinstance(obj, h5py.Dataset):
                # Copy selected shot numbers for most datasets
                if os.path.basename(name) not in copy_blacklist and 0 not in obj.shape:
                    if nshots in obj.shape:
                        shot_axis = obj.shape.index(nshots)
                        if shot_axis == 0:
                            tmp = obj[idx_start:idx_finish,...]
                            append_to_h5_dataset(name, out_group, tmp[idx_subset,...], chunksize=obj.chunks, shot_axis=shot_axis)
                        else:
                            tmp = obj[...,idx_start:idx_finish]
                            append_to_h5_dataset(name, out_group, tmp[...,idx_subset], chunksize=obj.chunks, shot_axis=shot_axis)
                    else:
                        # ancillary / short_term datasets
                        append_to_h5_dataset(name, out_group, obj, chunksize=obj.chunks)

        if len(dataset_list) > 0:
            for name in dataset_list:
                copy_selected_shots(name, group[name])
        else:
            group.visititems(copy_selected_shots)

        # Copy waveforms
        if product_id in ('1A','1B'):
            if len(dataset_list) == 0 or 'txwaveform' in dataset_list:
                self._copy_waveforms(group, 'txwaveform', out_group, 'tx', idx_start, idx_finish, idx_subset, output_2d)
            if len(dataset_list) == 0 or 'rxwaveform' in dataset_list:
                self._copy_waveforms(group, 'rxwaveform', out_group, 'rx', idx_start, idx_finish, idx_subset, output_2d)
        elif product_id == '2B':
            if len(dataset_list) == 0 or 'pgap_theta_z' in dataset_list:
                self._copy_waveforms(group, 'pgap_theta_z', out_group, 'rx', idx_start, idx_finish, idx_subset, output_2d)

    def _copy_waveforms(self, in_group, waveform_name, out_group, prefix, idx_start, idx_finish, idx_subset, output_2d):
        start_indices_name = '{}_sample_start_index'.format(prefix)
        counts_name = '{}_sample_count'.format(prefix)

        start_indices = in_group[start_indices_name][idx_start:idx_finish]
        start_indices = start_indices[idx_subset] - 1
        counts = in_group[counts_name][idx_start:idx_finish]
        counts = counts[idx_subset]
        waveforms = in_group[waveform_name]
        nshots = idx_subset.shape[0]

        if output_2d:
            # Get total waveform size
            max_count = numpy.max(counts)
            waveform_len = 1420 if prefix == 'rx' else 128
            out_len = max(max_count, waveform_len)
            out_shape = (out_len, nshots)
            out_waveforms = numpy.zeros(out_shape, dtype=waveforms.dtype)

            # Fill waveform array
            for i in range(nshots):
                out_waveforms[0:counts[i], i] = waveforms[start_indices[i]:start_indices[i] + counts[i]]

            # Write waveforms to disk
            append_to_h5_dataset(waveform_name, out_group, out_waveforms, chunksize=(128,128), shot_axis=1)
        else:
            # Get output waveform start indices
            out_start_indices = numpy.cumsum(counts)
            out_start_indices = numpy.roll(out_start_indices, 1)
            out_start_indices[0] = 0

            # Get total waveform size
            nsamples = numpy.sum(counts)
            out_waveforms = numpy.zeros(nsamples, dtype=waveforms.dtype)

            # Fill waveform arrays
            for i in range(nshots):
                if counts[i] > 0:
                    out_waveforms[out_start_indices[i]:out_start_indices[i]+counts[i]] \
                                = waveforms[start_indices[i]:start_indices[i]+counts[i]]

            # Offset the output start indices
            if start_indices_name in out_group:
                out_start_offset = int(out_group[start_indices_name][-1] + \
                        out_group[counts_name][-1] + 1)
                out_start_indices += out_start_offset
            else:
                out_start_indices += 1

            # Write waveforms to disk
            append_to_h5_dataset(waveform_name, out_group, out_waveforms, chunksize=waveforms.chunks)
            append_to_h5_dataset(start_indices_name, out_group, out_start_indices)
            append_to_h5_dataset(counts_name, out_group, counts)

    def get_quality_flag(self, beam, sensitivity=0.9):
        quality_name = GEDIPY_REFERENCE_DATASETS[self.get_product_id()]['quality']
        if quality_name:
            beam_sensitivity = self.fid[beam]['sensitivity'][()]
            quality_flag = (self.fid[beam][quality_name][()] == 1) & (beam_sensitivity >= sensitivity) 
        else:
            quality_flag = numpy.ones(self.get_nrecords(), dtype=numpy.bool)
        return quality_flag

    def get_utc_time(self, beam):
        delta_time = self.fid[beam][GEDIPY_REFERENCE_COORDS[self.get_product_id()]['t']][()]
        start_utc = datetime.datetime(2018, 1, 1)
        utc_time = [start_utc + datetime.timedelta(seconds=s) for s in delta_time]
        return numpy.array(utc_time)

    def get_coordinates(self, beam):
        longitude = self.fid[beam][GEDIPY_REFERENCE_COORDS[self.get_product_id()]['x']][()]
        latitude = self.fid[beam][GEDIPY_REFERENCE_COORDS[self.get_product_id()]['y']][()]
        return longitude, latitude
    
    def get_dataset(self, beam, name, index=None):
        if self.fid[beam][name].ndim > 1:
            if index:
                if self.get_product_id() == '2A':
                    dataset = self.fid[beam][name][:,index]
                else:
                    dataset = self.fid[beam][name][index,:]
            else:
                dataset = self.fid[beam][name][()]
        else:
            dataset = self.fid[beam][name][()]
        return dataset

