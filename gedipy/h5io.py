
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

from numba import njit
from numba import prange

from pygeos import box
from pygeos import contains
from pygeos import points

from . import userfunctions
from . import GEDIPY_REFERENCE_COORDS
from . import GEDIPY_REFERENCE_DATASETS
from . import GEDIPY_MODE_SELECTION_OPTS


def append_to_h5_dataset(name, group, new_data, shot_axis=0,
    skip_if_existing=False):
    """
    Write/append a dataset to a H5 file

    Parameters
    ----------
    name: str
        The dataset name
    group: obj
        The h5py handle to the group with the dataset
    new_data: array
        The new data to append to the dataset
    shot_axis: int
        The array axis corresponding to number of shots
    skip_if_existing: bool
        Do not append if the dataset already exists
    """
    if name not in group:
        if new_data.ndim == 2:
            chunksize = (128,128)
        else:
            chunksize = True
        maxshape = (None,) * new_data.ndim
        group.create_dataset(name, data=new_data, maxshape=maxshape,
                chunks=chunksize, compression='lzf')
    elif not skip_if_existing:
        oldsize = group[name].shape[shot_axis]
        newsize = oldsize + new_data.shape[shot_axis]
        group[name].resize(newsize, axis=shot_axis)
        if shot_axis == 0:
            group[name][oldsize:,...] = new_data
        else:
            group[name][...,oldsize:] = new_data


def write_dict_to_h5(group, name, data):
    """
    Write a dictionary to a GEDI H5 file

    Parameters
    ----------
    name: str
        The dataset name
    group: obj
        The h5py handle to the group with the dataset
    new_data: array-like
        The dictionary to write to the h5 file
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


class Error(Exception):
    """Base class for exceptions in this module."""
    pass


class GEDIPyDriverError(Error):
    """Raised when the driver is invalid"""
    pass


class LidarFile:
    """Parent class for lidar file drivers"""
    def __init__(self, filename):
        self.filename = filename


class GEDIH5File(LidarFile):
    """
    Generic object for I/O of GEDI .h5 data

    Parameters
    ----------
    filename: str
        Pathname to GEDI .h5 file

    """
    def __init__(self, filename):
        self.filename = filename
        self.filename_pattern = re.compile(r'GEDI0(1|2|4)_(A|B)_(\d{13})_O(\d{5})_T(\d{5})_(\d{2})_(\d{3})_(\d{2})\.h5')

    def is_valid(self):
        return h5py.is_hdf5(self.filename)

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

    def open_h5(self, short_name=None):
        self.fid = h5py.File(self.filename, 'r')
        gedi_product_names = ('GEDI_L1A','GEDI_L1B','GEDI_L2A','GEDI_L2B','GEDI_L4A')
        if not short_name:
            if 'short_name' in self.fid.attrs:
                short_name = self.fid.attrs['short_name']
        if short_name not in gedi_product_names:
            raise GEDIPyDriverError
        self.beams = [beam for beam in self.fid if beam.startswith('BEAM')]

    def close_h5(self):
        self.fid.close()
        self.beams = None

    def read_shots(self, beam, start=0, finish=None, dataset_list=[]):
        """
        Read data of GEDI .h5 files into numpy.ndarray

        Parameters
        ----------
        beam: str
            Name of beam assessed, i.e. 'BEAM0001'
        start: int
            start of np.ndarray like slicing, Default=0
        finish: int/ None
            end of np.ndarray like slicing, Default=None
        dataset_list: list of str
            List of GEDI h5 dataset paths

        Returns
        -------
        data: numpy.ndarray
            numpy.ndarray of read data
        """
        if not finish:
            finish = self.fid[beam+'/shot_number'].shape[0]

        dtype_list = []
        for name in dataset_list:
            if isinstance(self.fid[beam][name], h5py.Dataset):
                s = self.fid[beam][name].dtype.str
                if self.fid[beam][name].ndim > 1:
                    t = self.fid[beam][name].shape[1:]
                    dtype_list.append((str(name), s, t))
                else:
                    dtype_list.append((str(name), s))

        num_records = finish - start
        data = numpy.empty(num_records, dtype=dtype_list)
        for item in dtype_list:
            name = item[0]
            if isinstance(self.fid[beam][name], h5py.Dataset):
                data[name] = self.fid[beam][name][start:finish,...]
            else:
                print('{} not found'.format(name))

        return data

    @staticmethod
    @njit
    def _waveform_1d_to_2d(start_indices, counts, data, out_data, start_offset=0):
        for i in prange(start_indices.shape[0]):
            for j in prange(counts[i]):
                out_data[j+start_offset, i] = data[start_indices[i] + j]
        return out_data

    def read_tx_waveform(self, beam, start=0, finish=None, minlength=None):
        if not finish:
            finish = self.fid[beam+'/tx_sample_start_index'].shape[0]

        start_indices = self.fid[beam+'/tx_sample_start_index'][start:finish] - 1
        counts = self.fid[beam+'/tx_sample_count'][start:finish]
        waveforms = self.fid[beam+'/txwaveform'][start_indices[0]:(start_indices[-1]+counts[-1])]

        max_count = numpy.max(counts)
        if minlength:
            max_count = max(minlength, max_count)
        out_shape = (max_count, counts.shape[0])

        out_waveforms = numpy.zeros(out_shape, dtype=waveforms.dtype)
        start_indices -= numpy.min(start_indices)
        out_waveforms = self._waveform_1d_to_2d(start_indices, counts, waveforms, out_waveforms)

        return out_waveforms

    def read_rx_waveform(self, beam, start=0, finish=None, minlength=None, elevation=False):
        if not finish:
            finish = self.fid[beam+'/rx_sample_start_index'].shape[0]

        start_indices = self.fid[beam+'/rx_sample_start_index'][start:finish] - 1
        counts = self.fid[beam+'/rx_sample_count'][start:finish]
        waveforms = self.fid[beam+'/rxwaveform'][start_indices[0]:(start_indices[-1]+counts[-1])]

        max_count = numpy.max(counts)
        if minlength:
            max_count = max(minlength, max_count)
        out_shape = (max_count, counts.shape[0])

        out_waveforms = numpy.zeros(out_shape, dtype=waveforms.dtype)
        start_indices -= numpy.min(start_indices)
        out_waveforms = self._waveform_1d_to_2d(start_indices, counts, waveforms, out_waveforms)

        if elevation:
            elev_bin0 = self.fid[beam+'/geolocation/elevation_bin0'][start:finish]
            elev_lastbin = self.fid[beam+'/geolocation/elevation_lastbin'][start:finish]
            v = (elev_bin0 - elev_lastbin) / (counts - 1)

            bin_dist = numpy.expand_dims(numpy.arange(max_count), axis=1)
            out_elevation = (numpy.expand_dims(elev_bin0, axis=0) -
                numpy.repeat(bin_dist,v.shape[0],axis=1) * v)

            return out_waveforms, out_elevation
        else:
            return out_waveforms

    def read_pgap_theta_z(self, beam, start=0, finish=None, minlength=None,
                          height=False, start_offset=0):
        """
        Remap the 1D pgap_theta_z array to a 2D M x N array, where M is
        the number Pgap profiles bins and N is the number of GEDI shots

        Parameters
        ----------
        beam: str
            Name of beam assessed, i.e. 'BEAM0001'
        start: int
            start of np.ndarray like slicing, Default=0
        finish: int/ None
            end of np.ndarray like slicing, Default=None
        minlength int:
            Minimum value for M
            Default is the maximum pgap_theta_z array length
        height: bool
            Return the M x N array the same size of out_pgap_profile
            with the height above ground of each profile bin
            Default is False
        start_offset: int
            Offset the start each profile in out_pgap_profile by
            start_offset bins. These bins are filled with ones.
            Default = 0

        Returns
        -------
        out_pgap_profile: numpy.ndarray
            2D numpy.ndarray of read pgap_theta_z data
        out_height:
            2D numpy.ndarray of read height above ground data
        """
        if not finish:
            finish = len(self.fid[beam+'/rx_sample_start_index'])
        else:
            pass

        start_indices = self.fid[beam+'/rx_sample_start_index'][start:finish] - 1
        counts = self.fid[beam+'/rx_sample_count'][start:finish]
        pgap_profile = self.fid[beam+'/pgap_theta_z'][start_indices[0]:(start_indices[-1]+counts[-1])]

        max_count = numpy.max(counts) + start_offset
        if minlength:
            max_count = max(minlength, max_count)
        out_shape = (max_count, counts.shape[0])

        pgap = self.fid[beam+'/pgap_theta'][start:finish]
        out_pgap_profile = numpy.broadcast_to(pgap, (out_shape)).copy()
        out_pgap_profile[0:start_offset,:] = 1.0

        start_indices -= numpy.min(start_indices)
        out_pgap_profile = self._waveform_1d_to_2d(start_indices, counts, pgap_profile,
                               out_pgap_profile, start_offset=start_offset)

        if height:
            height_bin0 = self.fid[beam+'/geolocation/height_bin0'][start:finish]
            height_lastbin = self.fid[beam+'/geolocation/height_lastbin'][start:finish]
            v = (height_bin0 - height_lastbin) / (counts - 1)

            bin_dist = numpy.expand_dims(numpy.arange(max_count), axis=1)
            out_height = (numpy.expand_dims(height_bin0, axis=0) -
                numpy.repeat(bin_dist,v.shape[0],axis=1) * v +
                start_offset * v)

            return out_pgap_profile, out_height
        else:
            return out_pgap_profile

    def copy_attrs(self, output_fid, group):
        for key in self.fid[group].attrs.keys():
            if key not in output_fid[group].attrs.keys():
                output_fid[group].attrs[key] = self.fid[group].attrs[key]

    def export_shots(self, beam, subset, dataset_list=[], product_id=None):
        # Get the group information
        group = self.fid[beam]
        nshots = group['shot_number'].shape[0]
        if not product_id:
            product_id = self.get_product_id()

        # Find indices to extract
        if subset is not None:
            idx_extract = userfunctions.get_geom_indices(group, product_id, subset)

            # Use h5py simple indexing - faster
            if not numpy.any(idx_extract):
                return
            tmp, = numpy.nonzero(idx_extract)
            idx_start = numpy.min(tmp)
            idx_finish = numpy.max(tmp) + 1
            idx_subset = tmp - idx_start
        else:
            idx_start = 0
            idx_finish = self.get_nrecords(beam)
            idx_subset = None

        # Function to extract datasets for selected shots
        def _get_selected_shots(name, obj):
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
                    if idx_subset is not None:
                        df = pandas.DataFrame(data=arr[idx_subset,...], columns=colnames)
                    else:
                        df = pandas.DataFrame(data=arr, columns=colnames)
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
            _get_selected_shots(out_name, group[name])
        outdata = pandas.concat(datasets, axis=1)

        return outdata

    def copy_shots(self, output_fid, beam, subset, output_2d, geom=False,
                   dataset_list=[], product_id=None):
        group = self.fid[beam]
        nshots = group['shot_number'].size
        if not product_id:
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

        def _copy_selected_shots(name, obj):
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
                            append_to_h5_dataset(name, out_group,
                                                 tmp[idx_subset,...],
                                                 shot_axis=shot_axis)
                        else:
                            tmp = obj[...,idx_start:idx_finish]
                            append_to_h5_dataset(name, out_group, tmp[...,idx_subset],
                                                 shot_axis=shot_axis)
                    else:
                        # ancillary / short_term datasets
                        append_to_h5_dataset(name, out_group, obj)

        if len(dataset_list) > 0:
            for name in dataset_list:
                _copy_selected_shots(name, group[name])
        else:
            group.visititems(copy_selected_shots)

        # Copy waveforms
        if product_id in ('1A','1B'):
            if len(dataset_list) == 0 or 'txwaveform' in dataset_list:
                self._copy_waveforms(group, 'txwaveform', out_group,
                                     'tx', idx_start, idx_finish,
                                     idx_subset, output_2d)
            if len(dataset_list) == 0 or 'rxwaveform' in dataset_list:
                self._copy_waveforms(group, 'rxwaveform', out_group,
                                     'rx', idx_start, idx_finish,
                                     idx_subset, output_2d)
        elif product_id == '2B':
            if len(dataset_list) == 0 or 'pgap_theta_z' in dataset_list:
                self._copy_waveforms(group, 'pgap_theta_z', out_group,
                                     'rx', idx_start, idx_finish,
                                     idx_subset, output_2d)

    def _copy_waveforms(self, in_group, waveform_name, out_group,
                        prefix, idx_start, idx_finish, idx_subset, output_2d):
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
            append_to_h5_dataset(waveform_name, out_group, out_waveforms, shot_axis=1)
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
            append_to_h5_dataset(waveform_name, out_group, out_waveforms)
            append_to_h5_dataset(start_indices_name, out_group, out_start_indices)
            append_to_h5_dataset(counts_name, out_group, counts)

    def get_quality_flag(self, beam, product_id=None, night=False, power=False, **kwargs):
        if not product_id:
            product_id = self.get_product_id()

        quality_name = GEDIPY_REFERENCE_DATASETS[product_id]['quality']
        if quality_name:
            quality_flag = (self.fid[beam][quality_name][()] == 1)
            if 'degrade' in kwargs:
                if  kwargs['degrade']:
                    degrade_name = GEDIPY_REFERENCE_DATASETS[product_id]['degrade']
                    degrade_flag = (self.fid[beam][degrade_name][()] == 0)
                    quality_flag &= degrade_flag
            if 'sensitivity' in kwargs:
                beam_sensitivity = self.fid[beam+'/sensitivity'][()]
                quality_flag &= (beam_sensitivity >= kwargs['sensitivity'])
            if 'nonull' in kwargs:
                name = kwargs['nonull']
                index = kwargs.get('index')
                if index:
                    tmp = self.fid[beam][name][:,index]
                else:
                    tmp = self.fid[beam][name][()]
                quality_flag &= (tmp != -9999)
        else:
            quality_flag = numpy.ones(self.get_nrecords(beam), dtype=numpy.bool)
        
        solar_elevation = GEDIPY_REFERENCE_DATASETS[product_id]['solar_elevation']
        if solar_elevation:
            quality_flag &= (self.fid[beam][solar_elevation][()] < 0)

        if power:
            quality_flag &= (self.fid[beam+'/beam'][()] > 3)

        return quality_flag

    def get_utc_time(self, beam):
        delta_time = self.fid[beam][GEDIPY_REFERENCE_COORDS[self.get_product_id()]['t']][()]
        start_utc = datetime.datetime(2018, 1, 1)
        utc_time = [start_utc + datetime.timedelta(seconds=s) for s in delta_time]
        return numpy.array(utc_time)

    def get_coordinates(self, beam, product_id=None):
        if not product_id:
            product_id = self.get_product_id()
        longitude = self.fid[beam][GEDIPY_REFERENCE_COORDS[product_id]['x']][()]
        latitude = self.fid[beam][GEDIPY_REFERENCE_COORDS[product_id]['y']][()]
        return longitude, latitude

    def get_dataset(self, beam, name, index=None):
        if self.fid[beam][name].ndim > 1:
            if index:
                dataset = self.fid[beam][name][:,index]
            else:
                dataset = self.fid[beam][name][()]
        else:
            dataset = self.fid[beam][name][()]
        return dataset

    @staticmethod
    @njit
    def _select_mode(mean_noise, modelocs, modeamp, localen, iwave, botloc, back_threshold,
                     nummodes, maxamp, elevs_allmodes, selected_mode, mode_flag, delta_z, pulse_sep_thresh,
                     cumulative_energy_minimum, energy_thresh, amp_thresh, cumulative_energy_thresh,
                     botlocdist_limit1, ampval_limit2, botlocdist_limit2, ampval_limit3, botlocdist_limit3):
        for i in range(selected_mode.shape[0]):
            elev_lowestmode = elevs_allmodes[i,selected_mode[i]]
            for j in range(nummodes[i]-1, -1, -1):
                # Don't bother with this mode if zcross and botloc are
                # within a laser pulse width of each other OR amp is <
                # thresh OR the cumulative waveform is < value
                if ( (botloc[i]-modelocs[i,j] < pulse_sep_thresh) |
                     (modeamp[i,j] < back_threshold[i]-mean_noise[i]) |
                     (iwave[i,j] < cumulative_energy_minimum) ):
                    continue
                if ( (localen[i,j] > energy_thresh) |
                     (modeamp[i,j] > amp_thresh) |
                     (iwave[i,j] > cumulative_energy_thresh) ):
                    # it's ok, quit moving left
                    selected_mode[i] = j
                    delta_z[i] = elevs_allmodes[i,j] - elev_lowestmode
                    mode_flag[i] = 3
                    break

            if (mode_flag[i] == 1):
                # Unless that point is too close to botloc, in which case
                # we're going to move 1 left
                for j in range(nummodes[i]-1, -1, -1):
                    if (modelocs[i,j]-botloc[i] > pulse_sep_thresh):
                        selected_mode[i] = j
                        delta_z[i] = elevs_allmodes[i,j] - elev_lowestmode
                        mode_flag[i] = 2
                        break

            # Case if selected mode is more than limitdist from botloc, going to flag it
            botloclimit = botlocdist_limit1
            zcross = modelocs[i,selected_mode[i]]
            if (maxamp[i] > ampval_limit2):
                botloclimit = botlocdist_limit2
            if (maxamp[i] > ampval_limit3):
                botloclimit = botlocdist_limit3
            if (botloc[i]-zcross > botloclimit):
                mode_flag[i] = 4

    def apply_mode_selection(self, beam, algorithm_setting, mode_opts=GEDIPY_MODE_SELECTION_OPTS):
        shot_idx, = numpy.nonzero( (self.fid[beam+'/rx_processing_a{:d}/rx_nummodes'.format(algorithm_setting)][()] > 1) &
                                   (self.fid[beam+'/rx_assess/rx_maxamp'][()] > 8*self.fid[beam+'/rx_assess/sd_corrected'][()]) )

        mean_noise = self.fid[beam+'/rx_processing_a{:d}/mean'.format(algorithm_setting)][()][shot_idx]
        modelocs = self.fid[beam+'/rx_processing_a{:d}/rx_modelocs'.format(algorithm_setting)][()][shot_idx,:]
        modeamp = self.fid[beam+'/rx_processing_a{:d}/rx_modeamps'.format(algorithm_setting)][()][shot_idx,:] - mean_noise[:,numpy.newaxis]
        localen = self.fid[beam+'/rx_processing_a{:d}/rx_modelocalenergy'.format(algorithm_setting)][()][shot_idx,:]
        iwave = self.fid[beam+'/rx_processing_a{:d}/rx_iwaveamps'.format(algorithm_setting)][()][shot_idx,:] * 100
        botloc = self.fid[beam+'/rx_processing_a{:d}/botloc'.format(algorithm_setting)][()][shot_idx]
        back_threshold = self.fid[beam+'/rx_processing_a{:d}/back_threshold'.format(algorithm_setting)][()][shot_idx]
        nummodes = self.fid[beam+'/rx_processing_a{:d}/rx_nummodes'.format(algorithm_setting)][()][shot_idx]
        maxamp = self.fid[beam+'/rx_assess/rx_maxamp'][()][shot_idx]
        elevs_allmodes = self.fid[beam+'/geolocation/elevs_allmodes_a{:d}'.format(algorithm_setting)][()][shot_idx,:]

        selected_mode = self.fid[beam+'/rx_processing_a{:d}/selected_mode'.format(algorithm_setting)][()]
        selected_mode_tmp = selected_mode[shot_idx]
        mode_flag_tmp = numpy.ones(selected_mode_tmp.shape[0], dtype=numpy.uint8)
        delta_z_tmp = numpy.zeros(selected_mode_tmp.shape[0], dtype=numpy.float32)

        self._select_mode(mean_noise, modelocs, modeamp, localen, iwave, botloc, back_threshold,
                         nummodes, maxamp, elevs_allmodes, selected_mode_tmp, mode_flag_tmp, delta_z_tmp,
                         mode_opts['pulse_sep_thresh'], mode_opts['cumulative_energy_minimum'],
                         mode_opts['energy_thresh'], mode_opts['amp_thresh'],
                         mode_opts['cumulative_energy_thresh'], mode_opts['botlocdist_limit1'],
                         mode_opts['ampval_limit2'], mode_opts['botlocdist_limit2'],
                         mode_opts['ampval_limit3'], mode_opts['botlocdist_limit3'])

        mode_flag = numpy.zeros(self.fid[beam+'/shot_number'].shape[0], dtype=numpy.uint8)
        delta_z = numpy.zeros(self.fid[beam+'/shot_number'].shape[0], dtype=numpy.float32)

        selected_mode[shot_idx] = selected_mode_tmp
        mode_flag[shot_idx] = mode_flag_tmp
        delta_z[shot_idx] = delta_z_tmp

        return selected_mode,mode_flag,delta_z

    def predict_algorithm_setting_selection(self, beam, pft, region, elev_lowestmode_a10, selected_mode_flag_a10):
        elev_a1 = self.fid[beam+'/geolocation/elev_lowestmode_a1'][()]
        elev_a2 = self.fid[beam+'/geolocation/elev_lowestmode_a2'][()]
        elev_a5 = self.fid[beam+'/geolocation/elev_lowestmode_a5'][()]
        rx_maxamp = self.fid[beam+'/rx_assess/rx_maxamp'][()]

        selected_algorithm = numpy.ones(elev_a1.shape[0], dtype=numpy.uint8)

        idx = (((pft == 2) | (pft == 1) | (pft == 3) | (pft == 4)) &
              (rx_maxamp < 400) & (elev_a2 < elev_a1))
        selected_algorithm[idx] = 2

        ebt_idx = (pft == 2) & ((region == 4) | (region == 5))

        z_flag = elev_lowestmode_a10 < (elev_a1-2)

        idx = ebt_idx & z_flag
        selected_algorithm[idx] = 10

        idx = ebt_idx & z_flag & (selected_mode_flag_a10 == 4)
        selected_algorithm[idx] = 5

        return selected_algorithm


class ATL03H5File(LidarFile):
    """
    Generic object for I/O of ICESat-2 ATL03 .h5 data

    Parameters
    ----------
    filename: str
        Pathname to ATL03 .h5 file

    """
    def __init__(self, filename):
        self.filename = filename
        self.filename_pattern = re.compile(r'(ATL\d{2})_(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})_(\d{4})(\d{2})(\d{2})_(\d{3})_(\d{2})\.h5')

    def is_valid(self):
        return h5py.is_hdf5(self.filename)

    def is_valid_filename(self):
        m = self.filename_pattern.fullmatch(os.path.basename(self.filename))
        if m:
            return True
        else:
            self.filename_pattern = re.compile(r'(ATL\d{2})_30m_(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})_(\d{4})(\d{2})(\d{2})_(\d{3})_(\d{2})\.h5')
            m = self.filename_pattern.fullmatch(os.path.basename(self.filename))
            if m:
                return True
            else:
                return False

    def get_product_id(self):
        m = self.filename_pattern.fullmatch(os.path.basename(self.filename))
        if m:
            return m.group(1)
        else:
            raise ValueError('invalid ATL03 filename: "{}"'.format(self.filename))

    def get_datetime(self):
        m = self.filename_pattern.fullmatch(os.path.basename(self.filename))
        if m:
            return datetime.datetime(m.group(2), m.group(3), m.group(4), m.group(5), m.group(6), m.group(7))
        else:
            raise ValueError('invalid ATL03 filename: "{}"'.format(self.filename))

    def get_rgt_number(self):
        m = self.filename_pattern.fullmatch(os.path.basename(self.filename))
        if m:
            return int(m.group(8))
        else:
            raise ValueError('invalid ATL03 filename: "{}"'.format(self.filename))

    def get_cycle_number(self):
        m = self.filename_pattern.fullmatch(os.path.basename(self.filename))
        if m:
            return int(m.group(9))
        else:
            raise ValueError('invalid ATL03 filename: "{}"'.format(self.filename))

    def get_segment_number(self):
        m = self.filename_pattern.fullmatch(os.path.basename(self.filename))
        if m:
            return int(m.group(10))
        else:
            raise ValueError('invalid ATL03 filename: "{}"'.format(self.filename))

    def get_version_number(self):
        m = self.filename_pattern.fullmatch(os.path.basename(self.filename))
        if m:
            return int(m.group(11))
        else:
            raise ValueError('invalid ATL03 filename: "{}"'.format(self.filename))

    def get_revision_number(self):
        m = self.filename_pattern.fullmatch(os.path.basename(self.filename))
        if m:
            return int(m.group(12))
        else:
            raise ValueError('invalid ATL03 filename: "{}"'.format(self.filename))

    def open_h5(self):
        self.fid = h5py.File(self.filename, 'r')
        if 'short_name' in self.fid.attrs:
            if self.fid.attrs['short_name'] != b'ATL03':
                raise GEDIPyDriverError
        else:
            raise GEDIPyDriverError
        self.beams = [beam for beam in self.fid if beam.startswith('gt')]

    def close_h5(self):
        self.fid.close()
        self.beams = None

    def get_orbit(self):
        orbit = self.fid['ancillary_data/start_orbit'][0]
        return int(orbit)

    def get_atlas_orientation(self):
        flags = {0: 'backward', 1: 'forward', 2: 'transition'}
        sc_orient = self.fid['orbit_info/sc_orient'][0]
        return flags[sc_orient]

    def get_orbit_number(self):
        m = self.filename_pattern.fullmatch(os.path.basename(self.filename))
        if m:
            return '{}{}{}'.format(m.group(7),m.group(8),m.group(9))
        else:
            raise ValueError('invalid ATL03 filename: "{}"'.format(self.filename))

    def get_quality_flag(self, beam, night=False, power=False, **kwargs):
        quality_flag = self.fid[beam+'/heights/signal_conf_ph'][:,0]
        
        if len(kwargs) > 0:
            for k in kwargs:
                name = '{}/{}'.format(beam,k)
                if name in self.fid:
                    dataset = self.fid[name][()]
                    quality_flag &= (dataset == kwargs[k])
                    if numpy.issubdtype(dataset.dtype, numpy.integer):
                        quality_flag &= (dataset < numpy.iinfo(dataset.dtype).max)
                    else:
                        quality_flag &= (dataset < numpy.finfo(dataset.dtype).max)
         
        if night:
            quality_flag &= (self.fid[beam+'/geolocation/solar_elevation'][()] < 0)

        if power:
            beam_type = self.fid[beam].attrs['atlas_beam_type'].decode('utf-8')
            if beam_type != 'strong':
                quality_flag &= False

        return quality_flag

    def get_coordinates(self, beam, ht=False):
        longitude = self.fid[beam+'/heights/lon_ph'][()]
        latitude = self.fid[beam+'/heights/lat_ph'][()]
        if ht:
            elevation = self.fid[beam+'/heights']['h_ph'][()]
            return longitude, latitude, elevation
        else:
            return longitude, latitude

    def get_nrecords(self, beam):
        nrecords = self.fid[beam+'/heights/delta_time'].shape[0]
        return nrecords

    def get_photon_labels(self, beam, atl08_fid):
        ph_index_beg = self.fid[beam+'/geolocation/ph_index_beg'][()]
        segment_id = self.fid[beam+'/geolocation/segment_id'][()]
        valid_idx = ph_index_beg > 0

        idx = numpy.searchsorted(segment_id[valid_idx],
            atl08_fid.get_photon_segment_id(beam), side='left')
        atl08_ph_index_beg = ph_index_beg[valid_idx][idx] - 1

        atl08_ph_index = atl08_fid.get_photon_index(beam)
        idx = atl08_ph_index_beg + atl08_fid.get_photon_index(beam)
        atl03_class = numpy.zeros(self.get_nrecords(beam), dtype=numpy.uint8)
        atl03_class[idx] = atl08_fid.get_photon_class(beam)

        return atl03_class

    def get_dataset(self, beam, name, index=None):
        if index:
            dataset = self.fid[beam][name][:,index]
        else:
            dataset = self.fid[beam][name][()]
        return dataset

    def export_shots(self, beam, subset, dataset_list=[]):
        # Get the group information
        group = self.fid[beam]
        nshots = group['heights/delta_time'].shape[0]

        # Find indices to extract
        if subset:
            product_id = self.get_product_id()
            idx_extract = userfunctions.get_geom_indices(group, product_id, subset)

            # Use h5py simple indexing - faster
            if not numpy.any(idx_extract):
                return
            tmp, = numpy.nonzero(idx_extract)
            idx_start = numpy.min(tmp)
            idx_finish = numpy.max(tmp) + 1
            idx_subset = tmp - idx_start
        else:
            idx_start = 0
            idx_finish = self.get_nrecords(beam)
            idx_subset = None

        # Function to extract datasets for selected shots
        def _get_selected_shots(name, obj):
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
                    if idx_subset is not None:
                        df = pandas.DataFrame(data=arr[idx_subset,...], columns=colnames)
                    else:
                        df = pandas.DataFrame(data=arr, columns=colnames)
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
            _get_selected_shots(out_name, group[name])
        outdata = pandas.concat(datasets, axis=1)

        return outdata


class ATL08H5File(LidarFile):
    """
    Generic object for I/O of ICESat-2 ATL08 .h5 data

    Parameters
    ----------
    filename: str
        Pathname to ATL08 .h5 file

    """
    def __init__(self, filename):
        self.filename = filename
        self.filename_pattern = re.compile(r'(ATL\d{2})_((?:30m_|))(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})_(\d{4})(\d{2})(\d{2})_(\d{3})_(\d{2})\.h5')

    def is_valid(self):
        return h5py.is_hdf5(self.filename)

    def is_valid_filename(self):
        m = self.filename_pattern.fullmatch(os.path.basename(self.filename))
        if m:
            return True
        else:
            return False

    def get_product_id(self):
        m = self.filename_pattern.fullmatch(os.path.basename(self.filename))
        if m:
            return m.group(1)
        else:
            raise ValueError('invalid ATL08 filename: "{}"'.format(self.filename))

    def get_datetime(self):
        m = self.filename_pattern.fullmatch(os.path.basename(self.filename))
        if m:
            return datetime.datetime(m.group(3), m.group(4), m.group(5), m.group(6), m.group(7), m.group(8))
        else:
            raise ValueError('invalid ATL08 filename: "{}"'.format(self.filename))

    def get_rgt_number(self):
        m = self.filename_pattern.fullmatch(os.path.basename(self.filename))
        if m:
            return int(m.group(9))
        else:
            raise ValueError('invalid ATL08 filename: "{}"'.format(self.filename))

    def get_cycle_number(self):
        m = self.filename_pattern.fullmatch(os.path.basename(self.filename))
        if m:
            return int(m.group(10))
        else:
            raise ValueError('invalid ATL08 filename: "{}"'.format(self.filename))

    def get_segment_number(self):
        m = self.filename_pattern.fullmatch(os.path.basename(self.filename))
        if m:
            return int(m.group(11))
        else:
            raise ValueError('invalid ATL08 filename: "{}"'.format(self.filename))

    def get_version_number(self):
        m = self.filename_pattern.fullmatch(os.path.basename(self.filename))
        if m:
            return int(m.group(12))
        else:
            raise ValueError('invalid ATL08 filename: "{}"'.format(self.filename))

    def get_revision_number(self):
        m = self.filename_pattern.fullmatch(os.path.basename(self.filename))
        if m:
            return int(m.group(13))
        else:
            raise ValueError('invalid ATL08 filename: "{}"'.format(self.filename))

    def open_h5(self):
        self.fid = h5py.File(self.filename, 'r')
        if 'short_name' in self.fid.attrs:
            if self.fid.attrs['short_name'] != b'ATL08':
                raise GEDIPyDriverError
            m = self.filename_pattern.fullmatch(os.path.basename(self.filename))
            if len(m.group(2)) > 0:
                print('Processing will use 30m segments')
                self.subgroup = 'land_segments/30m_segment'
            else:
                self.subgroup = 'land_segments'
        else:
            raise GEDIPyDriverError
        self.beams = [beam for beam in self.fid if beam.startswith('gt')]

    def close_h5(self):
        self.fid.close()
        self.beams = None

    def get_orbit(self):
        orbit = self.fid['ancillary_data/start_orbit'][0]
        return int(orbit)

    def get_orbit_number(self):
        m = self.filename_pattern.fullmatch(os.path.basename(self.filename))
        if m:
            return '{}{}{}'.format(m.group(7),m.group(8),m.group(9))
        else:
            raise ValueError('invalid ATL08 filename: "{}"'.format(self.filename))

    def get_photon_class(self, beam):
        photon_class = self.fid[beam+'/signal_photons/classed_pc_flag'][()]
        return photon_class

    def get_photon_index(self, beam):
        photon_index = self.fid[beam+'/signal_photons/classed_pc_indx'][()] - 1
        return photon_index

    def get_photon_segment_id(self, beam):
        segment_id = self.fid[beam+'/signal_photons/ph_segment_id'][()]
        return segment_id

    def get_coordinates(self, beam):
        longitude = self.fid[beam+'/'+self.subgroup+'/longitude'][()]
        latitude = self.fid[beam+'/'+self.subgroup+'/latitude'][()]
        return longitude, latitude

    def get_night_flag(self, beam):
        night_flag = (self.fid[beam+'/'+self.subgroup+'/night_flag'][()] == 1)
        return night_flag

    def get_quality_flag(self, beam, night=False, power=False, **kwargs):
        """
        Other quality flag options to consider:
        quality_flag = (self.fid[beam+'/'subgroup+'/msw_flag'][()] == 0)
        quality_flag &= (self.fid[beam+'/'subgroup+'/terrain_flg'][()] == 0)
        quality_flag &= (self.fid[beam+'/'subgroup+'/segment_watermask'][()] == 0)
        quality_flag &= (self.fid[beam+'/'subgroup+'/cloud_flag_atm'][()] == 0)
        quality_flag &= (self.fid[beam+'/'subgroup+'/dem_removal_flag'][()] == 0)
        quality_flag &= (self.fid[beam+'/'subgroup+'/ph_removal_flag'][()] == 0)
        """
        quality_flag = numpy.ones(self.fid[beam+'/'+self.subgroup+'/delta_time'].shape, dtype=numpy.bool)
        
        if 'nonull' in kwargs:
            if not isinstance(kwargs['nonull'], list):
                kwargs['nonull'] = list(kwargs['nonull'])
            for name in kwargs['nonull']:
                name = '{}/{}'.format(beam,kwargs['nonull'])
                if name in self.fid:
                    dataset = self.fid[name][()]
                    if numpy.issubdtype(dataset.dtype, numpy.integer):
                        quality_flag &= (dataset < numpy.iinfo(dataset.dtype).max)
                    else:
                        quality_flag &= (dataset < numpy.finfo(dataset.dtype).max)
        
        if night:
            quality_flag &= (self.fid[beam+'/'+self.subgroup+'/solar_elevation'][()] < 0)
        
        if power:
            beam_type = self.fid[beam].attrs['atlas_beam_type'].decode('utf-8')
            if beam_type != 'strong':
                quality_flag &= False
        
        return quality_flag

    def get_dataset(self, beam, name, index=None):
        if index:
            dataset = self.fid[beam][name][:,index]
        else:
            dataset = self.fid[beam][name][()]
        return dataset

    def export_shots(self, beam, subset, dataset_list=[], **kwargs):
        # Get the group information
        group = self.fid[beam]
        nshots = group[self.subgroup+'/delta_time'].shape[0]

        # Find indices to extract
        if subset:
            product_id = self.get_product_id()
            idx_extract = userfunctions.get_geom_indices(group, product_id, subset)

            # Use h5py simple indexing - faster
            if not numpy.any(idx_extract):
                return
            tmp, = numpy.nonzero(idx_extract)
            idx_start = numpy.min(tmp)
            idx_finish = numpy.max(tmp) + 1
            idx_subset = tmp - idx_start
        else:
            idx_start = 0
            idx_finish = self.get_nrecords(beam)
            idx_subset = None

        # Function to extract datasets for selected shots
        def _get_selected_shots(name, obj):
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
                    if idx_subset is not None:
                        df = pandas.DataFrame(data=arr[idx_subset,...], columns=colnames)
                    else:
                        df = pandas.DataFrame(data=arr, columns=colnames)
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
            _get_selected_shots(out_name, group[name])
        outdata = pandas.concat(datasets, axis=1)

        return outdata


class LVISH5File(LidarFile):
    """
    Generic object for I/O of LVIS .h5 data

    Parameters
    ----------
    filename: str
        Pathname to LVIS .h5 file

    """
    def __init__(self, filename):
        self.filename = filename
        self.filename_pattern = re.compile(r'LVIS(C|F)(\d{1})(A|B)_()_(\d{4})_R(\d{4})_(\d{6})_\.h5')

    def is_valid(self):
        return h5py.is_hdf5(self.filename)

    def is_valid_filename(self):
        m = self.filename_pattern.fullmatch(os.path.basename(self.filename))
        if m:
            return True
        else:
            return False

    def get_product_id(self):
        m = self.filename_pattern.fullmatch(os.path.basename(self.filename))
        if m:
            return 'LVIS{}{:d}{}'.format( m.group(1), m.group(2), m.group(3) )
        else:
            raise ValueError('invalid LVIS filename: "{}"'.format(self.filename))

    def get_nrecords(self):
        nshots = self.fid['SHOTNUMBER'].shape[0]
        return nshots

    def open_h5(self, short_name=None):
        self.fid = h5py.File(self.filename, 'r')
        lvis_product_names = ('L1B HDF','L2B HDF')
        if not short_name:
            if 'short_name' in self.fid.attrs:
                short_name = self.fid.attrs['short_name'][0].decode('utf-8')
            else:
                raise GEDIPyDriverError
        if short_name not in lvis_product_names:
            raise GEDIPyDriverError
        val, cnt = numpy.unique(self.fid['LFID'], return_counts=True)
        self.lfid = list(val)
        self.counts = list(cnt)

    def close_h5(self):
        self.fid.close()
        self.lfid = None
        self.counts = None

    def read_shots(self, start=0, finish=None, dataset_list=[]):
        """
        Read data of LVIS .h5 files into numpy.ndarray

        Parameters
        ----------
        start: int
            start of np.ndarray like slicing, Default=0
        finish: int/ None
            end of np.ndarray like slicing, Default=None
        dataset_list: list of str
            List of LVIS h5 dataset paths

        Returns
        -------
        data: numpy.ndarray
            numpy.ndarray of read data
        """
        if not finish:
            finish = self.get_nrecords()

        dtype_list = []
        for name in dataset_list:
            if isinstance(self.fid[name], h5py.Dataset):
                s = self.fid[name].dtype.str
                if self.fid[name].ndim > 1:
                    t = self.fid[name].shape[1:]
                    dtype_list.append((str(name), s, t))
                else:
                    dtype_list.append((str(name), s))

        num_records = finish - start
        data = numpy.empty(num_records, dtype=dtype_list)
        for item in dtype_list:
            name = item[0]
            if isinstance(self.fid[name], h5py.Dataset):
                data[name] = self.fid[name][start:finish,...]
            else:
                print('{} not found'.format(name))

        return data

    def read_tx_waveform(self, start=0, finish=None, minlength=None):
        if not finish:
            finish = self.get_nrecords()

        out_waveforms = self.fid['TXWAVE'][start:finish,:]

        if minlength:
            if minlength > self.fid['TXWAVE'].shape[1]:
                out_shape = (finish - start, minlength)
                out_waveforms = numpy.broadcast_to(out_waveforms, (out_shape)).copy()

        return out_waveforms

    def read_rx_waveform(self, start=0, finish=None, minlength=None, elevation=False):
        if not finish:
            finish = self.get_nrecords()

        out_waveforms = self.fid['RXWAVE'][start:finish,:]

        if minlength:
            if minlength > self.fid['RXWAVE'].shape[1]:
                out_shape = (finish - start, minlength)
                out_waveforms = numpy.broadcast_to(out_waveforms, (out_shape)).copy()

        if elevation:
            elev_bin0 = self.fid['Z0'][start:finish]
            elev_lastbin = self.fid['Z1023'][start:finish]
            v = (elev_bin0 - elev_lastbin) / (out_waveforms.shape[1] - 1)

            bin_dist = numpy.expand_dims(numpy.arange(out_waveforms.shape[1]), axis=1)
            out_elevation = (numpy.expand_dims(elev_bin0, axis=0) -
                numpy.repeat(bin_dist,v.shape[0],axis=1) * v)

            return out_waveforms, out_elevation
        else:
            return out_waveforms

    def copy_attrs(self, output_fid, group):
        for key in self.fid[group].attrs.keys():
            if key not in output_fid[group].attrs.keys():
                output_fid[group].attrs[key] = self.fid[group].attrs[key]

    def export_shots(self, subset, dataset_list=[]):
        # Find indices to extract
        if subset:
            product_id = self.get_product_id()
            idx_extract = userfunctions.get_geom_indices(self.fid, product_id, subset)

            # Use h5py simple indexing - faster
            if not numpy.any(idx_extract):
                return
            tmp, = numpy.nonzero(idx_extract)
            idx_start = numpy.min(tmp)
            idx_finish = numpy.max(tmp) + 1
            idx_subset = tmp - idx_start
        else:
            idx_start = 0
            idx_finish = self.get_nrecords()
            idx_subset = None

        # Function to extract datasets for selected shots
        nshots = self.get_nrecords()
        def _get_selected_shots(name, obj):
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
                    if idx_subset is not None:
                        df = pandas.DataFrame(data=arr[idx_subset,...], columns=colnames)
                    else:
                        df = pandas.DataFrame(data=arr, columns=colnames)
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
            _get_selected_shots(out_name, self.fid[name])
        outdata = pandas.concat(datasets, axis=1)

        return outdata

    def copy_shots(self, output_fid, subset, geom=False, dataset_list=[]):
        nshots = self.get_nrecords()
        product_id = self.get_product_id()

        # Find indices to extract
        if geom:
            idx_extract = userfunctions.get_geom_indices(self.fid, product_id, subset)
        else:
            shot_numbers = self.fid['SHOTNUMBER'][()]
            idx_extract = userfunctions.get_shot_indices(subset, shot_numbers)

        # Use h5py simple indexing - faster
        if not numpy.any(idx_extract):
            return
        tmp, = numpy.nonzero(idx_extract)
        idx_start = numpy.min(tmp)
        idx_finish = numpy.max(tmp) + 1
        idx_subset = tmp - idx_start

        def _copy_selected_shots(name, obj):
            if isinstance(obj, h5py.Group):
                if name not in output_fid:
                    output_fid.create_group(name)
            elif isinstance(obj, h5py.Dataset):
                # Copy selected shot numbers for most datasets
                if 0 not in obj.shape:
                    if nshots in obj.shape:
                        shot_axis = obj.shape.index(nshots)
                        if shot_axis == 0:
                            tmp = obj[idx_start:idx_finish,...]
                            append_to_h5_dataset(name, output_fid,
                                                 tmp[idx_subset,...],
                                                 shot_axis=shot_axis)
                        else:
                            tmp = obj[...,idx_start:idx_finish]
                            append_to_h5_dataset(name, output_fid, tmp[...,idx_subset],
                                                 shot_axis=shot_axis)
                    else:
                        # ancillary / short_term datasets
                        append_to_h5_dataset(name, output_fid, obj)

        if len(dataset_list) > 0:
            for name in dataset_list:
                _copy_selected_shots(name, self.fid[name])
        else:
            self.fid.visititems(copy_selected_shots)

    def get_coordinates(self):
        longitude = self.fid[GEDIPY_REFERENCE_COORDS[self.get_product_id()]['x']][()]
        latitude = self.fid[GEDIPY_REFERENCE_COORDS[self.get_product_id()]['y']][()]
        return longitude, latitude

    def get_dataset(self, name, index=None):
        if self.fid[name].ndim > 1:
            if index:
                dataset = self.fid[name][:,index]
            else:
                dataset = self.fid[name][()]
        else:
            dataset = self.fid[name][()]
        return dataset
