import numpy
import linecache
import kmpfit
from . import h5io
from . import signal


class Canopy:
    """Parent class for the canopy module"""
    def __init__(self, filename):
        self.filename = filename

    def load_lvis_l2(self, fn, dataset_list=None):
        line = linecache.getline(fn, 20)
        colnames = line[2:].strip().split()
        outdata = pandas.read_table(fn, header=None, skiprows=20, names=colnames,
            low_memory=True, delim_whitespace=True, usecols=dataset_list)
        return outdata.reset_index(drop=True)


class WaveformProfile(Canopy):
    def __init__(self, l1b_filename, l2a_filename):
        self.l1b_fid = LidarFile(l1b_filename)
        self.l2a_fid = LidarFile(l2a_filename)

    def set_algorithm_run_flag(self, beam):
        """
        Determine which shots to run the algorithms on
        """
        self.algorithm_run_flag = None

    def normalize_tx(self, beam, tx_noise_std_filter_n=4):
        """
        Return a normalized txwave symmetrically centered on peak
        using TX_NOISE_STD_FILTER_N*std_noise level to aggressively remove noise
        """
        l1b_data = self.l1b_fid.read_shots(beam, dataset_list=['tx_egbias','tx_egbias_error'])
        tx_high_noise_threshold = l1b_data['tx_egbias'] + tx_noise_std_filter_n * l1b_data['tx_egbias_error']

        tx = self.l1b_fid.read_tx_waveform(beam)
        tx = numpy.clip(tx - tx_high_noise_threshold, 0, None)

        peakloc = numpy.argmax(tx, axis=0)
        peakloc_shift = -(peakloc - tx.shape[0] / 2).astype(numpy.int16)

        row_idx, col = numpy.ogrid[:tx.shape[0],:tx.shape[1]]
        peakloc_shift[peakloc_shift < 0] += tx.shape[0]
        row_idx = row_idx - peakloc_shift
        self.tx_norm = tx[row_idx,col]

    @staticfunction
    @jit(nopython=True)
    def matchfilter_ground(self, rx, tx, zcross, rx_conv, rg, valid):
        """
        Use a matchfilter and mirror the below ground component to estimate ground return energy
        """
        for i in range(rx.shape[1]):
            if valid[i]:
                tmp = numpy.zeros(rx.shape[0])
                convolve_numba(rx[:,i], tx[:,i], tmp)
                for j in range(rx.shape[0]):
                    rx_conv[j,i] = tmp[j]
                for j in range(rx.shape[0]):
                    if (j < zcross[i]):
                        k = int(zcross[i] + (zcross[i] - j))
                        if k < rx.shape[0]:
                            rg[j,i] = rx_conv[k,i]
                    elif (j >= zcross[i]):
                        rg[j,i] = rx_conv[j,i]
            else:
                for j in range(rx.shape[0]):
                    rx_conv[j,i] = 0
                    rg[j,i] = 0

    def unmix_rx_alg2(self, tx_noise_std_filter_n=4):
        """
        Convolve the rx with a flipped TX and then mirror the belowground component to
        the aboveground part
        """
        tx_norm = self.normalize_tx(tx_noise_std_filter_n=tx_noise_std_filter_n)
        tx_norm_r = numpy.flip(tx_norm, axis=0)

        zcross = self.l2a_fid.get_dataset(beam, 'zcross', rx_processing=True)

        rx_idx = numpy.arange(self.rx.shape[1], dtype=numpy.float32)
        rx_conv = numpy.zeros(self.rx.shape, dtype=numpy.float32)
        rg = numpy.zeros(self.rx.shape, dtype=numpy.float32)
        matchfilter_ground(self.rx, tx_norm_r, zcross, rx_conv, rg, self.algorithm_run_flag)

        self.rg = rg
        self.rv = rx_conv - rg
        canopy = (rx_idx < zcross) & (self.rv > 0)
        self.rv[~canopy] = 0

    def unmix_rx_alg1(self, beam, ground_constraint_buff=4):
        """
        Unmix ground and return energy using an extended Gaussian fit to the ground return
        """
        # Read the input data
        l1b_data = self.l1b_fid.read_shots(beam, dataset_list=['tx_egsigma','tx_egsigma_error',
            'tx_eggamma','tx_eggamma_error'])
        rx_idx = numpy.arange(rx.shape[1], dtype=numpy.float32)
        zcross = self.l2a_fid.get_dataset(beam, 'zcross', rx_processing=True)

        # Prepare the mask, starting parameters and bounds
        rg_mask = rx_idx > (zcross - ground_constraint_buff * 2) # pad bins above ground center
        est_g_amp = numpy.max(self.rx, axis=0) # using maximum value at ~ground return range
        egloc_lower = zcross - ground_constraint_buff
        egloc_upper = zcross + ground_constraint_buff
        gwidth_lower = l1b_data['tx_egsigma'] - 3 * l1b_data['tx_egsigma_error']
        eggamma_lower = l1b_data['tx_eggamma'] - 3 * l1b_data['tx_eggamma_error']
        eggamma_upper = l1b_data['tx_eggamma'] + 3 * l1b_data['tx_eggamma_error']

        # Process each valid shot
        self.rg = numpy.zeros(self.rx.shape, dtype=numpy.float32)
        valid_idx, = numpy.nonzero(self.algorithm_run_flag)
        for i in valid_idx:
            p = (est_g_amp[i],zcross[i],l1b_data['tx_egsigma'][i],l1b_data['tx_eggamma'][i])
            bounds = ([-numpy.inf,egloc_lower[i],egwidth_lower[i],eggamma_lower[i]],
                      [numpy.inf,egloc_upper[i],numpy.inf,eggamma_upper[i]])
            result = least_squares(expgaussian, p, bounds=bounds,
                                   args=(rx_idx[rg_mask[:,i],i], self.rx[rg_mask[:,i],i]),
                                   max_nfev=100)

            self.l2b_data['rg_eg_flag'][i] = result.status
            if result.success:
                self.l2b_data['rg_eg_niter'][i] = result.nfev
                self.l2b_data['rg_eg_amplitude'][i] = result.x[0]
                self.l2b_data['rg_eg_center'][i] = result.x[1]
                self.l2b_data['rg_eg_sigma'][i] = result.x[2]
                self.l2b_data['rg_eg_gamma'][i] = result.x[3]
                self.l2b_data['rg_eg_cost'][i] = result.cost
                self.rg[:,i] = eval_expgaussian(result.x, rx_idx[:,i])

        self.rg[self.rg <= numpy.finfo(numpy.float32).min] = 0
        self.rv = self.rx - self.rg
        canopy = (rx_idx < zcross) & (self.rv > 0)
        self.rv[~canopy] = 0

    def get_rx_signal(self, beam, signal_search_buff=10):
        """
        Preprocess the rx signal to remove noise
        """
        toploc = self.l2a_fid.get_dataset(beam, 'toploc', rx_processing=True)
        botloc = self.l2a_fid.get_dataset(beam, 'botloc', rx_processing=True)

        rx = self.l1b_fid.read_rx_waveform(beam)
        l1b_data = self.l1b_fid.read_shots(beam, dataset_list=['noise_mean_corrected','rx_sample_count'])

        vegloc_start = numpy.clip(toploc - signal_search_buff, 0, rx_sample_count - 1)
        gloc_end = numpy.clip(botloc + signal_search_buff, 0, rx_sample_count - 1)

        rx_bins = numpy.arange(rx.shape[1], dtype=numpy.float32)
        signal_mask = (rx_bins > vegloc_start) & (rx_bins < gloc_end)

        self.rx = numpy.clip(rx - noise_mean_corrected, 0, None)
        self.rx[~signal_mask] = 0.0

    def get_canopy_top_range(self, beam):
        """
        Get range to top of canopy
        """
        toploc = self.l2a_fid.get_dataset(beam, 'toploc', rx_processing=True)
        l1b_data = self.l1b_fid.read_shots(beam, dataset_list=['geolocation/neutat_delay_total_bin0',
            'geolocation/neutat_delay_total_lastbin','geolocation/bounce_time_offset_bin0',
            'geolocation/bounce_time_offset_lastbin','rx_sample_count'])

        w = toploc / (l1b_data['rx_sample_count'] - 1)
        neutat_delay_total_toploc = l1b_data['neutat_delay_total_bin0'] * (1 - w) + l1b_data['neutat_delay_total_lastbin'] * w
        bounce_time_offset_toploc = l1b_data['bounce_time_offset_bin0'] * (1 - w) + l1b_data['bounce_time_offset_lastbin'] * w
        rx_range_highestreturn = bounce_time_offset_toploc * constants.c - neutat_delay_total_toploc

        return rx_range_highestreturn

    def get_pgap_theta(self, rhov, rhog):
        """
        Derive the directional gap probability profile
        Once error estimates for rhov and rhog are available,
        the bootstrap code to propogate the errors will be available here
        """
        rv_cum = numpy.ma.cumsum(self.rv, axis=0)
        rv = numpy.sum(self.rv, axis=0)
        rg = numpy.sum(self.rg, axis=0)

        self.pgap_theta_z = numpy.ones(rg.shape, numpy.float32)
        run_pgap = ( (rv > 0) | (rg > 0) ) & self.algorithm_run_flag

        rhov_rhog = rhov / rhog
        self.pgap_theta_z[run_pgap,:] = ( 1.0 - rv_cum[run_pgap,:] /
            (rg[run_pgap] * rhov_rhog[run_pgap] + rv[run_pgap]) )
        pgap_theta = numpy.min(self.pgap_theta_z, axis=1)

        return pgap_theta

    def calculate_fhd(self, pavd, normalize=True):
        """
        Calculate FHD for a given vertical PAVD profile
        """
        if normalize:
            pavd_tot = numpy.sum(pavd, axis=1)
            pavd_gt0 = pavd_tot > 0
            pavd[pavd_gt0] = pavd[pavd_gt0] / numpy.expand_dims(pavd_tot[pavd_gt0], axis=1)

        b = numpy.log(pavd)
        t = pavd * b

        fhd = -numpy.sum(t, axis=1)

        return fhd

    @staticfunction
    @jit(nopython=True)
    def extract_dz_vals(self, profiles, height, dz_vals, max_val, profile_vals, valid, null_value):
        """
        Extract the values of profile at dz height increments
        Fix (2019-11-17): Ensured the height bin with the dz value is included
        and do not clip height values at zero (ground bin height val may be negative)
        """
        for i in range(profiles.shape[1]):
            if valid[i]:
                for j in range(dz_vals.shape[0]):
                    tmp = height[:,i] - dz_vals[j]
                    idx = numpy.argmin(tmp > 0)
                    val = height[idx,i]
                    if val >= max_val:
                        profile_vals[i,j] = null_value
                    else:
                        profile_vals[i,j] = profiles[idx,i]
            else:
                for j in range(dz_vals.shape[0]):
                    profile_vals[i,j] = null_value

    def calculate_L2Bmetrics(self, dz=1.0, maxheight=60, rossg=0.5, omega=1.0):
        """
        Calculate L2B metrics
        """
        l1b_data = self.l1b_fid.read_shots(beam, dataset_list=['geolocation/local_beam_elevation'])

        dz_vals = numpy.arange(maxheight / dz) * dz
        max_dz_val = dz_vals[-1] + dz
        cos_zenith = numpy.abs(numpy.sin(l1b_data['geolocation/local_beam_elevation']))

        cover_z = cos_zenith * (1.0 - self.gap_theta_z)
        cover_z_r = numpy.fill((cover_z.shape[1], dz_vals.shape[0]), self.null_value, dtype=numpy.float32)
        extract_dz_vals(cover_z, height, dz_vals, max_dz_val, cover_z_r)

        pai_z = -(1.0 / (rossg * omega)) * numpy.log(self.pgap_theta_z) * cos_zenith
        pai_z_r = numpy.fill((pai_z.shape[1], dz_vals.shape[0]), self.null_value, dtype=numpy.float32)
        extract_dz_vals(pai_z, self.height, dz_vals, max_dz_val, pai_z_r)

        cover = numpy.max(cover_z[:,valid], axis=0)
        pai = numpy.max(pai_z[:,valid], axis=0)
        pavd_z = -numpy.gradient(pai_z_r, dz_vals, axis=1)
        fhd_normal = calculate_fhd(pavd_z)
