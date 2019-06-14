import numpy as np
from numpy import sqrt, exp, pi, square
import time

import pandas as pd

from scipy.interpolate import interp1d
from scipy.stats import norm, poisson

from numba import int64, float64, guvectorize, jit
from itp_map import InterpolatingMap

from pax import units, utils, datastructure
import logging
log = logging.getLogger('SimulationCore')


import pax
import os

PAX_DIR = pax.__file__.replace('__init__.py', '')


def data_file_path(filename):
    """Returns filename if a file exists there, else returns PAX_DIR/data/filename"""
    if os.path.exists(filename):
        return filename
    new_filename = os.path.join(PAX_DIR, 'data', filename)
    if os.path.exists(new_filename):
        return new_filename
    else:
        raise ValueError('File name or path %s not found!' % filename)


# Pulse building class
class Pulse(object):

    def __init__(self, config):
        self.config = config

        self.init_pmt_current_templates()  # PMT + digitizer response to incident photon
        self.init_spe_scaling_factor_distributions()
        self.clear_pulse_cache()

        self.turned_off_pmts = np.arange(len(config['gains']))[np.array(config['gains']) == 0]

    def __call__(self):
        """
        PMTs' response to incident photons
        Use _photon_timings, _photon_channels to build pulses
        """
        if ('_photon_timings' not in self.__dict__) or \
           ('_photon_channels' not in self.__dict__):
            raise NotImplementedError

        # Correct for PMT Transition Time Spread
        self._photon_timings += np.random.normal(self.config['pmt_transit_time_mean'],
                                                 self.config['pmt_transit_time_spread'],
                                                 len(self._photon_timings))

        for channel in np.unique(self._photon_channels):
            _channel_photon_timings = self._photon_timings[self._photon_channels == channel]

            # Compute sample index（quotient）and reminder
            # Determined by division between photon timing and sample duraiton.
            dt = self.config.get('sample_duration', 10)
            _channel_photon_reminders = (_channel_photon_timings -
                                         np.floor(_channel_photon_timings/dt).astype(int)*dt).astype(int)
            _channel_photon_timings = np.floor(_channel_photon_timings/dt).astype(int)

            # If gain of each photon is not specifically assigned
            # Sample from spe scaling factor distribution and to individual gain
            if '_photon_gains' not in self.__dict__:
                _channel_photon_gains = self.config['gains'][channel] \
                    * self.uniform_to_pe_arr[channel](np.random.random(len(_channel_photon_timings)))

                # Effectively adding double photoelectron emission by doubling gain
                n_double_pe = np.random.binomial(len(_channel_photon_timings),
                                                 p=self.config['p_double_pe_emision'])
                _dpe_index = np.random.choice(np.arange(len(_channel_photon_timings)),
                                              size=n_double_pe, replace=False)
                _channel_photon_gains[_dpe_index] += self.config['gains'][channel] \
                    * self.uniform_to_pe_arr[channel](np.random.random(n_double_pe))
            else:
                _channel_photon_gains = self._photon_gains[self._photon_channels == channel]

            # Build a simulated waveform, length depends on min and max of photon timings
            min_timing, max_timing = np.min(
                _channel_photon_timings), np.max(_channel_photon_timings)
            pulse_left = min_timing - int(self.config['samples_before_pulse_center'])
            pulse_right = max_timing + int(self.config['samples_after_pulse_center'])
            pulse_current = np.zeros(pulse_right - pulse_left + 1)

            Pulse.add_current(_channel_photon_timings - min_timing,
                              _channel_photon_reminders, _channel_photon_gains, pulse_current)

            # For single event, data of pulse level is small enough to store in dataframe
            self._pulses.append(dict(channel=channel,
                                     left=pulse_left,
                                     right=pulse_right,
                                     duration=pulse_right-pulse_left+1,
                                     current=pulse_current))

    def init_pmt_current_templates(self):
        """
        Create spe templates, for 10ns sample duration and 1ns rounding we have:
        _pmt_current_templates[i] : photon timing fall between [10*m+i, 10*m+i+1)
        (i, m are integers)
        """
        # 'Hidden' global variables to avoid passing class instance into guvectorized function
        global _pmt_current_templates, _template_length
        _pmt_current_templates = []
        _template_length = 0

        # Interpolate on cdf ensures that each spe pulse would sum up to 1 pe*sample duration^-1
        pe_pulse_function = interp1d(
            self.config.get('pe_pulse_ts'),
            np.cumsum(self.config.get('pe_pulse_ys')),
            bounds_error=False, fill_value=(0, 1))

        # Samples are always multiples of sample_duration
        sample_duration = self.config.get('sample_duration', 10)
        samples_before = self.config.get('samples_before_pulse_center', 2)
        samples_after = self.config.get('samples_after_pulse_center', 20)
        pmt_pulse_time_rounding = self.config.get('pmt_pulse_time_rounding', 1.0)

        samples = np.linspace(-samples_before * sample_duration,
                              + samples_after * sample_duration,
                              1 + samples_before + samples_after)
        _template_length = len(samples) - 1

        for r in np.arange(0, sample_duration, pmt_pulse_time_rounding):
            pmt_current = np.diff(pe_pulse_function(samples - r)) / sample_duration  # pe / 10 ns
            # Normalize here to counter tiny rounding error from interpolation
            pmt_current *= (1 / sample_duration) / np.sum(pmt_current)  # pe / 10 ns
            _pmt_current_templates.append(pmt_current)

        log.debug('Create spe waveform templates with %s ns resolution' % pmt_pulse_time_rounding)

    def init_spe_scaling_factor_distributions(self):
        # Extract the spe pdf from a csv file into a pandas dataframe
        spe_shapes = pd.read_csv(data_file_path(self.config['photon_area_distribution']))

        # Create a converter array from uniform random numbers to SPE gains (one interpolator per channel)
        # Scale the distributions so that they have an SPE mean of 1 and then calculate the cdf
        uniform_to_pe_arr = []
        for ch in spe_shapes.columns[1:]:  # skip the first element which is the 'charge' header
            if spe_shapes[ch].sum() > 0:
                mean_spe = (spe_shapes['charge'] * spe_shapes[ch]).sum() / spe_shapes[ch].sum()
                scaled_bins = spe_shapes['charge'] / mean_spe
                cdf = np.cumsum(spe_shapes[ch]) / np.sum(spe_shapes[ch])
            else:
                # if sum is 0, just make some dummy axes to pass to interpolator
                cdf = np.linspace(0, 1, 10)
                scaled_bins = np.zeros_like(cdf)

            uniform_to_pe_arr.append(interp1d(cdf, scaled_bins))
        if uniform_to_pe_arr != []:
            self.uniform_to_pe_arr = np.array(uniform_to_pe_arr)

        log.debug('Initialize spe scaling factor distributions')

    def clear_pulse_cache(self):
        self._pulses = []

    @staticmethod  # Staticmethod decorator doesn't seem to do anything
    @guvectorize([(int64, int64, float64, float64[:])], '(), (), (), (n)')
    def add_current(_photon_timing_start, _reminder, _photon_gain, pulse):
        """
        Simulate single channel waveform given the photon timings
        _photon_timing_start   - dim-1 integer array of photon timings in unit of samples
        _reminder              - dim-1 integer array of complimentary photon timings
        _photon_gain           - dim-1 float array of ph. 2 el. gain individual photons
        _pulse                 - waveform
        _pmt_current_templates - list of spe templates of different reminders
        The self argument is intentionally left out of this function.
        """
        pulse[_photon_timing_start:_photon_timing_start+_template_length] += \
            _pmt_current_templates[_reminder] * _photon_gain

    def singlet_triplet_delays(self, size, singlet_ratio):
        """
        Given the amount of the eximer, return time between excimer decay 
        and their time of generation.
        size           - amount of eximer
        self.phase     - 'liquid' or 'gas'
        singlet_ratio  - fraction of excimers that become singlets
                         (NOT the ratio of singlets/triplets!)
        """
        if self.phase == 'liquid':
            t1, t3 = (self.config['singlet_lifetime_liquid'],
                      self.config['triplet_lifetime_liquid'])
        elif self.phase == 'gas':
            t1, t3 = (self.config['singlet_lifetime_gas'],
                      self.config['triplet_lifetime_gas'])

        delay = np.random.choice([t1, t3], size, replace=True,
                                 p=[singlet_ratio, 1-singlet_ratio])
        return np.random.exponential(1, size) * delay


class S1(Pulse):
    """
    Given temperal inputs as well as number of photons
    Random generate photon timing and channel distribution.
    """

    def __init__(self, config):
        super().__init__(config)
        self.s1_light_yield_map = InterpolatingMap(
            utils.data_file_name(self.config['s1_light_yield_map']))
        self.s1_pattern_map = InterpolatingMap(
            utils.data_file_name(self.config['s1_patterns_file']))
        self.phase = 'liquid'  # To distinguish singlet/triplet time delay.

    def __call__(self, instruction):
        _, _, t, x, y, z, n_photons, recoil_type = instruction  # temporary solution for instruction passing
        ly = self.s1_light_yield_map([[x, y, z]]) * self.config['s1_detection_efficiency']
        n_photons = np.random.binomial(n=n_photons, p=ly)

        self.photon_timings(t, n_photons, recoil_type)
        # The new way iterpolation is written always require a list
        self.photon_channels([[x, y, z]])

        super().__call__()

    def photon_channels(self, points):
        if len(self._photon_timings) == 0:
            self._photon_channels = []
            return 0

        channels = np.array(self.config['channels_in_detector']['tpc'])
        p_per_channel = self.s1_pattern_map(points)[0]
        p_per_channel[np.in1d(channels, self.turned_off_pmts)] = 0
        p_per_channel /= np.sum(p_per_channel)

        self._photon_channels = np.random.choice(
            channels,
            size=len(self._photon_timings),
            p=p_per_channel,
            replace=True)

    def photon_timings(self, t, n_photons, recoil_type):
        if n_photons == 0:
            self._photon_timings = np.array([])
            return 0
        try:
            self._photon_timings = t + getattr(self, recoil_type.lower())(n_photons)
        except AttributeError:
            raise AttributeError('Recoil type must be ER, NR, alpha or LED, not %s' % recoil_type)

    def alpha(self, size):
        # Neglible recombination time
        return self.singlet_triplet_delays(size, self.config['s1_ER_alpha_singlet_fraction'])

    def led(self, size):
        # distribute photons uniformly within the LED pulse length
        return np.random.uniform(0, self.config['led_pulse_length'], size)

    def er(self, size):
        # How many of these are primary excimers? Others arise through recombination.
        efield = (self.config['drift_field'] / (units.V / units.cm))
        self.config['s1_ER_recombination_time'] = 3.5 / \
            0.18 * (1 / 20 + 0.41) * np.exp(-0.009 * efield)

        reco_time, p_fraction, max_reco_time = (
            self.config['s1_ER_recombination_time'],
            self.config['s1_ER_primary_singlet_fraction'],
            self.config['maximum_recombination_time'])

        timings = np.random.choice([0, reco_time], size, replace=True,
                                   p=[p_fraction, 1-p_fraction])
        primary = timings == 0
        timings *= 1 / (1-np.random.uniform(0, 1, size)) - 1
        timings = np.clip(timings, 0, self.config['maximum_recombination_time'])
        size_primary = len(timings[primary])
        timings[primary] += self.singlet_triplet_delays(
            size_primary, self.config['s1_ER_primary_singlet_fraction'])
        timings[~primary] += self.singlet_triplet_delays(
            size-size_primary, self.config['s1_ER_secondary_singlet_fraction'])
        return timings

    def nr(self, size):
        return self.singlet_triplet_delays(size, self.config['s1_NR_singlet_fraction'])


class S2(Pulse):
    """
    Given temperal inputs as well as number of electrons
    Random generate photon timing and channel distribution.
    """

    def __init__(self, config):
        super().__init__(config)

        self.s2_light_yield_map = InterpolatingMap(
            utils.data_file_name(self.config['s2_light_yield_map']))
        self.s2_pattern_map = InterpolatingMap(
            utils.data_file_name(self.config['s2_patterns_file']))
        self.phase = 'gas'  # To distinguish singlet/triplet time delay.
        self.luminescence_switch_threshold = 100  # More then those electrons use simplified luminescence model

    def __call__(self, instruction):
        if isinstance(instruction, list) or len(instruction.shape) < 2:
            instruction = np.array([instruction])

        _, _, t, x, y, z, n_electron, recoil_type = [
            np.array(v).reshape(-1) for v in zip(*instruction)]

        poistions = np.array([x, y]).T  # For map interpolation
        sc_gain = self.s2_light_yield_map(poistions) * self.config['s2_secondary_sc_gain']

        # Average drift time of the electrons
        self.drift_time_mean = - z / \
            self.config['drift_velocity_liquid'] + self.config['drift_time_gate']

        # Absorb electrons during the drift
        electron_lifetime_correction = np.exp(- 1 * self.drift_time_mean /
                                              self.config['electron_lifetime_liquid'])
        cy = self.config['electron_extraction_yield'] * electron_lifetime_correction
        n_electron = np.random.binomial(n=list(n_electron), p=cy)

        # Second generate photon timing and channel
        self.photon_timings(t, n_electron, z, sc_gain)
        self.photon_channels(poistions)

        super().__call__()

    def luminescence_timings(self, shape):
        """
        Luminescence time distribution computation
        """
        number_density_gas = self.config['pressure'] / \
            (units.boltzmannConstant * self.config['temperature'])
        alpha = self.config['gas_drift_velocity_slope'] / number_density_gas

        dG = self.config['elr_gas_gap_length']
        rA = self.config['anode_field_domination_distance']
        rW = self.config['anode_wire_radius']
        dL = self.config['gate_to_anode_distance'] - dG

        VG = self.config['anode_voltage'] / (1 + dL/dG/self.config['lxe_dielectric_constant'])
        E0 = VG/((dG - rA)/rA + np.log(rA/rW))

        def Efield_r(r): return np.clip(E0/r, E0/rA, E0/rW)

        def velosity_r(r): return alpha * Efield_r(r)

        def Yield_r(r): return Efield_r(r) / (units.kV / units.cm) - \
            0.8 * self.config['pressure'] / units.bar

        r = np.linspace(dG, rW, 1000)
        dt = - np.diff(r)[0] / velosity_r(r)
        dy = Yield_r(r)/np.sum(Yield_r(r))

        uniform_to_emission_time = interp1d(np.cumsum(dy), np.cumsum(dt),
                                            bounds_error=False, fill_value=(0, sum(dt)))

        probabilities = 1 - np.random.uniform(0, 1, size=shape)
        return uniform_to_emission_time(probabilities)

    def electron_timings(self, t, n_electron, z, sc_gain):

        # Diffusion model from Sorensen 2011
        drift_time_mean = - z / \
            self.config['drift_velocity_liquid'] + self.config['drift_time_gate']
        drift_time_stdev = np.sqrt(2 * self.config['diffusion_constant_liquid'] * drift_time_mean)
        drift_time_stdev /= self.config['drift_velocity_liquid']

        # Calculate electron arrival times in the ELR region
        _electron_timings = t + \
            np.random.exponential(self.config['electron_trapping_time'], n_electron)
        if drift_time_stdev:
            _electron_timings += np.random.normal(drift_time_mean, drift_time_stdev, n_electron)

        self._electron_timings = np.append(self._electron_timings, _electron_timings)
        self._electron_gains = np.append(
            self._electron_gains, np.repeat(sc_gain, len(_electron_timings)))

    def photon_timings(self, t, n_electron, z, sc_gain):
        # First generate electron timing
        self._electron_timings = np.array([])
        self._electron_gains = np.array([])
        list(map(self.electron_timings, t, n_electron, z, sc_gain))

        # TODO log this
        if len(self._electron_timings) < 1:
            self._photon_timings = []
            return 1

        # For vectorized calculation, artificially top #photon per electron at +4 sigma
        nele = len(self._electron_timings)
        npho = np.ceil(np.max(self._electron_gains) +
                       4 * np.sqrt(np.max(self._electron_gains))).astype(int)

        self._photon_timings = self.luminescence_timings((nele, npho))
        self._photon_timings += np.repeat(self._electron_timings, npho).reshape((nele, npho))

        # Crop number of photons by random number generated with poisson
        probability = np.tile(np.arange(npho), nele).reshape((nele, npho))
        threshold = np.repeat(np.random.poisson(self._electron_gains), npho).reshape((nele, npho))
        self._photon_timings = self._photon_timings[probability < threshold]

        # Special index for match photon to original electron poistion
        self._instruction = np.repeat(
            np.repeat(np.arange(len(t)), n_electron), npho).reshape((nele, npho))
        self._instruction = self._instruction[probability < threshold]

        self._photon_timings += self.singlet_triplet_delays(
            len(self._photon_timings), self.config['singlet_fraction_gas'])

    def photon_channels(self, points):
        # TODO log this
        if len(self._photon_timings) == 0:
            self._photon_channels = []
            return 1

        # Probability of each top channel given area fraction top
        p_top = self.config['s2_mean_area_fraction_top']
        # A fraction of photons are given uniformally to top pmts regardless of pattern
        p_random = self.config.get('randomize_fraction_of_s2_top_array_photons', 0)
        # Use pattern to get top channel probability
        # p_pattern = self.s2_pattern_map(points)
        p_pattern = self.s2_pattern_map_pp(points)

        # Probability of each bottom channels
        p_per_channel_bottom = (1-p_top) / len(self.config['channels_bottom']) \
            * np.ones_like(self.config['channels_bottom'])

        # Randomly assign to channel given probability of each channel
        # Sum probabilities over channels should be 1
        self._photon_channels = np.array([])
        for u, n in zip(*np.unique(self._instruction, return_counts=True)):
            p_per_channel_top = p_pattern[u] / np.sum(p_pattern[u]) * p_top * (1 - p_random)
            channels = np.array(self.config['channels_in_detector']['tpc'])
            p_per_channel = np.concatenate([p_per_channel_top, p_per_channel_bottom])
            # p_per_channel[np.in1d(channels, self.turned_off_pmts)] = 0

            _photon_channels = np.random.choice(
                channels,
                size=n,
                p=p_per_channel/np.sum(p_per_channel),
                replace=True)

            self._photon_channels = np.append(self._photon_channels, _photon_channels)

        self._photon_channels = self._photon_channels.astype(int)

    def s2_pattern_map_pp(self, pos):
        if 'params' not in self.__dict__:
            all_map_params = pd.read_pickle(
                '/project2/lgrandi/zhut/Kr83m_Ddriven_per_pmt_params_dataframe.pkl')
            self.params = all_map_params.loc[all_map_params.kr_run_id == 10,
                                             ['amp0', 'amp1', 'tao0', 'tao1', 'pmtx', 'pmty']].values.T

        amp0, amp1, tao0, tao1, pmtx, pmty = self.params

        def rms(x, y): return np.sqrt(np.square(x) + np.square(y))

        distance = rms(pmtx - pos[:, 0].reshape([-1, 1]), pmty - pos[:, 1].reshape([-1, 1]))
        frac = amp0 * np.exp(- distance / tao0) + amp1 * np.exp(- distance / tao1)
        frac = (frac.T / np.sum(frac, axis=-1)).T

        return frac


class Afterpulse_Electron(S2):
    """
    Produce electron after pulse simulation, using already built cdfs
    The cdfs follow distribution parameters extracted from data.
    """

    def __init__(self, config):
        super().__init__(config)
        self.uniform_to_ele_ap = np.load(
            '/project2/lgrandi/zhut/sim/WFSimDev/ele_after_pulse.npy').item()
        self.element_list = ['liquid']
        self._photon_timings = []

    def __call__(self, signal_pulse):
        if len(signal_pulse._photon_timings) == 0:
            return

        self.electron_afterpulse(signal_pulse)

        if len(self.inst) < 1:
            return
        super().__call__(self.inst.reindex(
            ['event_number', 'type', 't', 'x', 'y', 'z', 'amp', 'recoil'], axis=1).values)

    def electron_afterpulse(self, signal_pulse):
        """
        For electron afterpulses we assume a uniform x, y
        """
        now = time.time()

        delaytime_cdf = self.uniform_to_ele_ap['liquid']['delaytime_cdf']

        # To save calculation we first find out how many photon will give rise ap
        n_electron = np.random.poisson(delaytime_cdf[-1] * len(signal_pulse._photon_timings))

        # Assign each electron a random uniform number rU0 (0, delaytime_cdf[-1]]
        rU0 = (1 - np.random.uniform(size=n_electron)) * delaytime_cdf[-1]

        # The map is made so that the indices are delay time in unit of ns
        # ap_delay = np.argmin(
        #    np.abs(delaytime_cdf - rU0[:, None]), axis=-1)

        # 80,0000 is bit too slow, so we lower it to 800 + 1000
        ap_delay = np.argmin(
            np.abs(delaytime_cdf[500::1000] - rU0[:, None]), axis=-1)
        ap_delay = ap_delay * 1000 + \
            np.argmin(
                np.abs(delaytime_cdf[ap_delay[:, None]*1000 + np.arange(1000)] - rU0[:, None]), axis=-1)

        # Randomly select original photon as time zeros
        t_zeros = signal_pulse._photon_timings[np.random.randint(
            low=0, high=len(signal_pulse._photon_timings),
            size=n_electron)]

        self.inst = pd.DataFrame()
        self.inst['type'] = 's2'
        self.inst['t'] = t_zeros
        self.inst['x'], self.inst['y'] = self._randomize_XY(n_electron)
        self.inst['z'] = - (ap_delay - self.config['drift_time_gate']) * \
            self.config['drift_velocity_liquid']
        self.inst['amp'] = 1

    def _randomize_XY(self, n):
        Rupper = 46

        r = np.sqrt(np.random.uniform(0, Rupper*Rupper, n))
        angle = np.random.uniform(-np.pi, np.pi, n)

        x = r * np.cos(angle)
        y = r * np.sin(angle)

        return x, y


class Afterpulse_PMT(Pulse):
    """
    Produce pmt after pulse simulation, using already built cdfs
    The cdfs follow distribution parameters extracted from data.
    """

    def __init__(self, config):
        super().__init__(config)
        self.uniform_to_pmt_ap = np.load(
            '/project2/lgrandi/zhut/sim/WFSimDev/pmt_after_pulse.npy').item()
        self.element_list = ['Uniform', 'Ar', 'CH4', 'He', 'N2', 'Ne', 'Xe', 'Xe2+']

    def __call__(self, signal_pulse):
        if len(signal_pulse._photon_timings) == 0:
            return

        self._photon_timings = []
        self._photon_channels = []
        self._photon_amplitude = []

        self.photon_afterpulse(signal_pulse)
        super().__call__()

    def photon_afterpulse(self, signal_pulse):
        """
        For pmt afterpulses, gain and dpe generation is a bit different from standard photons
        """
        for element in self.element_list:
            delaytime_cdf = self.uniform_to_pmt_ap[element]['delaytime_cdf']
            amplitude_cdf = self.uniform_to_pmt_ap[element]['delaytime_cdf']

            # Assign each photon two random uniform number rU0, rU1 from (0, 1]
            rU0 = 1 - np.random.uniform(size=len(signal_pulse._photon_timings))
            rU1 = 1 - np.random.uniform(size=len(signal_pulse._photon_timings))

            # Select those photons with U <= max of cdf of specific channel
            sel_photon_id = np.where(rU0 <= delaytime_cdf[signal_pulse._photon_channels, -1])[0]
            sel_photon_channel = signal_pulse._photon_channels[sel_photon_id]

            # The map is made so that the indices are delay time in unit of ns
            ap_delay = np.argmin(
                np.abs(
                    delaytime_cdf[sel_photon_channel]
                    - rU0[sel_photon_id][:, None]), axis=-1)

            if element == 'Uniform':
                ap_amplitude = np.ones_like(ap_delay)
            else:
                ap_amplitude = np.argmin(
                    np.abs(
                        amplitude_cdf[sel_photon_channel]
                        - rU1[sel_photon_id][:, None]), axis=-1)/100.

            self._photon_timings += list(signal_pulse._photon_timings[sel_photon_id] + ap_delay)
            self._photon_channels += list(signal_pulse._photon_channels[sel_photon_id])
            self._photon_amplitude += list(ap_amplitude)

        self._photon_timings = np.array(self._photon_timings)
        self._photon_channels = np.array(self._photon_channels).astype(int)
        self._photon_amplitude = np.array(self._photon_amplitude)
        self._photon_gain = np.array(self.config['gains'])[self._photon_channels] \
            * self._photon_amplitude


class Event(object):

    def __init__(self, config):
        self.config = config

        self.pulses = dict(
            s1=S1(config),
            s2=S2(config),
            ele_ap=Afterpulse_Electron(config),
            pmt_ap=Afterpulse_PMT(config))

        self.ptypes = self.pulses.keys()

    def __call__(self, instructions):
        for ptype in self.ptypes:
            self.pulses[ptype].clear_pulse_cache()

        for instruction in instructions:
            # Instruction must be
            # [id, type, t, x, y, z, amp, recoil type]
            # [0, 's1', 1000000.0, 0, 0, 0, 10000, 'ER']
            ptype = instruction[1]
            self.pulses[ptype](instruction)
            self.pulses['ele_ap'](self.pulses[ptype])

            self.pulses['pmt_ap'](self.pulses[ptype])
            self.pulses['pmt_ap'](self.pulses['ele_ap'])

        self._pulses = []
        for ptype in self.ptypes:
            self._pulses += getattr(self.pulses[ptype], '_pulses', [])
        self._pulses = pd.DataFrame(self._pulses)

        self.raw_data()

    def raw_data(self):
        if len(self._pulses) > 0:
            self.event_duration = event_duration = max(self._pulses.right.max()+1, 200000)

            self.current_2_adc = self.config['pmt_circuit_load_resistor'] \
                * self.config['external_amplification'] \
                / (self.config['digitizer_voltage_range'] / 2 ** (self.config['digitizer_bits']))

            # Use noise array to pave the fundation of the pulses
            #self._raw_data = self.get_real_noise(np.array([0]), np.array([event_duration]))[0]
            self._raw_data = np.zeros((event_duration, len(
                self.config['channels_in_detector']['tpc'])))*1.0

            for ix, _pulse in self._pulses.iterrows():
                # Could round instead of trunc... no one cares!
                adc_wave = - np.trunc(_pulse.current * self.current_2_adc)
                self._raw_data[_pulse.left:_pulse.right+1, _pulse.channel] += adc_wave

            # Digitizers have finite number of bits per channel, so clip the signal.
            self._raw_data += self.config['digitizer_reference_baseline']
            self._raw_data = np.clip(self._raw_data, 0, 2 ** (self.config['digitizer_bits']))

    def get_real_noise(self, lefts, durations):
        """
        Get chunk(s) of noise sample from real noise data
        """
        try:
            self.noise_files
        except AttributeError:
            self.noise_files = np.genfromtxt(self.config['noise_file_index'], dtype='<U32')

        # Randomly choose where in to start copying
        real_data_sample_size = 100000
        template_duration = len(self.noise_files) * real_data_sample_size
        output_duration = np.max(lefts+durations) - np.min(lefts)
        roll = np.random.randint(0, template_duration-output_duration)

        # Find the indicies in noise template files to copy
        result = np.zeros((np.sum(durations), len(self.config['channels_in_detector']['tpc'])))
        id_s = np.arange(len(result))
        id_t = roll+np.concatenate([np.arange(lefts[ix], lefts[ix]+durations[ix])
                                    for ix in range(len(lefts))])
        id_t, sub_id_t = np.floor_divide(
            id_t, real_data_sample_size), np.mod(id_t, real_data_sample_size)

        for unique_id_t in np.unique(id_t):
            mask = id_t == unique_id_t
            data = np.load(self.config['noise_file_folder']+'/%s.npz' %
                           (self.noise_files[unique_id_t]))['arr_0']
            result[id_s[mask], :] = data[sub_id_t[mask].astype(int)]

        return np.split(result, np.cumsum(durations)[:-1])


class Simulator(object):

    def __init__(self, config):
        self.config = config
        self.event = Event(config)

    def __call__(self, master_instruction):
        if not isinstance(master_instruction, pd.DataFrame):
            print('Reqire a dataframe see:')

        m_inst = master_instruction

        for event_number in m_inst.event_number.unique():
            # Build event ingradients
            e_inst = m_inst[m_inst.event_number == event_number]
            e_inst = e_inst.loc[:, ['event_number', 'type',
                                    't', 'x', 'y', 'z', 'amp', 'recoil']].values
            # print(e_inst)
            self.event(e_inst)

            start_time = int(time.time() * units.s)
            start_index = 0

            # Make proper datastructure
            event = datastructure.Event(n_channels=self.config['n_channels'],
                                        start_time=start_time,
                                        stop_time=start_time + self.event.event_duration *
                                        self.config['sample_duration'],  # in ns
                                        sample_duration=self.config['sample_duration'])

            for channel in self.event._pulses.channel.unique():
                event.pulses.append(datastructure.Pulse(
                    channel=channel,
                    left=start_index,
                    raw_data=self.event._raw_data[:, channel].astype(np.int16)))

            event.event_number = event_number
            yield event
