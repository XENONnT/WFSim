import logging
from numba import njit
import numpy as np
from scipy.interpolate import interp1d
from strax import exporter, deterministic_hash
from ..load_resource import load_config


export, __all__ = exporter()
logging.basicConfig(handlers=[logging.StreamHandler()])
log = logging.getLogger('wfsim.core')
log.setLevel('WARNING')


__all__ += ['_cached_pmt_current_templates', '_cached_uniform_to_pe_arr']
_cached_pmt_current_templates = {}
_cached_uniform_to_pe_arr = {}


@export
class Pulse(object):
    """Pulse building class"""

    def __init__(self, config):
        self.config = config
        self.config.update(self.config.get(self.__class__.__name__, {}))
        self.resource = load_config(config)

        self.init_pmt_current_templates()
        self.init_spe_scaling_factor_distributions()
        self.config['turned_off_pmts'] = np.arange(len(config['gains']))[np.array(config['gains']) == 0]
        self.current_max = np.max(np.array(self._pmt_current_templates), axis=1)
        self.current_2_adc = self.config['pmt_circuit_load_resistor'] \
            * self.config['external_amplification'] \
            / (self.config['digitizer_voltage_range'] / 2 ** (self.config['digitizer_bits']))

        self.clear_pulse_cache()

    def __call__(self, *args):
        """
        PMTs' response to incident photons
        Use _photon_timings, _photon_channels to build pulses
        """
        if ('_photon_timings' not in self.__dict__) or \
                ('_photon_channels' not in self.__dict__):
            raise NotImplementedError

        # The pulse cache should be immediately transferred after call this function
        self.clear_pulse_cache()

        # Correct for PMT Transition Time Spread (skip for pmt after-pulses)
        # note that PMT datasheet provides FWHM TTS, so sigma = TTS/(2*sqrt(2*log(2)))=TTS/2.35482
        if '_photon_gains' not in self.__dict__:
            self._photon_timings += np.random.normal(self.config['pmt_transit_time_mean'],
                                                     self.config['pmt_transit_time_spread'] / 2.35482,
                                                     len(self._photon_timings)).astype(np.int64)

        dt = self.config.get('sample_duration', 10)  # Getting dt from the lib just once
        self._n_photon = self._n_photon_bottom = 0  # For truth output
        self._n_pe = self._n_pe_bottom = 0
        self._n_photon_trigger = self._n_photon_trigger_bottom = 0
        self._n_pe_trigger = self._n_pe_trigger_bottom = 0
        self._raw_area = self._raw_area_bottom = 0
        self._raw_area_trigger = self._raw_area_trigger_bottom = 0

        # Assign DPE, and save it for compute after-pulses
        p_dpe = self.config['p_double_pe_emision']
        self._photon_is_dpe = np.random.binomial(n=1,
                                                p=p_dpe,
                                                size=len(self._photon_timings)).astype(np.bool_)

        counts_start = 0  # Secondary loop index for assigning channel
        for channel, counts in zip(*np.unique(self._photon_channels, return_counts=True)):

            # Use 'counts' amount of photon for this channel
            _channel_photon_timings = self._photon_timings[counts_start:counts_start+counts]
            _channel_photon_is_dpe = self._photon_is_dpe[counts_start:counts_start+counts]

            counts_start += counts
            if channel in self.config['turned_off_pmts']:
                continue

            # If gain of each photon is not specifically assigned
            # Sample from spe scaling factor distribution and to individual gain
            # In contrast to pmt afterpulse that should have gain determined before this step
            if '_photon_gains' not in self.__dict__:

                _channel_photon_gains = self.config['gains'][channel] \
                    * self.uniform_to_pe_arr(np.random.random(len(_channel_photon_timings)), channel)

                # Add some double photoelectron emission by adding another sampled gain
                n_double_pe = _channel_photon_is_dpe.sum()
                _channel_photon_gains[_channel_photon_is_dpe] += self.config['gains'][channel] \
                    * self.uniform_to_pe_arr(np.random.random(n_double_pe), channel)

            else:
                n_double_pe = 0
                _channel_photon_gains = np.array(self._photon_gains[self._photon_channels == channel])

            # += truth per channel to internal truth fields
            self.add_truth(
                _channel_photon_timings,
                _channel_photon_gains,
                dt,
                channel,
                n_double_pe,)

            # Build a simulated waveform, length depends on min and max of photon timings
            min_timing, max_timing = np.min(
                _channel_photon_timings), np.max(_channel_photon_timings)

            pulse_left = (int(min_timing // dt) 
                          - int(self.config['samples_to_store_before'])
                          - self.config.get('samples_before_pulse_center', 2))

            pulse_right = (int(max_timing // dt) 
                           + int(self.config['samples_to_store_after'])
                           + self.config.get('samples_after_pulse_center', 20))
            pulse_current = np.zeros(pulse_right - pulse_left + 1)

            Pulse.add_current(_channel_photon_timings.astype(np.int64),
                              _channel_photon_gains,
                              pulse_left,
                              dt,
                              self._pmt_current_templates,
                              pulse_current)

            # For single event, data of pulse level is small enough to store in dataframe
            self._pulses.append(dict(
                photons  = len(_channel_photon_timings),
                channel  = channel,
                left     = pulse_left,
                right    = pulse_right,
                duration = pulse_right - pulse_left + 1,
                current  = pulse_current,))

    def init_pmt_current_templates(self):
        """
        Create spe templates, for 10ns sample duration and 1ns rounding we have:
        _pmt_current_templates[i] : photon timing fall between [10*m+i, 10*m+i+1)
        (i, m are integers)
        """
        h = deterministic_hash(self.config)
        if h in _cached_pmt_current_templates:
            self._pmt_current_templates = _cached_pmt_current_templates[h]
            return

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

        # Let's fix this, so everything can be turned into int
        assert pmt_pulse_time_rounding == 1

        samples = np.linspace(-samples_before * sample_duration,
                              + samples_after * sample_duration,
                              1 + samples_before + samples_after)
        self._template_length = np.int64(len(samples) - 1)

        templates = []
        for r in np.arange(0, sample_duration, pmt_pulse_time_rounding):
            pmt_current = np.diff(pe_pulse_function(samples - r)) / sample_duration  # pe / 10 ns
            # Normalize here to counter tiny rounding error from interpolation
            pmt_current *= (1 / sample_duration) / np.sum(pmt_current)  # pe / 10 ns
            templates.append(pmt_current)
        self._pmt_current_templates = np.array(templates)
        _cached_pmt_current_templates[h] = self._pmt_current_templates

        log.debug('Spe waveform templates created with %s ns resolution, cached with key %s'
                  % (pmt_pulse_time_rounding, h))

    def init_spe_scaling_factor_distributions(self):
        h = deterministic_hash(self.config)
        if h in _cached_uniform_to_pe_arr:
            self.__uniform_to_pe_arr = _cached_uniform_to_pe_arr[h]
            return

        # Extract the spe pdf from a csv file into a pandas dataframe
        spe_shapes = self.resource.photon_area_distribution

        # Create a converter array from uniform random numbers to SPE gains (one interpolator per channel)
        # Scale the distributions so that they have an SPE mean of 1 and then calculate the cdf
        uniform_to_pe_arr = []
        for ch in spe_shapes.columns[1:]:  # skip the first element which is the 'charge' header
            if spe_shapes[ch].sum() > 0:
                # mean_spe = (spe_shapes['charge'].values * spe_shapes[ch]).sum() / spe_shapes[ch].sum()
                scaled_bins = spe_shapes['charge'].values  # / mean_spe
                cdf = np.cumsum(spe_shapes[ch]) / np.sum(spe_shapes[ch])
            else:
                # if sum is 0, just make some dummy axes to pass to interpolator
                cdf = np.linspace(0, 1, 10)
                scaled_bins = np.zeros_like(cdf)

            grid_cdf = np.linspace(0, 1, 2001)
            grid_scale = interp1d(cdf, scaled_bins,
                                  kind='next',
                                  bounds_error=False,
                                  fill_value=(scaled_bins[0], scaled_bins[-1]))(grid_cdf)

            uniform_to_pe_arr.append(grid_scale)

        if len(uniform_to_pe_arr):
            self.__uniform_to_pe_arr = np.stack(uniform_to_pe_arr)
            _cached_uniform_to_pe_arr[h] = self.__uniform_to_pe_arr

        log.debug('Spe scaling factors created, cached with key %s' % h)

    def uniform_to_pe_arr(self, p, channel=0):
        indices = (p * 2000).astype(np.int64) + 1
        return self.__uniform_to_pe_arr[channel, indices]

    def add_truth(self,
                  photon_timings,
                  photon_gains,
                  dt,
                  channel,
                  n_double_pe,
                  ):
        """Add required information to the fields used for the truth information"""
        if channel in self.config['turned_off_pmts']:
            return

        if str(channel) in self.config.get('special_thresholds', {}):
            threshold = self.config['special_thresholds'][str(channel)] - 0.5
        else:
            threshold = self.config['zle_threshold'] - 0.5
        # Figure out if we were above threshold
        # - Current max is the highest value in the SPE pulse model given a
        #   remainder [0-9] ns (offset from 10 ns sample time).
        #   The SPE pulse model sums up to 0.1 (pe/ns), and the peak is ~0.03 (pe/ns)
        # - Multiply this by the sampled gain of each photon, we get (electron/ns)
        # - The current_2_adc take into account n_electron -> current -> voltage
        #   -> amplification -> digitization, converting the electron/ns to adc value.
        remainder = (photon_timings % dt).astype(int)
        max_amplitude_adc = photon_gains * self.current_max[remainder] * self.current_2_adc
        above_threshold = max_amplitude_adc > threshold
        trigger_photon = np.sum(above_threshold)
        trigger_dpe = np.sum(above_threshold[:n_double_pe])
        raw_area = np.sum(photon_gains) / self.config['gains'][channel]
        trigger_raw_area = np.sum(photon_gains[above_threshold]) / self.config['gains'][channel]

        self._n_photon += len(photon_timings)
        self._n_pe += len(photon_timings) + n_double_pe
        self._n_photon_trigger += trigger_photon
        self._n_pe_trigger += trigger_photon + trigger_dpe
        self._raw_area += raw_area
        self._raw_area_trigger += trigger_raw_area

        if channel in self.config['channels_bottom']:
            self._n_photon_bottom += len(photon_timings)
            self._n_pe_bottom += len(photon_timings) + n_double_pe
            self._n_photon_trigger_bottom += trigger_photon
            self._n_pe_trigger_bottom += trigger_photon + trigger_dpe
            self._raw_area_bottom += raw_area
            self._raw_area_trigger_bottom += trigger_raw_area

    def clear_pulse_cache(self):
        self._pulses = []

    @staticmethod
    @njit
    def add_current(photon_timings,
                    photon_gains,
                    pulse_left,
                    dt,
                    pmt_current_templates,
                    pulse_current):
        #         """
        #         Simulate single channel waveform given the photon timings
        #         photon_timing         - dim-1 integer array of photon timings in unit of ns
        #         photon_gain           - dim-1 float array of ph. 2 el. gain individual photons
        #         pulse_left            - left of the pulse in unit of 10 ns
        #         dt                    - mostly it is 10 ns
        #         pmt_current_templates - list of spe templates of different reminders
        #         pulse_current         - waveform
        #         """
        if not len(photon_timings):
            return
        
        template_length = len(pmt_current_templates[0])
        i_photons = np.argsort(photon_timings)
        # Convert photon_timings to int outside this function
        # photon_timings = photon_timings // 1

        gain_total = 0
        tmp_photon_timing = photon_timings[i_photons[0]]
        for i in i_photons:
            if photon_timings[i] > tmp_photon_timing:
                start = int(tmp_photon_timing // dt) - pulse_left
                reminder = int(tmp_photon_timing % dt)
                pulse_current[start:start + template_length] += \
                    pmt_current_templates[reminder] * gain_total

                gain_total = photon_gains[i]
                tmp_photon_timing = photon_timings[i]
            else:
                gain_total += photon_gains[i]

        start = int(tmp_photon_timing // dt) - pulse_left
        reminder = int(tmp_photon_timing % dt)
        pulse_current[start:start + template_length] += \
            pmt_current_templates[reminder] * gain_total

    @staticmethod
    def singlet_triplet_delays(size, singlet_ratio, config, phase):
        """
        Given the amount of the excimer, return time between excimer decay
        and their time of generation.
        size           - amount of excimer
        self.phase     - 'liquid' or 'gas'
        singlet_ratio  - fraction of excimers that become singlets
                         (NOT the ratio of singlets/triplets!)
        """
        if phase == 'liquid':
            t1, t3 = (config['singlet_lifetime_liquid'],
                      config['triplet_lifetime_liquid'])
        elif phase == 'gas':
            t1, t3 = (config['singlet_lifetime_gas'],
                      config['triplet_lifetime_gas'])
        else:
            t1, t3 = 0, 0

        delay = np.random.choice([t1, t3], size, replace=True,
                                 p=[singlet_ratio, 1 - singlet_ratio])
        return (np.random.exponential(1, size) * delay).astype(np.int64)
