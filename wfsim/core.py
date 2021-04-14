import logging

from numba import njit
import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm

from .load_resource import load_config, DummyMap
from strax import exporter, deterministic_hash
from . import units
from .utils import find_intervals_below_threshold

export, __all__ = exporter()
__all__.append('PULSE_TYPE_NAMES')

logging.basicConfig(handlers=[
    # logging.handlers.WatchedFileHandler('wfsim.log'),
    logging.StreamHandler()])
log = logging.getLogger('wfsim.core')
log.setLevel('WARNING')

PULSE_TYPE_NAMES = ('RESERVED', 's1', 's2', 'unknown', 'pi_el', 'pmt_ap', 'pe_el')
_cached_pmt_current_templates = {}
_cached_uniform_to_pe_arr = {}


@export
class NestId:
    """Nest ids for referring to different scintillation models, only ER is actually validated"""
    NR = [0]
    ALPHA = [6]
    ER = [7, 8, 11, 12]
    LED = [20]
    _ALL = NR + ALPHA + ER + LED


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
        self._n_double_pe = self._n_double_pe_bot = 0  # For truth aft output

        counts_start = 0  # Secondary loop index for assigning channel
        for channel, counts in zip(*np.unique(self._photon_channels, return_counts=True)):

            # Use 'counts' amount of photon for this channel
            _channel_photon_timings = self._photon_timings[counts_start:counts_start+counts]
            counts_start += counts
            if channel in self.config['turned_off_pmts']:
                continue

            # If gain of each photon is not specifically assigned
            # Sample from spe scaling factor distribution and to individual gain
            # In contrast to pmt afterpulse that should have gain determined before this step
            if '_photon_gains' not in self.__dict__:

                _channel_photon_gains = self.config['gains'][channel] \
                    * self.uniform_to_pe_arr(np.random.random(len(_channel_photon_timings)))

                # Add some double photoelectron emission by adding another sampled gain
                n_double_pe = np.random.binomial(len(_channel_photon_timings),
                                                 p=self.config['p_double_pe_emision'])
                self._n_double_pe += n_double_pe
                if channel in self.config['channels_bottom']:
                    self._n_double_pe_bot += n_double_pe

                if self.config['detector'] == 'XENON1T':
                    _channel_photon_gains[:n_double_pe] += self.config['gains'][channel] \
                        * self.uniform_to_pe_arr(np.random.random(n_double_pe), channel)
                else:
                    _channel_photon_gains[:n_double_pe] += self.config['gains'][channel] \
                        * self.uniform_to_pe_arr(np.random.random(n_double_pe))
            else:
                _channel_photon_gains = np.array(self._photon_gains[self._photon_channels == channel])

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
        self._template_length = np.int(len(samples) - 1)

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
        else:
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


@export
class S1(Pulse):
    """
    Given temperal inputs as well as number of photons
    Random generate photon timing and channel distribution.
    """

    def __init__(self, config):
        super().__init__(config)
        self.phase = 'liquid'  # To distinguish singlet/triplet time delay.

    def __call__(self, instruction):
        """Main s1 simulation function. Called by RawData for s1 simulation. 
        Generates first number of photons in the s1, then timings and channels.
        These arrays are fed to Pulse to generate the data.

        param instructions: Array with dtype wfsim.instruction_dtype """
        if len(instruction.shape) < 1:
            # shape of recarr is a bit strange
            instruction = np.array([instruction])

        _, _, t, x, y, z, n_photons, recoil_type, *rest = [
            np.array(v).reshape(-1) for v in zip(*instruction)]

        positions = np.array([x, y, z]).T  # For map interpolation
        n_photons = self.get_n_photons(n_photons=n_photons,
                                       positions=positions,
                                       s1_light_yield_map=self.resource.s1_light_yield_map,
                                       config=self.config)

        self._photon_timings = self.photon_timings(t=t,
                                                   n_photons=n_photons, 
                                                   recoil_type=recoil_type,
                                                   config=self.config,
                                                   phase=self.phase)

        # The new way interpolation is written always require a list
        self._photon_channels = self.photon_channels(positions=positions,
                                                     n_photons=n_photons,
                                                     config=self.config, 
                                                     s1_pattern_map=self.resource.s1_pattern_map)

        super().__call__()

    @staticmethod
    def get_n_photons(n_photons, positions, s1_light_yield_map, config):
        """Calculates number of detected photons based on number of photons in total and the positions
        :param n_photons: 1d array of ints with number of photons:
        :param positions: 2d array with xyz positions of interactions
        :param s1_light_yield_map: interpolator instance of s1 light yield map
        :param config: dict wfsim config 
        
        return array with number photons"""
        if config['detector'] == 'XENONnT':
            ly = np.squeeze(s1_light_yield_map(positions),
                            axis=-1)/(1+config['p_double_pe_emision'])
        elif config['detector'] == 'XENON1T':
            ly = s1_light_yield_map(positions)
            ly *= config['s1_detection_efficiency']
        n_photons = np.random.binomial(n=n_photons, p=ly)
        return n_photons

    @staticmethod
    def photon_channels(positions, n_photons, config, s1_pattern_map):
        """Calculate photon arrival channels
        :params positions: 2d array with xy positions of interactions
        :params n_photons: 1d array of ints with number of photons to simulate
        :params config: dict wfsim config
        :params s1_pattern_map: interpolator instance of the s1 pattern map

        returns nested array with photon channels   
        """
        channels = np.arange(config['n_tpc_pmts'])  # +1 for the channel map
        p_per_channel = s1_pattern_map(positions)
        p_per_channel[:, np.in1d(channels, config['turned_off_pmts'])] = 0

        _photon_channels = np.array([]).astype(np.int64)
        for ppc, n in zip(p_per_channel, n_photons):
            _photon_channels = np.append(_photon_channels,
                                         np.random.choice(
                                             channels,
                                             size=n,
                                             p=ppc / np.sum(ppc),
                                             replace=True))
        return _photon_channels

    @staticmethod
    def photon_timings(t, n_photons, recoil_type, config, phase):
        """Calculate distribution of photon arrival timnigs
        :param t: 1d array of ints
        :param n_photons: 1d array of ints
        :param recoil_type: 1d array of ints
        :param config: dict wfsim config
        :param phase: str "liquid"

        returns photon timing array"""
        _photon_timings = np.repeat(t, n_photons)
        if len(_photon_timings) == 0:
            return _photon_timings.astype(np.int64)
        
        if (config['s1_model_type'] == 'simple' and
                np.isin(recoil_type, NestId._ALL).all()):
            # Simple S1 model enabled: use it for ER and NR.
            _photon_timings += np.random.exponential(config['s1_decay_time'], len(_photon_timings)).astype(np.int64)
            _photon_timings += np.random.normal(0, config['s1_decay_spread'], len(_photon_timings)).astype(np.int64)
            return _photon_timings

        counts_start = 0
        for i, counts in enumerate(n_photons):
            for k in vars(NestId):
                if k.startswith('_'):
                    continue
                if recoil_type[i] in getattr(NestId, k):
                    str_recoil_type = k
            try:
                _photon_timings[counts_start: counts_start + counts] += \
                    getattr(S1, str_recoil_type.lower())(
                    size=counts,
                    config=config,
                    phase=phase).astype(np.int64)
            except AttributeError:
                raise AttributeError(f"Recoil type must be ER, NR, alpha or LED, not {recoil_type}. Check nest ids")
            counts_start += counts
        return _photon_timings

    @staticmethod
    def alpha(size, config, phase):
        """  Calculate S1 photon timings for an alpha decay. Neglible recombination time, not validated
        :param size: 1d array of ints, number of photons
        :param config: dict wfsim config
        :param phase: str "liquid"
        
        return 1d array of photon timings"""
        return Pulse.singlet_triplet_delays(size, config['s1_ER_alpha_singlet_fraction'], config, phase)

    @staticmethod
    def led(size, config, **kwargs):
        """  distribute photons uniformly within the LED pulse length, not validated
        :param size: 1d array of ints, number of photons
        :param config: dict wfsim config

        return 1d array of photon timings"""
        return np.random.uniform(0, config['led_pulse_length'], size)

    @staticmethod
    def er(size, config, phase):
        """Complex ER model, not validated
        :param size: 1d array of ints, number of photons
        :param config: dict wfsim config
        :param phase: str "liquid"
        return 1d array of photon timings
        """

        # How many of these are primary excimers? Others arise through recombination.
        # This config is not set for the nT fax config todo
        config.setdefault('liquid_density', 1.872452802978054e+30)
        density = config['liquid_density'] / (units.g / units.cm ** 3)
        excfrac = 0.4 - 0.11131 * density - 0.0026651 * density ** 2    # primary / secondary excimers
        excfrac = 1 / (1 + excfrac)                                     # primary / all excimers
        # primary / all excimers that produce a photon:
        excfrac /= 1 - (1 - excfrac) * (1 - config['s1_ER_recombination_fraction'])
        config['s1_ER_primary_excimer_fraction'] = excfrac
        log.debug('Inferred s1_ER_primary_excimer_fraction %s' % excfrac)

        # Recombination time from NEST 2014
        # 3.5 seems fishy, they fit an exponential to data, but in the code they use a non-exponential distribution...
        efield = (config['drift_field'] / (units.V / units.cm))
        reco_time = 3.5 / \
            0.18 * (1 / 20 + 0.41) * np.exp(-0.009 * efield)
        config['s1_ER_recombination_time'] = reco_time
        log.debug('Inferred s1_ER_recombination_time %s' % reco_time)

        timings = np.random.choice([0, reco_time], size, replace=True,
                                   p=[excfrac, 1 - excfrac])
        primary = timings == 0
        size_primary = len(timings[primary])

        timings[primary] += Pulse.singlet_triplet_delays(
            size_primary, config['s1_ER_primary_singlet_fraction'], config, phase)

        # Correct for the recombination time
        # For the non-exponential distribution: see Kubota 1979, solve eqn 2 for n/n0.
        # Alternatively, see Nest V098 source code G4S1Light.cc line 948
        timings[~primary] *= 1 / (-1 + 1 / np.random.uniform(0, 1, size - size_primary))
        # Update max recombine time in the nT fax config
        config['maximum_recombination_time'] = 1000
        timings[~primary] = np.clip(timings[~primary], 0, config['maximum_recombination_time'])
        timings[~primary] += Pulse.singlet_triplet_delays(
            size - size_primary, config['s1_ER_secondary_singlet_fraction'], config, phase)

        return timings

    @staticmethod
    def nr(size, config, phase):
        """NR model model, not validated
        :param size: 1d array of ints, number of photons
        :param config: dict wfsim config
        :param phase: str "liquid"
        return 1d array of photon timings
        """
        return Pulse.singlet_triplet_delays(size, config['s1_NR_singlet_fraction'], config, phase)


@export
class S2(Pulse):
    """
    Given temperal inputs as well as number of electrons
    Random generate photon timing and channel distribution.
    """

    def __init__(self, config):
        super().__init__(config)

        self.phase = 'gas'  # To distinguish singlet/triplet time delay.
        self.luminescence_switch_threshold = 100  # When to use simplified model (NOT IN USE)

    def __call__(self, instruction):
        if len(instruction.shape) < 1:
            # shape of recarr is a bit strange
            instruction = np.array([instruction])

        _, _, t, x, y, z, n_electron, recoil_type, *rest = [
            np.array(v).reshape(-1) for v in zip(*instruction)]
        
        # Reverse engineerring FDC
        if self.config['field_distortion_on']:
            z_obs, positions = self.inverse_field_distortion(x, y, z, resource=self.resource)
        else:
            z_obs, positions = z, np.array([x, y]).T

        sc_gain = self.get_s2_light_yield(positions=positions,
                                          config=self.config,
                                          resource=self.resource)

        n_electron = self.get_electron_yield(n_electron=n_electron,
                                             positions=positions,
                                             z_obs=z_obs,
                                             config=self.config,
                                             resource=self.resource)

        # Second generate photon timing and channel
        self._electron_timings, self._photon_timings, self._instruction = self.photon_timings(t, n_electron, z_obs,
                                                                                              positions, sc_gain,
                                                                                              config=self.config,
                                                                                              resource=self.resource,
                                                                                              phase=self.phase)

        self._photon_channels, self._photon_timings = self.photon_channels(n_electron=n_electron,
                                                                           z_obs=z_obs,
                                                                           positions=positions,
                                                                           _photon_timings=self._photon_timings,
                                                                           _instruction=self._instruction,
                                                                           config=self.config,
                                                                           resource=self.resource)
        super().__call__()

    @staticmethod
    def get_s2_drift_time_params(z_obs, positions, config, resource):
        """Calculate s2 drift time mean and spread

        :param positions: 1d array of z (floats)
        :param positions: 2d array of positions (floats)
        :param config: dict with wfsim config
        :param resource: instance of the resource class

        returns two arrays of floats (mean drift time, drift time spread) 
        """

        if config['enable_field_dependencies']['drift_speed_map']:
            drift_velocity_liquid = resource.field_dependencies_map(z_obs, positions, map_name='drift_speed_map')  # mm/µs
            drift_velocity_liquid *= 1e-4  # cm/ns
        else:
            drift_velocity_liquid = config['drift_velocity_liquid']

        if config['enable_field_dependencies']['diffusion_longitudinal_map']:
            diffusion_constant_longitudinal = resource.field_dependencies_map(z_obs, positions, map_name='diffusion_longitudinal_map')  # cm²/s
            diffusion_constant_longitudinal *= 1e-9  # cm²/ns
        else:
            diffusion_constant_longitudinal = config['diffusion_constant_longitudinal']

        drift_time_mean = - z_obs / \
            drift_velocity_liquid + config['drift_time_gate']
        _drift_time_mean = np.clip(drift_time_mean, 0, np.inf)
        drift_time_spread = np.sqrt(2 * diffusion_constant_longitudinal * _drift_time_mean)
        drift_time_spread /= drift_velocity_liquid
        return drift_time_mean, drift_time_spread

    @staticmethod
    def get_s2_light_yield(positions, config, resource):
        """Calculate s2 light yield...
        
        :param positions: 2d array of positions (floats)
        :param config: dict with wfsim config
        :param resource: instance of the resource class
        
        returns array of floats (mean expectation) 
        """
        if config['detector'] == 'XENONnT':
            sc_gain = np.squeeze(resource.s2_light_yield_map(positions), axis=-1) \
                * config['s2_secondary_sc_gain']
        elif config['detector'] == 'XENON1T':
            sc_gain = resource.s2_light_yield_map(positions) \
                * config['s2_secondary_sc_gain']
        return sc_gain

    @staticmethod
    def get_electron_yield(n_electron, positions, z_obs, config, resource):
        """Drift electrons up to the gas interface and absorb them

        :param n_electron: 1d array with ints as number of electrons
        :param positions: 2d array of positions (floats)
        :param z_obs: 1d array of floats with the observed z positions
        :param config: dict with wfsim config

        returns 1d array ints with number of electrons
        """
        # Average drift time of the electrons
        drift_time_mean, drift_time_spread = S2.get_s2_drift_time_params(z_obs, positions, config, resource)

        # Absorb electrons during the drift
        electron_lifetime_correction = np.exp(- 1 * drift_time_mean /
                                              config['electron_lifetime_liquid'])
        cy = config['electron_extraction_yield'] * electron_lifetime_correction

        # Remove electrons in insensitive volumne
        if config['enable_field_dependencies']['survival_probability_map']:
            survival_probability = resource.field_dependencies_map(z_obs, positions, map_name='survival_probability_map')
            cy *= survival_probability

        # why are there cy greater than 1? We should check this
        cy = np.clip(cy, a_min = 0, a_max = 1)
        n_electron = np.random.binomial(n=n_electron, p=cy)
        return n_electron

    @staticmethod
    def inverse_field_distortion(x, y, z, resource):
        """For 1T the pattern map is a data driven one so we need to reverse engineer field distortion
        into the simulated positions
        :param x: 1d array of float
        :param y: 1d array of float
        :param z: 1d array of float
        :param resource: instance of resource class
        returns z: 1d array, postions 2d array 
        """
        positions = np.array([x, y, z]).T
        for i_iter in range(6):  # 6 iterations seems to work
            dr = resource.fdc_3d(positions)
            if i_iter > 0:
                dr = 0.5 * dr + 0.5 * dr_pre  # Average between iter
            dr_pre = dr

            r_obs = np.sqrt(x**2 + y**2) - dr
            x_obs = x * r_obs / (r_obs + dr)
            y_obs = y * r_obs / (r_obs + dr)
            z_obs = - np.sqrt(z**2 + dr**2)
            positions = np.array([x_obs, y_obs, z_obs]).T

        positions = np.array([x_obs, y_obs]).T 
        return z_obs, positions

    @staticmethod
    @njit
    def _luminescence_timings_simple(n, dG, E0, r, dr, rr, alpha, uE, p, n_electron, shape):
        emission_time = np.zeros(shape, np.int64)
        """
        Luminescence time distribution computation, calculates emission timings of photons from the excited electrons
        return 1d nested array with ints
        """

        ci = 0
        for i in range(n):
            ne = n_electron[i]
            dt = dr / (alpha * E0[i] * rr)
            dy = E0[i] * rr / uE - 0.8 * p  # arXiv:physics/0702142
            avgt = np.sum(np.cumsum(dt) * dy) / np.sum(dy)

            j = np.argmax(r <= dG[i])
            t = np.cumsum(dt[j:]) - avgt
            y = np.cumsum(dy[j:])

            probabilities = np.random.rand(ne, shape[1])
            emission_time[ci:ci+ne, :] = np.interp(probabilities, y / y[-1], t).astype(np.int64)
            ci += ne

        return emission_time

    @staticmethod
    def luminescence_timings_simple(xy, n_electron, shape, config, resource):
        """
        Luminescence time distribution computation according to simple s2 model (many many many single electrons)
        :param xy: 1d array with positions
        :param n_electron: 1d array with ints for number f electrons
        :param shape: tuple with nelectron,nphotons
        :param config: dict wfsim config
        :param resource: instance of wfsim resource
        returns _luminescence_timings_simple
        """
        assert len(n_electron) == len(xy), 'Input number of n_electron should have same length as positions'
        assert np.sum(n_electron) == shape[0], 'Total number of electron does not agree with shape[0]'

        number_density_gas = config['pressure'] / \
            (units.boltzmannConstant * config['temperature'])
        alpha = config['gas_drift_velocity_slope'] / number_density_gas
        uE = units.kV / units.cm
        pressure = config['pressure'] / units.bar

        if config.get('enable_gas_gap_warping', True):
            dG = resource.gas_gap_length(xy)
        else:
            dG = np.ones(len(xy)) * config['elr_gas_gap_length']
        rA = config['anode_field_domination_distance']
        rW = config['anode_wire_radius']
        dL = config['gate_to_anode_distance'] - dG

        VG = config['anode_voltage'] / (1 + dL / dG / config['lxe_dielectric_constant'])
        E0 = VG / ((dG - rA) / rA + np.log(rA / rW))  # V / cm

        dr = 0.0001  # cm
        r = np.arange(np.max(dG), rW, -dr)
        rr = np.clip(1 / r, 1 / rA, 1 / rW)

        return S2._luminescence_timings_simple(len(xy), dG, E0, 
                                               r, dr, rr, alpha, uE,
                                               pressure, n_electron, shape)

    @staticmethod
    def luminescence_timings_garfield(xy, n_electron, shape, config, resource):
        """
        Luminescence time distribution computation according to garfield scintillation maps
        :param xy: 1d array with positions
        :param n_electron: 1d array with ints for number f electrons
        :param shape: tuple with nelectron,nphotons
        :param config: dict wfsim config
        :param resource: instance of wfsim resource

        returns 2d array with ints for photon timings of input param 'shape'
        """
        assert 's2_luminescence' in resource.__dict__, 's2_luminescence model not found'
        assert len(n_electron) == len(xy), 'Input number of n_electron should have same length as positions'
        assert np.sum(n_electron) == shape[0], 'Total number of electron does not agree with shape[0]'
        assert len(resource.s2_luminescence['t'].shape) == 2, 'Timing data is expected to have D2'

        tilt = config.get('anode_xaxis_angle', np.pi / 4)
        pitch = config.get('anode_pitch', 0.5)
        rotation_mat = np.array(((np.cos(tilt), -np.sin(tilt)), (np.sin(tilt), np.cos(tilt))))

        jagged = lambda relative_y: (relative_y + pitch / 2) % pitch - pitch / 2
        distance = jagged(np.matmul(xy, rotation_mat)[:, 1])  # shortest distance from any wire

        index_row = [np.argmin(np.abs(d - resource.s2_luminescence['x'])) for d in distance]
        index_row = np.repeat(index_row, n_electron).astype(np.int64)
        index_col = np.random.randint(0, resource.s2_luminescence['t'].shape[1], shape, np.int64)

        return resource.s2_luminescence['t'][index_row[:, None], index_col].astype(np.int64)

    @staticmethod
    @njit
    def electron_timings(t, n_electron, drift_time_mean, drift_time_spread, sc_gain, timings, gains,
            electron_trapping_time):
        """Calculate arrival times of the electrons. Data is written to the timings and gains arrays
        :param t: 1d array of ints
        :param n_electron:1 d array of ints
        :param drift_time_mean: 1d array of floats
        :param drift_time_spread: 1d array of floats
        :param sc_gain: secondairy scintallation gain       
        :param timings: empty array with length sum(n_electron)
        :param gains: empty array with length sum(n_electron)
        :param electron_trapping_time: configuration values
        """
        assert len(timings) == np.sum(n_electron)
        assert len(gains) == np.sum(n_electron)
        assert len(sc_gain) == len(t)

        i_electron = 0
        for i in np.arange(len(t)):
            # Calculate electron arrival times in the ELR region
            for _ in np.arange(n_electron[i]):
                _timing = np.random.exponential(electron_trapping_time)
                _timing += np.random.normal(drift_time_mean[i], drift_time_spread[i])
                timings[i_electron] = t[i] + int(_timing)

                # add manual fluctuation to sc gain
                gains[i_electron] = sc_gain[i]
                i_electron += 1

    @staticmethod
    def photon_timings(t, n_electron, z, xy, sc_gain, config, resource, phase):
        """Generates photon timings for S2s. Returns a list of photon timings and instructions repeated for original electron
        
        :param t: 1d float array arrival time of the electrons
        :param n_electron: 1d float array number of electrons to simulate
        :param z: float array. Z positions of s2
        :param xy: 1d float array, xy positions of s2
        :param sc_gain: float, secondary s2 gain
        :param config: dict of the wfsim config
        :param resource: instance of the resource class
        :param phase: string, "gas" """
        # First generate electron timings
        _electron_timings = np.zeros(np.sum(n_electron), np.int64)
        _electron_gains = np.zeros(np.sum(n_electron), np.float64)
        drift_time_mean, drift_time_spread = S2.get_s2_drift_time_params(z, xy, config, resource)
        S2.electron_timings(t, n_electron, drift_time_mean, drift_time_spread, sc_gain, 
            _electron_timings, _electron_gains, 
            config['electron_trapping_time'])

        if len(_electron_timings) < 1:
            return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0)

        # For vectorized calculation, artificially top #photon per electron at +4 sigma
        nele = len(_electron_timings)
        npho = np.ceil(np.max(_electron_gains) +
                       4 * np.sqrt(np.max(_electron_gains))).astype(np.int64)

        if config['s2_luminescence_model'] == 'simple':
            _photon_timings = S2.luminescence_timings_simple(xy, n_electron, (nele, npho), 
                                                             config=config,
                                                             resource=resource)
        elif config['s2_luminescence_model'] == 'garfield':
            _photon_timings = S2.luminescence_timings_garfield(
                xy, n_electron, (nele, npho),
                config=config,
                resource=resource)

        _photon_timings += np.repeat(_electron_timings, npho).reshape((nele, npho))

        # Crop number of photons by random number generated with poisson
        probability = np.tile(np.arange(npho), nele).reshape((nele, npho))
        threshold = np.repeat(np.random.poisson(_electron_gains), npho).reshape((nele, npho))
        _photon_timings = _photon_timings[probability < threshold]

        # Special index for match photon to original electron poistion
        _instruction = np.repeat(
            np.repeat(np.arange(len(t)), n_electron), npho).reshape((nele, npho))
        _instruction = _instruction[probability < threshold]

        _photon_timings += Pulse.singlet_triplet_delays(
            len(_photon_timings), config['singlet_fraction_gas'], config, phase)

        _photon_timings += np.random.normal(0, config['s2_time_spread'], len(_photon_timings)).astype(np.int64)
        # The timings generated is NOT randomly ordered, must do shuffle
        # Shuffle within each given n_electron[i]
        # We can do this by first finding out cumulative sum of the photons
        cumulate_npho = np.pad(np.cumsum(threshold[:, 0]), [1, 0])[np.cumsum(n_electron)]
        for i in range(len(cumulate_npho)):
            if i == 0:
                s = slice(0, cumulate_npho[i])
            else:
                s = slice(cumulate_npho[i-1], cumulate_npho[i])
            np.random.shuffle(_photon_timings[s])

        return _electron_timings, _photon_timings, _instruction

    @staticmethod
    def s2_pattern_map_diffuse(n_electron, z, xy, config, resource):
        """Returns an array of pattern of shape [n interaction, n PMTs]
        pattern of each interaction is an average of n_electron patterns evaluated at
        diffused position near xy. The diffused positions sample from 2d symmetric gaussian
        with spread scale with sqrt of drift time.

        :param n_electron: a 1d int array
        :param z: a 1d float array
        :param xy: a 2d float array of shape [n interaction, 2]
        :param config: dict of the wfsim config
        :param resource: instance of the resource class
        """
        drift_time_gate = config['drift_time_gate']
        drift_velocity_liquid = config['drift_velocity_liquid']
        diffusion_constant_transverse = getattr(config, 'diffusion_constant_transverse', 0)

        assert all(z < 0), 'All S2 in liquid should have z < 0'

        drift_time_mean = - z / drift_velocity_liquid + drift_time_gate  # Add gate time for consistancy?
        hdiff_stdev = np.sqrt(2 * diffusion_constant_transverse * drift_time_mean)

        hdiff = np.random.normal(0, 1, (np.sum(n_electron), 2)) * \
            np.repeat(hdiff_stdev, n_electron, axis=0).reshape((-1, 1))
        # Should we also output this xy position in truth?
        xy_multi = np.repeat(xy, n_electron, axis=0) + hdiff  # One entry xy per electron
        # Remove points outside tpc, and the pattern will be the average inside tpc
        # Should be done natually with the s2 pattern map, however, there's some bug there so we apply this hard cut
        mask = np.sum(xy_multi ** 2, axis=1) <= config['tpc_radius'] ** 2

        if isinstance(resource.s2_pattern_map, DummyMap):
            output_dim = resource.s2_pattern_map.shape[-1]
        else:
            output_dim = resource.s2_pattern_map.data['map'].shape[-1]
        pattern = np.zeros((len(n_electron), output_dim))
        n0 = 0
        # Average over electrons for each s2
        for ix, ne in enumerate(n_electron):
            s = slice(n0, n0+ne)
            pattern[ix, :] = np.average(resource.s2_pattern_map(xy_multi[s][mask[s]]), axis=0)
            n0 += ne

        return pattern

    @staticmethod
    def photon_channels(n_electron, z_obs, positions, _photon_timings, _instruction, config, resource):
        """Set the _photon_channels property list of length same as _photon_timings

        :param n_electron: a 1d int array
        :param z_obs: a 1d float array
        :param positions: a 2d float array of shape [n interaction, 2] for the xy coordinate
        :param _photon_timings: 1d int array of photon timings,
        :param _instruction: array of instructions with dtype wfsim.instructions_dtype
        :param config: dict wfsim config
        :param resource: instance of resource class
        """
        if len(_photon_timings) == 0:
            _photon_channels = []
            return _photon_timings, _photon_channels

        aft = config['s2_mean_area_fraction_top']
        aft_random = config.get('randomize_fraction_of_s2_top_array_photons', 0)
        channels = np.arange(config['n_tpc_pmts']).astype(np.int64)
        top_index = np.arange(config['n_top_pmts'])
        bottom_index = np.array(config['channels_bottom'])

        if config.get('diffusion_constant_transverse', 0) > 0:
            pattern = S2.s2_pattern_map_diffuse(n_electron, z_obs, positions, config, resource)  # [position, pmt]
        else:
            pattern = resource.s2_pattern_map(positions)  # [position, pmt]
        if pattern.shape[1] - 1 not in bottom_index:
            pattern = np.pad(pattern, [[0, 0], [0, len(bottom_index)]], 
                             'constant', constant_values=1)
        sum_pat = np.sum(pattern, axis=1).reshape(-1, 1)
        pattern = np.divide(pattern, sum_pat, out=np.zeros_like(pattern), where=sum_pat != 0)

        assert pattern.shape[0] == len(positions)
        assert pattern.shape[1] == len(channels)

        _buffer_photon_channels = []
        # Randomly assign to channel given probability of each channel
        for unique_i, count in zip(*np.unique(_instruction, return_counts=True)):
            pat = pattern[unique_i]  # [pmt]

            if aft > 0:  # Redistribute pattern with user specified aft
                _aft = aft * (1 + np.random.normal(0, aft_random))
                _aft = np.clip(_aft, 0, 1)
                pat[top_index] = pat[top_index] / pat[top_index].sum() * _aft
                pat[bottom_index] = pat[bottom_index] / pat[bottom_index].sum() * (1 - _aft)

            if np.isnan(pat).sum() > 0:  # Pattern map return zeros
                _photon_channels = np.array([-1] * count)
            else:
                _photon_channels = np.random.choice(
                    channels,
                    size=count,
                    p=pat,
                    replace=True)

            _buffer_photon_channels.append(_photon_channels)
        
        _photon_channels = np.concatenate(_buffer_photon_channels)
        # Remove photon with channel -1
        mask = _photon_channels != -1
        _photon_channels = _photon_channels[mask]
        _photon_timings = _photon_timings[mask]
        
        sorted_index = np.argsort(_photon_channels)

        return _photon_channels[sorted_index], _photon_timings[sorted_index]


@export
class PhotoIonization_Electron(S2):
    """
    Produce electron after pulse simulation, using already built cdfs
    The cdfs follow distribution parameters extracted from data.
    """

    def __init__(self, config):
        super().__init__(config)
        self._photon_timings = []

    def generate_instruction(self, signal_pulse, signal_pulse_instruction):
        if len(signal_pulse._photon_timings) == 0:
            return []
        return self.electron_afterpulse(signal_pulse, signal_pulse_instruction)

    def electron_afterpulse(self, signal_pulse, signal_pulse_instruction):
        """
        For electron afterpulses we assume a uniform x, y
        """
        delaytime_pmf_hist = self.resource.uniform_to_ele_ap

        # To save calculation we first find out how many photon will give rise ap
        n_electron = np.random.poisson(delaytime_pmf_hist.n
                                       * len(signal_pulse._photon_timings)
                                       * self.config['photoionization_modifier'])

        ap_delay = delaytime_pmf_hist.get_random(n_electron).clip(
            self.config['drift_time_gate'] + 1, None)

        # Randomly select original photon as time zeros
        t_zeros = signal_pulse._photon_timings[np.random.randint(
            low=0, high=len(signal_pulse._photon_timings),
            size=n_electron)]

        instruction = np.repeat(signal_pulse_instruction[0], n_electron)

        instruction['type'] = 4  # pi_el
        instruction['time'] = t_zeros + self.config['drift_time_gate']
        instruction['x'], instruction['y'] = self._rand_position(n_electron)
        instruction['z'] = - ap_delay * self.config['drift_velocity_liquid']
        instruction['amp'] = 1

        return instruction

    def _rand_position(self, n):
        Rupper = self.config['tpc_radius']

        r = np.sqrt(np.random.uniform(0, Rupper*Rupper, n))
        angle = np.random.uniform(-np.pi, np.pi, n)

        return r * np.cos(angle), r * np.sin(angle)


@export
class PhotoElectric_Electron(S2):
    """
    Produce electron after S2 pulse simulation, using a gaussian distribution
    """

    def __init__(self, config):
        super().__init__(config)
        self._photon_timings = []

    def generate_instruction(self, signal_pulse, signal_pulse_instruction):
        if len(signal_pulse._photon_timings) == 0:
            return []
        return self.electron_afterpulse(signal_pulse, signal_pulse_instruction)

    def electron_afterpulse(self, signal_pulse, signal_pulse_instruction):

        n_electron = np.random.poisson(self.config['photoelectric_p']
                                       * len(signal_pulse._photon_timings)
                                       * self.config['photoelectric_modifier'])

        ap_delay = np.clip(
            np.random.normal(self.config['photoelectric_t_center'] + self.config['drift_time_gate'], 
                             self.config['photoelectric_t_spread'],
                             n_electron), 0, None)

        # Randomly select original photon as time zeros
        t_zeros = signal_pulse._photon_timings[np.random.randint(
            low=0,
            high=len(signal_pulse._photon_timings),
            size=n_electron)]

        instruction = np.repeat(signal_pulse_instruction[0], n_electron)

        instruction['type'] = 6  # pe_el
        instruction['time'] = t_zeros + self.config['drift_time_gate']
        instruction['x'], instruction['y'] = self._rand_position(n_electron)
        instruction['z'] = - ap_delay * self.config['drift_velocity_liquid']
        instruction['amp'] = 1

        return instruction

    def _rand_position(self, n):
        Rupper = self.config['tpc_radius']

        r = np.sqrt(np.random.uniform(0, Rupper*Rupper, n))
        angle = np.random.uniform(-np.pi, np.pi, n)

        return r * np.cos(angle), r * np.sin(angle)


@export
class PMT_Afterpulse(Pulse):
    """
    Produce pmt after pulse simulation, using already built cdfs
    The cdfs follow distribution parameters extracted from data.
    """

    def __init__(self, config):
        if not config['enable_pmt_afterpulses']:
            return

        super().__init__(config)

        # Convert lists back to ndarray. As ndarray not supported by json
        for k in self.resource.uniform_to_pmt_ap.keys():
            for q in self.resource.uniform_to_pmt_ap[k].keys():
                if isinstance(self.resource.uniform_to_pmt_ap[k][q], list):
                    self.resource.uniform_to_pmt_ap[k][q] = np.array(self.resource.uniform_to_pmt_ap[k][q])

    def __call__(self, signal_pulse):
        if len(signal_pulse._photon_timings) == 0:
            self.clear_pulse_cache()
            return

        self._photon_timings, self._photon_channels, self._photon_gains = \
            self.photon_afterpulse(signal_pulse, self.resource, self.config)

        super().__call__()

    @staticmethod
    def photon_afterpulse(signal_pulse, resource, config):
        """
        For pmt afterpulses, gain and dpe generation is a bit different from standard photons
        """
        element_list = resource.uniform_to_pmt_ap.keys()
        _photon_timings = []
        _photon_channels = []
        _photon_amplitude = []

        for element in element_list:
            delaytime_cdf = resource.uniform_to_pmt_ap[element]['delaytime_cdf']
            amplitude_cdf = resource.uniform_to_pmt_ap[element]['amplitude_cdf']

            # Assign each photon FRIST random uniform number rU0 from (0, 1] for timing
            rU0 = 1 - np.random.rand(len(signal_pulse._photon_timings))

            # Select those photons with U <= max of cdf of specific channel
            cdf_max = delaytime_cdf[signal_pulse._photon_channels, -1]
            sel_photon_id = np.where(rU0 <= cdf_max * config['pmt_ap_modifier'])[0]
            if len(sel_photon_id) == 0:
                continue
            sel_photon_channel = signal_pulse._photon_channels[sel_photon_id]

            # Assign selected photon SECOND random uniform number rU1 from (0, 1] for amplitude
            rU1 = 1 - np.random.rand(len(sel_photon_channel))

            # The map is made so that the indices are delay time in unit of ns
            if 'Uniform' in element:
                ap_delay = np.random.uniform(delaytime_cdf[sel_photon_channel, 0], 
                                             delaytime_cdf[sel_photon_channel, 1])
                ap_amplitude = np.ones_like(ap_delay)
            else:
                ap_delay = (np.argmin(
                    np.abs(
                        delaytime_cdf[sel_photon_channel]
                        - rU0[sel_photon_id][:, None]), axis=-1)
                            - config['pmt_ap_t_modifier'])
                if len(amplitude_cdf.shape) == 2:
                    ap_amplitude = np.argmin(
                        np.abs(
                            amplitude_cdf[sel_photon_channel]
                            - rU1[:, None]), axis=-1) / 100.
                else:
                    ap_amplitude = np.argmin(
                        np.abs(
                            amplitude_cdf[None, :]
                            - rU1[:, None]), axis=-1) / 100.

            _photon_timings.append(signal_pulse._photon_timings[sel_photon_id] + ap_delay)
            _photon_channels.append(signal_pulse._photon_channels[sel_photon_id])
            _photon_amplitude.append(np.atleast_1d(ap_amplitude))

        if len(_photon_timings) > 0:
            _photon_timings = np.hstack(_photon_timings)
            _photon_channels = np.hstack(_photon_channels).astype(np.int64)
            _photon_amplitude = np.hstack(_photon_amplitude)
            _photon_gains = np.array(config['gains'])[_photon_channels] * _photon_amplitude

            return _photon_timings, _photon_channels, _photon_gains

        else:
            return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0)


@export
class RawData(object):

    def __init__(self, config):
        self.config = config
        self.pulses = dict(
            s1=S1(config),
            s2=S2(config),
            pi_el=PhotoIonization_Electron(config),
            pe_el=PhotoElectric_Electron(config),
            pmt_ap=PMT_Afterpulse(config),
        )
        self.resource = load_config(self.config)

    def __call__(self, instructions, truth_buffer=None, progress_bar=True, **kwargs):
        if truth_buffer is None:
            truth_buffer = []

        # Pre-load some constents from config
        v = self.config['drift_velocity_liquid']
        rext = self.config['right_raw_extension']

        # Data cache
        self._pulses_cache = []
        self._raw_data_cache = []

        # Iteration conditions
        self.source_finished = False
        self.last_pulse_end_time = - np.inf
        self.instruction_event_number = np.min(instructions['event_number'])
        # Primary instructions must be sorted by signal time
        # int(type) by design S1-esque being odd, S2-esque being even
        # thus type%2-1 is 0:S1-esque;  -1:S2-esque
        # Make a list of clusters of instructions, with gap smaller then rext
        inst_time = instructions['time'] + (instructions['z'] / v * (instructions['type'] % 2 - 1)).astype(np.int64)
        inst_queue = np.argsort(inst_time)
        inst_queue = np.split(inst_queue, np.where(np.diff(inst_time[inst_queue]) > rext)[0]+1)

        # Instruction buffer
        instb = np.zeros(20000, dtype=instructions.dtype)  # size ~ 1% of size of primary
        instb_filled = np.zeros_like(instb, dtype=bool)  # Mask of where buffer is filled

        # ik those are illegible, messy logic. lmk if you have a better way
        if progress_bar:
            pbar = tqdm(total=len(inst_queue), desc='Simulating Raw Records')
        while not self.source_finished:
            # A) Add a new instruction into buffer
            try:
                ixs = inst_queue.pop(0)  # The index from original instruction list
                self.source_finished = len(inst_queue) == 0
                assert len(np.where(~instb_filled)[0]) > len(ixs), "Run out of instruction buffer"
                ib = np.where(~instb_filled)[0][:len(ixs)]  # The index of first empty slot in buffer
                instb[ib] = instructions[ixs]
                instb_filled[ib] = True
                if progress_bar:
                    pbar.update(1)
            except IndexError:
                pass

            # B) Cluster instructions again with gap size <= rext
            instb_indx = np.where(instb_filled)[0]
            instb_type = instb[instb_indx]['type']
            instb_time = instb[instb_indx]['time'] + \
                         (instb[instb_indx]['z']  / v * (instb_type % 2 - 1)).astype(np.int64)
            instb_queue = np.argsort(instb_time,  kind='stable')
            instb_queue = np.split(instb_queue, 
                                   np.where(np.diff(instb_time[instb_queue]) > rext)[0]+1)
            
            # C) Push pulse cache out first if nothing comes right after them
            if np.min(instb_time) - self.last_pulse_end_time > rext and not np.isinf(self.last_pulse_end_time):
                self.digitize_pulse_cache()
                yield from self.ZLE()

            # D) Run all clusters before the current source
            stop_at_this_group = False
            for ibqs in instb_queue:
                for ptype in [1, 2, 4, 6]:  # S1 S2 PI Gate
                    mask = instb_type[ibqs] == ptype
                    if mask.sum() == 0:
                        continue  # No such instruction type

                    if ptype == 1:
                        stop_at_this_group = True
                        # Group S1 within 100 ns apart, truth info would be summarized within the group
                        instb_run = np.split(instb_indx[ibqs[mask]],
                                             np.where(np.diff(instb_time[ibqs[mask]]) > 100)[0] + 1)
                    elif ptype == 2:
                        stop_at_this_group = True
                        # Group S2 within 2 mm apart, truth info would be summarized within the group
                        instb_run = np.split(instb_indx[ibqs[mask]], 
                                             np.where(np.diff(instb_time[ibqs[mask]]) > int(0.2 / v))[0] + 1)
                    else:
                        instb_run = [instb_indx[ibqs[mask]]]

                    # Run pulse simulation for real
                    n_set = len(instb_run)
                    for i_set, instb_run_i in enumerate(instb_run):
                        if 'g4id' in instructions.dtype.names:
                            g4id = instb[instb_run_i]['g4id'][0]
                            log.debug(f'Making S{ptype} pulse set ({i_set+1}/{n_set}) for g4 event {g4id}')
                        for instb_secondary in self.sim_data(instb[instb_run_i]):
                            n_too_much = 0
                            if len(np.where(~instb_filled)[0]) - 10 < len(instb_secondary):
                                n_too_much = len(instb_secondary) - len(np.where(~instb_filled)[0]) + 10
                                log.warning(f'Running out of instruction buffer removing {n_too_much} secondaries')
                            ib = np.where(~instb_filled)[0][:len(instb_secondary) - n_too_much]
                            instb[ib] = instb_secondary[:len(instb_secondary) - n_too_much]
                            instb_filled[ib] = True

                        if len(truth_buffer):  # Extract truth info
                            self.get_truth(instb[instb_run_i], truth_buffer)

                        instb_filled[instb_run_i] = False  # Free buffer AFTER copyting into truth buffer

                if stop_at_this_group: 
                    break
                self.digitize_pulse_cache()  # from pulse cache to raw data
                yield from self.ZLE()
                
            self.source_finished = len(inst_queue) == 0 and np.sum(instb_filled) == 0
        if progress_bar:
            pbar.close()

    @staticmethod
    def symtype(ptype):
        return PULSE_TYPE_NAMES[ptype]

    def sim_primary(self, primary_pulse, instruction, **kwargs):
        self.pulses[primary_pulse](instruction)

    def sim_data(self, instruction, **kwargs):
        """Simulate a pulse according to instruction, and yield any additional instructions
        for secondary electron afterpulses.
        """
        # Simulate the primary pulse
        primary_pulse = self.symtype(instruction['type'][0])

        self.sim_primary(primary_pulse, instruction, **kwargs)

        # Add PMT afterpulses, if requested
        do_pmt_ap = self.config.get('enable_pmt_afterpulses', True)
        if do_pmt_ap:
            self.pulses['pmt_ap'](self.pulses[primary_pulse])

        # Append pulses we just simulated to our cache
        for pt in [primary_pulse, 'pmt_ap']:
            if pt == 'pmt_ap' and not do_pmt_ap:
                continue

            _pulses = getattr(self.pulses[pt], '_pulses')
            if len(_pulses) > 0:
                self._pulses_cache += _pulses
                self.last_pulse_end_time = max(
                    self.last_pulse_end_time,
                    np.max([p['right'] for p in _pulses]) * self.config['sample_duration'])

        # Make new instructions for electron afterpulses, if requested
        if primary_pulse in ['s1', 's2']:
            if self.config.get('enable_electron_afterpulses', True):
                yield self.pulses['pi_el'].generate_instruction(
                    self.pulses[primary_pulse], instruction)
                if primary_pulse in ['s2']:  # Only add gate ap to s2
                    yield self.pulses['pe_el'].generate_instruction(
                        self.pulses[primary_pulse], instruction)
            self.instruction_event_number = instruction['event_number'][0]
        
    def digitize_pulse_cache(self):
        """
        Superimpose pulses (wfsim definition) into WFs w/ dynamic range truncation
        """
        if len(self._pulses_cache) == 0:
            self._raw_data = []
        else:
            self.current_2_adc = self.config['pmt_circuit_load_resistor'] \
                * self.config['external_amplification'] \
                / (self.config['digitizer_voltage_range'] / 2 ** (self.config['digitizer_bits']))

            self.left = np.min([p['left'] for p in self._pulses_cache]) - self.config['trigger_window']
            self.right = np.max([p['right'] for p in self._pulses_cache]) + self.config['trigger_window']
            pulse_length = self.right - self.left
            log.debug(f'Digitizing pulse from {self.left} - {self.right} of {pulse_length} samples')
            assert self.right - self.left < 1000000, "Pulse cache too long"

            if self.left % 2 != 0:
                self.left -= 1  # Seems like a digizier effect

            self._raw_data = np.zeros((801, self.right - self.left + 1), dtype='<i8')

            # Use this mask to by pass non-activated channels
            # Set to true when working with real noise
            self._channel_mask = np.zeros(801, dtype=[('mask', '?'), ('left', 'i8'), ('right', 'i8')])
            self._channel_mask['left'] = int(2**63-1)

            for _pulse in self._pulses_cache:
                ch = _pulse['channel']
                self._channel_mask['mask'][ch] = True
                self._channel_mask['left'][ch] = min(_pulse['left'], self._channel_mask['left'][ch])
                self._channel_mask['right'][ch] = max(_pulse['right'], self._channel_mask['right'][ch])
                adc_wave = - np.trunc(_pulse['current'] * self.current_2_adc).astype(np.int64)
                _slice = slice(_pulse['left'] - self.left, _pulse['right'] - self.left + 1)

                self._raw_data[ch, _slice] += adc_wave

                if self.config['detector'] == 'XENONnT':
                    adc_wave_he = adc_wave * int(self.config['high_energy_deamplification_factor'])
                    if ch < self.config['n_top_pmts']:
                        ch_he = np.arange(self.config['channel_map']['he'][0],
                                          self.config['channel_map']['he'][1] + 1)[ch]
                        self._raw_data[ch_he, _slice] += adc_wave_he
                        self._channel_mask[ch_he] = True
                        self._channel_mask['left'][ch_he] = self._channel_mask['left'][ch]
                        self._channel_mask['right'][ch_he] = self._channel_mask['right'][ch]
                    elif ch <= self.config['channels_bottom'][-1]:
                        self.sum_signal(adc_wave_he,
                                        _pulse['left'] - self.left,
                                        _pulse['right'] - self.left + 1,
                                        self._raw_data[self.config['channel_map']['sum_signal']])

            self._pulses_cache = []

            self._channel_mask['left'] -= self.left + self.config['trigger_window']
            self._channel_mask['right'] -= self.left - self.config['trigger_window']

            # Adding noise, baseline and digitizer saturation
            
            if self.config.get('enable_noise', True):
                self.add_noise(data=self._raw_data,
                               channel_mask=self._channel_mask,
                               noise_data=self.resource.noise_data,
                               noise_data_length=len(self.resource.noise_data))
            self.add_baseline(self._raw_data, self._channel_mask, 
                              self.config['digitizer_reference_baseline'],)
            self.digitizer_saturation(self._raw_data, self._channel_mask)

    def ZLE(self):
        """
        Modified software zero lengh encoding, coverting WFs into pulses (XENON definition)
        """
        # Ask for memory allocation just once
        if 'zle_intervals_buffer' not in self.__dict__:
            self.zle_intervals_buffer = -1 * np.ones((50000, 2), dtype=np.int64)

        for ix, data in enumerate(self._raw_data):
            if not self._channel_mask['mask'][ix]:
                continue
            channel_left, channel_right = self._channel_mask['left'][ix], self._channel_mask['right'][ix]
            data = data[channel_left:channel_right+1]

            # For simulated data taking reference baseline as baseline
            # Operating directly on digitized downward waveform        
            if str(ix) in self.config.get('special_thresholds', {}):
                threshold = self.config['digitizer_reference_baseline'] \
                    - self.config['special_thresholds'][str(ix)] - 1
            else:
                threshold = self.config['digitizer_reference_baseline'] - self.config['zle_threshold'] - 1

            n_itvs_found = find_intervals_below_threshold(
                data,
                threshold=threshold,
                holdoff=self.config['trigger_window'] + self.config['trigger_window'] + 1,
                result_buffer=self.zle_intervals_buffer,)

            itvs_to_encode = self.zle_intervals_buffer[:n_itvs_found]
            itvs_to_encode[:, 0] -= self.config['trigger_window']
            itvs_to_encode[:, 1] += self.config['trigger_window']
            itvs_to_encode = np.clip(itvs_to_encode, 0, len(data) - 1)
            # Land trigger window on even numbers
            itvs_to_encode[:, 0] = np.ceil(itvs_to_encode[:, 0] / 2.0) * 2
            itvs_to_encode[:, 1] = np.floor(itvs_to_encode[:, 1] / 2.0) * 2

            for itv in itvs_to_encode:
                yield ix, self.left + channel_left + itv[0], self.left + channel_left + itv[1], data[itv[0]:itv[1]+1]

    def get_truth(self, instruction, truth_buffer):
        """Write truth in the first empty row of truth_buffer

        :param instruction: Array of instructions that were simulated as a
        single cluster, and should thus get one line in the truth info.
        :param truth_buffer: Truth buffer to write in.
        """
        ix = np.argmin(truth_buffer['fill'])
        tb = truth_buffer[ix]
        peak_type = self.symtype(instruction['type'][0])
        pulse = self.pulses[peak_type]

        for quantum in 'photon', 'electron':
            times = getattr(pulse, f'_{quantum}_timings', [])
            if len(times):
                tb[f'n_{quantum}'] = len(times)
                tb[f't_mean_{quantum}'] = np.mean(times)
                tb[f't_first_{quantum}'] = np.min(times)
                tb[f't_last_{quantum}'] = np.max(times)
                tb[f't_sigma_{quantum}'] = np.std(times)
            else:
                # Peak does not have photons / electrons
                # zero-photon afterpulses can be removed from truth info
                if peak_type not in ['s1', 's2'] and quantum == 'photon':
                    return
                tb[f'n_{quantum}'] = 0
                tb[f't_mean_{quantum}'] = np.nan
                tb[f't_first_{quantum}'] = np.nan
                tb[f't_last_{quantum}'] = np.nan
                tb[f't_sigma_{quantum}'] = np.nan
        
        # Endtime is the end of the last pulse
        tb['endtime'] = np.mean(instruction['time']) if np.isnan(tb['t_last_photon']) else tb['t_last_photon'] + \
            (self.config['samples_before_pulse_center'] + self.config['samples_after_pulse_center'] + 1) \
            * self.config['sample_duration']
        channels = getattr(pulse, '_photon_channels', [])
        if self.config.get('exclude_dpe_in_truth', False):
            n_dpe = n_dpe_bot = 0
        else:
            n_dpe = getattr(pulse, '_n_double_pe', 0)
            n_dpe_bot = getattr(pulse, '_n_double_pe_bot', 0)
        tb['n_photon'] += n_dpe
        tb['n_photon'] -= np.sum(np.isin(channels, getattr(pulse, 'turned_off_pmts', [])))
        # this turned_off guy, check how this works with a config['turned_off_guys']
        channels_bottom = list(
            set(self.config['channels_bottom']).difference(getattr(pulse, 'turned_off_pmts', [])))
        tb['n_photon_bottom'] = (
            np.sum(np.isin(channels, channels_bottom))
            + n_dpe_bot)

        # Summarize the instruction cluster in one row of the truth file
        for field in instruction.dtype.names:
            value = instruction[field]
            if len(instruction) > 1 and field in 'txyz':
                tb[field] = np.mean(value)
            elif len(instruction) > 1 and field == 'amp':
                tb[field] = np.sum(value)
            else:
                # Cannot summarize intelligently: just take the first value
                tb[field] = value[0]

        # Signal this row is now filled, so it won't be overwritten
        tb['fill'] = True

    @staticmethod
    @njit
    def sum_signal(adc_wave, left, right, sum_template):
        sum_template[left:right] += adc_wave
        return sum_template

    @staticmethod
    @njit
    def add_noise(data, channel_mask, noise_data, noise_data_length):
        """
        Get chunk(s) of noise sample from real noise data
        """
        for ch in range(data.shape[0]):
            if not channel_mask['mask'][ch]:
                continue
            left, right = channel_mask['left'][ch], channel_mask['right'][ch]
            id_t = np.random.randint(low=0, high=noise_data_length-right+left)
            for ix in range(left, right+1):
                if id_t+ix >= noise_data_length or ix >= len(data[ch]):
                    # Don't create value-errors
                    continue
                data[ch, ix] += noise_data[id_t+ix]

    @staticmethod
    @njit
    def add_baseline(data, channel_mask, baseline):
        for ch in range(data.shape[0]):
            if not channel_mask['mask'][ch]:
                continue
            left, right = channel_mask['left'][ch], channel_mask['right'][ch]
            for ix in range(left, right+1):
                data[ch, ix] += baseline

    @staticmethod
    @njit
    def digitizer_saturation(data, channel_mask):
        for ch in range(data.shape[0]):
            if not channel_mask['mask'][ch]:
                continue
            left, right = channel_mask['left'][ch], channel_mask['right'][ch]
            for ix in range(left, right+1):
                if data[ch, ix] < 0:
                    data[ch, ix] = 0


@export
class RawDataOptical(RawData):

    def __init__(self, config, channels=[], timings=[]):
        self.config = config
        self.pulses = dict(
            s1=Pulse(config),
            pi_el=PhotoIonization_Electron(config),
            pe_el=PhotoElectric_Electron(config),
            pmt_ap=PMT_Afterpulse(config))
        self.resource = load_config(config)
        self.channels = channels
        self.timings = timings

    def sim_primary(self, primary_pulse, instruction):
        if primary_pulse == 's1':
            ixs = [np.arange(inst['_first'], inst['_last']) for inst in instruction]
            event_time = np.repeat(instruction['time'], instruction['_last'] - instruction['_first'])

            if len(ixs) == 0:
                self.pulses[primary_pulse].clear_pulse_cache()
            else:
                ixs = np.hstack(ixs).astype(int)
                # Some photons come too early or late, exceeding memory allocation
                nveto_cutoff = self.config.get('nveto_time_max_cutoff', int(1e6))
                mask = (self.timings[ixs] >= 0) & (self.timings[ixs] < nveto_cutoff)
                if (~mask).sum() > 0:
                    log.debug('Removing %d photons from optical input' % ((~mask).sum()))
                # By channel sorting is needed due to a speed boosting trick in pulse generation
                sorted_index = np.argsort(self.channels[ixs][mask])
                self.pulses[primary_pulse]._photon_channels = self.channels[ixs][mask][sorted_index]
                self.pulses[primary_pulse]._photon_timings = (self.timings[ixs][mask] + event_time[mask])[sorted_index]
                self.pulses[primary_pulse]()
        elif primary_pulse in ['pi_el', 'pe_el']:
            self.pulses[primary_pulse](instruction)
