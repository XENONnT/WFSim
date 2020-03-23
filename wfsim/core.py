import logging

from numba import njit
import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm

from .load_resource import load_config
from strax import exporter
from . import units
from .utils import find_intervals_below_threshold

export, __all__ = exporter()
__all__.append('PULSE_TYPE_NAMES')

log = logging.getLogger('SimulationCore')

PULSE_TYPE_NAMES = ('RESERVED', 's1', 's2', 'unknown', 'pi_el', 'pmt_ap', 'pe_el')


@export
class Pulse(object):
    """Pulse building class"""

    def __init__(self, config):
        self.config = config
        self.config.update(getattr(self.config, self.__class__.__name__, {}))
        self.resource = load_config(config)

        self.init_pmt_current_templates()
        self.init_spe_scaling_factor_distributions()
        self.turned_off_pmts = np.arange(len(config['gains']))[np.array(config['gains']) == 0]
        
        self.clear_pulse_cache()


    def __call__(self):
        """
        PMTs' response to incident photons
        Use _photon_timings, _photon_channels to build pulses
        """
        if ('_photon_timings' not in self.__dict__) or \
                ('_photon_channels' not in self.__dict__):
            raise NotImplementedError
        
        # The pulse cache should be immediately transfered after call this function
        self.clear_pulse_cache()

        # Correct for PMT Transition Time Spread (skip for pmt afterpulses)
        if '_photon_gains' not in self.__dict__:
            self._photon_timings += np.random.normal(self.config['pmt_transit_time_mean'],
                                                     self.config['pmt_transit_time_spread'],
                                                     len(self._photon_timings))

        dt = self.config.get('sample_duration', 10) # Getting dt from the lib just once
        self._n_double_pe = self._n_double_pe_bot = 0 # For truth aft output

        counts_start = 0 # Secondary loop index for assigning channel
        for channel, counts in zip(*np.unique(self._photon_channels, return_counts=True)):
            # Use 'counts' amount of photon for this channel 
            _channel_photon_timings = self._photon_timings[counts_start:counts_start+counts]
            counts_start += counts
            if channel in self.turned_off_pmts: continue

            # Compute sample index（quotient）and reminder
            # Determined by division between photon timing and sample duraiton.
            _channel_photon_reminders = (_channel_photon_timings -
                                         np.floor(_channel_photon_timings / dt).astype(int) * dt).astype(int)
            _channel_photon_timings = np.floor(_channel_photon_timings / dt).astype(int)

            # If gain of each photon is not specifically assigned
            # Sample from spe scaling factor distribution and to individual gain
            # In contrast to pmt afterpulse that should have gain determined before this step
            if '_photon_gains' not in self.__dict__:
                if self.config['detector'] == 'XENON1T':
                    _channel_photon_gains = self.config['gains'][channel] \
                    * self.uniform_to_pe_arr[channel](np.random.random(len(_channel_photon_timings)))

                else:
                    _channel_photon_gains = self.config['gains'][channel] \
                    * self.uniform_to_pe_arr[0](np.random.random(len(_channel_photon_timings)))
                    
                # Add some double photoelectron emission by adding another sampled gain
                n_double_pe = np.random.binomial(len(_channel_photon_timings),
                                                 p=self.config['p_double_pe_emision'])
                self._n_double_pe += n_double_pe
                if channel in self.config['channels_bottom']:
                    self._n_double_pe_bot += n_double_pe

                _dpe_index = np.random.choice(np.arange(len(_channel_photon_timings)),
                                              size=n_double_pe, replace=False)
                if self.config['detector'] == 'XENON1T':
                    _channel_photon_gains[_dpe_index] += self.config['gains'][channel] \
                    * self.uniform_to_pe_arr[channel](np.random.random(n_double_pe))
                else:
                    _channel_photon_gains[_dpe_index] += self.config['gains'][channel] \
                    * self.uniform_to_pe_arr[0](np.random.random(n_double_pe))
            else:
                _channel_photon_gains = np.array(self._photon_gains[self._photon_channels == channel])

            # Build a simulated waveform, length depends on min and max of photon timings
            min_timing, max_timing = np.min(
                _channel_photon_timings), np.max(_channel_photon_timings)
            pulse_left = min_timing - int(self.config['samples_to_store_before'])
            pulse_right = max_timing + int(self.config['samples_to_store_after'])
            pulse_current = np.zeros(pulse_right - pulse_left + 1)

            Pulse.add_current(_channel_photon_timings - pulse_left,
                              _channel_photon_reminders, _channel_photon_gains,
                              self._pmt_current_templates, self._template_length,
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
        self._template_length = np.int(len(samples) - 1)

        templates = []
        for r in np.arange(0, sample_duration, pmt_pulse_time_rounding):
            pmt_current = np.diff(pe_pulse_function(samples - r)) / sample_duration  # pe / 10 ns
            # Normalize here to counter tiny rounding error from interpolation
            pmt_current *= (1 / sample_duration) / np.sum(pmt_current)  # pe / 10 ns
            templates.append(pmt_current)
        self._pmt_current_templates = np.array(templates)

        log.debug('Create spe waveform templates with %s ns resolution' % pmt_pulse_time_rounding)


    def init_spe_scaling_factor_distributions(self):
        # Extract the spe pdf from a csv file into a pandas dataframe
        spe_shapes = self.resource.photon_area_distribution

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

    @staticmethod
    @njit
    def add_current(_photon_timing_start, _reminder, _photon_gain,
                    _pmt_current_templates, _template_length,
                    pulse):
        #         """
        #         Simulate single channel waveform given the photon timings
        #         _photon_timing_start   - dim-1 integer array of photon timings in unit of samples
        #         _reminder              - dim-1 integer array of complimentary photon timings
        #         _photon_gain           - dim-1 float array of ph. 2 el. gain individual photons
        #         _pulse                 - waveform
        #         _pmt_current_templates - list of spe templates of different reminders
        #         The self argument is intentionally left out of this function.
        #         """
        for i in range(len(_photon_timing_start)):
            start = _photon_timing_start[i]
            pulse[start:start + _template_length] += \
                _pmt_current_templates[_reminder[i]] * _photon_gain[i]
        return pulse


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
                                 p=[singlet_ratio, 1 - singlet_ratio])
        return np.random.exponential(1, size) * delay


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
        if len(instruction.shape) < 1:
            # shape of recarr is a bit strange
            instruction = np.array([instruction])

        _, _, t, x, y, z, n_photons, recoil_type, *rest = [
            np.array(v).reshape(-1) for v in zip(*instruction)]
        
        positions = np.array([x, y, z]).T  # For map interpolation
        if self.config['detector']=='XENONnT':
            #for some reason light yield map crashes TODO
            ly = 1
        else:
            ly = self.resource.s1_light_yield_map(positions) * self.config['s1_detection_efficiency']
        n_photons = np.random.binomial(n=n_photons, p=ly)

        self._photon_timings = np.array([])
        list(map(self.photon_timings, t, n_photons, recoil_type))
        # The new way iterpolation is written always require a list
        self.photon_channels(positions, n_photons)

        super().__call__()

    def photon_channels(self, points, n_photons):
        channels = np.array(self.config['channels_in_detector']['tpc'])
        p_per_channel = self.resource.s1_pattern_map(points)
        p_per_channel[:, np.in1d(channels, self.turned_off_pmts)] = 0
        
        self._photon_channels = np.array([]).astype(int)
        for ppc, n in zip(p_per_channel, n_photons):
            self._photon_channels = np.append(self._photon_channels,
                    np.random.choice(
                        channels,
                        size=n,
                        p=ppc / np.sum(ppc),
                        replace=True))

    def photon_timings(self, t, n_photons, recoil_type):
        if n_photons == 0:
            return
        try:
            self._photon_timings = np.append(self._photon_timings,
                t + getattr(self, recoil_type.lower())(n_photons))
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
                                   p=[p_fraction, 1 - p_fraction])
        primary = timings == 0
        timings *= 1 / (1 - np.random.uniform(0, 1, size)) - 1
        timings = np.clip(timings, 0, self.config['maximum_recombination_time'])
        size_primary = len(timings[primary])
        timings[primary] += self.singlet_triplet_delays(
            size_primary, self.config['s1_ER_primary_singlet_fraction'])
        timings[~primary] += self.singlet_triplet_delays(
            size - size_primary, self.config['s1_ER_secondary_singlet_fraction'])
        return timings

    def nr(self, size):
        return self.singlet_triplet_delays(size, self.config['s1_NR_singlet_fraction'])


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

        positions = np.array([x, y]).T  # For map interpolation
        if self.config['detector'] == 'XENONnT':
            #Light yield map crashes, but the result is 1 for all positions in the tpc
            sc_gain = np.repeat(self.config['s2_secondary_sc_gain'], len(positions))
        else:
            sc_gain = self.resource.s2_light_yield_map(positions) * self.config['s2_secondary_sc_gain']

        # Average drift time of the electrons
        self.drift_time_mean = - z / \
            self.config['drift_velocity_liquid'] + self.config['drift_time_gate']

        # Absorb electrons during the drift
        electron_lifetime_correction = np.exp(- 1 * self.drift_time_mean /
            self.config['electron_lifetime_liquid'])
        cy = self.config['electron_extraction_yield'] * electron_lifetime_correction

        #why are there cy greater than 1? We should check this
        cy = np.clip(cy, a_min = 0, a_max = 1)

        n_electron = np.random.binomial(n=n_electron, p=cy)

        # Second generate photon timing and channel
        self.photon_timings(t, n_electron, z, sc_gain)
        self.photon_channels(positions)

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

        VG = self.config['anode_voltage'] / (1 + dL / dG / self.config['lxe_dielectric_constant'])
        E0 = VG / ((dG - rA) / rA + np.log(rA / rW))

        def Efield_r(r): return np.clip(E0 / r, E0 / rA, E0 / rW)

        def velosity_r(r): return alpha * Efield_r(r)

        def Yield_r(r): return Efield_r(r) / (units.kV / units.cm) - \
                               0.8 * self.config['pressure'] / units.bar

        r = np.linspace(dG, rW, 1000)
        dt = - np.diff(r)[0] / velosity_r(r)
        dy = Yield_r(r) / np.sum(Yield_r(r))

        uniform_to_emission_time = interp1d(np.cumsum(dy), np.cumsum(dt),
                                            bounds_error=False, fill_value=(0, sum(dt)))

        probabilities = 1 - np.random.uniform(0, 1, size=shape)
        return uniform_to_emission_time(probabilities)

    def electron_timings(self,t, n_electron, z, sc_gain):

        # Diffusion model from Sorensen 2011
        drift_time_mean = - z / \
                          self.config['drift_velocity_liquid'] + self.config['drift_time_gate']
        _drift_time_mean = np.clip(drift_time_mean, 0, None)
        drift_time_stdev = np.sqrt(2 * self.config['diffusion_constant_liquid'] * _drift_time_mean)
        drift_time_stdev /= self.config['drift_velocity_liquid']
        # Calculate electron arrival times in the ELR region
        _electron_timings = t + \
                            np.random.exponential(self.config['electron_trapping_time'], n_electron)
        if drift_time_stdev:
            _electron_timings += np.random.normal(drift_time_mean, drift_time_stdev, n_electron)

        self._electron_timings = np.append(self._electron_timings, _electron_timings)
        self._electron_gains = np.append(
            self._electron_gains, np.repeat(sc_gain, len(_electron_timings)))

    def photon_timings(self,t, n_electron, z, sc_gain):
        # First generate electron timinga
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
        self._photon_channels = np.array([])
        if self.config['detector'] == 'XENON1T':
            p_pattern = self.s2_pattern_map_pp(points)
        # Probability of each bottom channels
            p_per_channel_bottom = (1 - p_top) / len(self.config['channels_bottom']) \
                                   * np.ones_like(self.config['channels_bottom'])

            # Randomly assign to channel given probability of each channel
            # Sum probabilities over channels should be 1
            for u, n in zip(*np.unique(self._instruction, return_counts=True)):
                p_per_channel_top = p_pattern[u] / np.sum(p_pattern[u]) * p_top * (1 - p_random)
                channels = np.array(self.config['channels_in_detector']['tpc'])
                p_per_channel = np.concatenate([p_per_channel_top, p_per_channel_bottom])

                _photon_channels = np.random.choice(
                    channels,
                    size=n,
                    p=p_per_channel / np.sum(p_per_channel),
                    replace=True)

                self._photon_channels = np.append(self._photon_channels, _photon_channels)


        if self.config['detector'] == 'XENONnT':
            p_pattern = self.resource.s2_pattern_map(points)[0]

            for u, n in zip(*np.unique(self._instruction, return_counts=True)):
                channels = np.array(self.config['channels_in_detector']['tpc'])
                _photon_channels = np.random.choice(
                    channels,
                    size=n,
                    p=p_pattern / np.sum(p_pattern),
                    replace=True)
                self._photon_channels = np.append(self._photon_channels, _photon_channels)

        self._photon_channels = self._photon_channels.astype(int)

    def s2_pattern_map_pp(self, pos):
        if 'params' not in self.__dict__:
            all_map_params = self.resource.s2_per_pmt_params
            self.params = all_map_params.loc[all_map_params.kr_run_id == 10,
                                             ['amp0', 'amp1', 'tao0', 'tao1', 'pmtx', 'pmty']].values.T

        amp0, amp1, tao0, tao1, pmtx, pmty = self.params

        def rms(x, y): return np.sqrt(np.square(x) + np.square(y))

        distance = rms(pmtx - pos[:, 0].reshape([-1, 1]), pmty - pos[:, 1].reshape([-1, 1]))
        frac = amp0 * np.exp(- distance / tao0) + amp1 * np.exp(- distance / tao1)
        frac = (frac.T / np.sum(frac, axis=-1)).T

        return frac


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
        if len(signal_pulse._photon_timings) == 0: return []
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

        instruction['type'] = 4 # pi_el
        instruction['time'] = t_zeros
        instruction['x'], instruction['y'] = self._rand_position(n_electron)
        instruction['z'] = - (ap_delay - self.config['drift_time_gate']) * \
            self.config['drift_velocity_liquid']
        instruction['amp'] = 1

        return instruction

    def _rand_position(self, n):
        Rupper = 46

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
        if len(signal_pulse._photon_timings) == 0: return []
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
            low=0, high=len(signal_pulse._photon_timings),
            size=n_electron)]

        instruction = np.repeat(signal_pulse_instruction[0], n_electron)

        instruction['type'] = 6 # pe_el
        instruction['time'] = t_zeros
        instruction['x'], instruction['y'] = self._rand_position(n_electron)
        instruction['z'] = - (ap_delay - self.config['drift_time_gate']) * \
            self.config['drift_velocity_liquid']
        instruction['amp'] = 1

        return instruction

    def _rand_position(self, n):
        Rupper = 46

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
        super().__init__(config)

    def __call__(self, signal_pulse):
        if len(signal_pulse._photon_timings) == 0:
            self.clear_pulse_cache()
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
        self.element_list = self.resource.uniform_to_pmt_ap.keys()
        for element in self.element_list:
            delaytime_cdf = self.resource.uniform_to_pmt_ap[element]['delaytime_cdf']
            amplitude_cdf = self.resource.uniform_to_pmt_ap[element]['amplitude_cdf']

            # Assign each photon FRIST random uniform number rU0 from (0, 1] for timing
            rU0 = 1 - np.random.uniform(size=len(signal_pulse._photon_timings))

            # Select those photons with U <= max of cdf of specific channel
            cdf_max = delaytime_cdf[signal_pulse._photon_channels, -1]
            sel_photon_id = np.where(rU0 <= cdf_max * self.config['pmt_ap_modifier'])[0]
            if len(sel_photon_id) == 0: continue
            sel_photon_channel = signal_pulse._photon_channels[sel_photon_id]

            # Assign selected photon SECOND random uniform number rU1 from (0, 1] for amplitude
            rU1 = 1 - np.random.uniform(size=len(sel_photon_id))

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
                            - self.config['pmt_ap_t_modifier'])
                ap_amplitude = np.argmin(
                    np.abs(
                        amplitude_cdf[sel_photon_channel]
                        - rU1[:, None]), axis=-1)/100.

            self._photon_timings += (signal_pulse._photon_timings[sel_photon_id] + ap_delay).tolist()
            self._photon_channels += signal_pulse._photon_channels[sel_photon_id].tolist()
            self._photon_amplitude += np.atleast_1d(ap_amplitude).tolist()

        self._photon_timings = np.array(self._photon_timings)
        self._photon_channels = np.array(self._photon_channels).astype(int)
        self._photon_amplitude = np.array(self._photon_amplitude)
        self._photon_gain = np.array(self.config['gains'])[self._photon_channels] \
            * self._photon_amplitude


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

    def __call__(self, instructions, truth_buffer=None):
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
        inst_time = instructions['time'] + instructions['z']  / v * (instructions['type'] % 2 - 1)
        inst_queue = np.argsort(inst_time)
        inst_queue = np.split(inst_queue, np.where(np.diff(inst_time[inst_queue]) > rext)[0]+1)

        # Instruction buffer
        instb = np.zeros(100000, dtype=instructions.dtype) # size ~ 1% of size of primary
        instb_filled = np.zeros_like(instb, dtype=bool) # Mask of where buffer is filled

        # ik those are illegible, messy logic. lmk if you have a better way
        pbar = tqdm(total=len(inst_queue), desc='Simulating Raw Records')
        while not self.source_finished:

            # A) Add a new instruction into buffer
            try:
                ixs = inst_queue.pop(0) # The index from original instruction list
                self.source_finished = len(inst_queue) == 0
                assert len(np.where(~instb_filled)[0]) > len(ixs), "Run out of instruction buffer"
                ib = np.where(~instb_filled)[0][:len(ixs)] # The index of first empty slot in buffer
                instb[ib] = instructions[ixs]
                instb_filled[ib] = True
                pbar.update(1)
            except: pass

            # B) Cluster instructions again with gap size <= rext
            instb_indx = np.where(instb_filled)[0]
            instb_type = instb[instb_indx]['type']
            instb_time = instb[instb_indx]['time'] + instb[instb_indx]['z']  \
                / v * (instb_type % 2 - 1)
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
                for ptype in [1, 2, 4, 6]: # S1 S2 PI Gate
                    mask = instb_type[ibqs] == ptype
                    if np.sum(mask) == 0: continue # No such instruction type
                    instb_run = instb_indx[ibqs[mask]] # Take hold of todo list

                    if self.symtype(ptype) in ['s1', 's2']:
                        stop_at_this_group = True # Stop group iteration
                        _instb_run = np.array_split(instb_run, len(instb_run))
                    else: _instb_run = [instb_run] # Small trick to make truth small

                    # Run pulse simulation for real
                    for instb_run in _instb_run:
                        for instb_secondary in self.sim_data(instb[instb_run]):
                            ib = np.where(~instb_filled)[0][:len(instb_secondary)]
                            instb[ib] = instb_secondary
                            instb_filled[ib] = True

                        if len(truth_buffer): # Extract truth info
                            self.get_truth(instb[instb_run], truth_buffer)

                        instb_filled[instb_run] = False # Free buffer AFTER copyting into truth buffer

                if stop_at_this_group: break
                self.digitize_pulse_cache() # from pulse cache to raw data
                yield from self.ZLE()
                
            self.source_finished = len(inst_queue) == 0 and np.sum(instb_filled) == 0
        pbar.close()

    @staticmethod
    def symtype(ptype):
        return PULSE_TYPE_NAMES[ptype]

    def sim_data(self, instruction):
        """Simulate a pulse according to instruction, and yield any additional instructions
        for secondary electron afterpulses.
        """
        # Any additional fields in instruction correspond to temporary
        # configuration overrides. No need to restore the old config:
        # next instruction cannot but contain the same fields.
        if len(instruction) > 8:
            for par in instruction.dtype.names:
                if par in self.config:
                    self.config[par] = instruction[par]

        # Simulate the primary pulse
        primary_pulse = self.symtype(instruction['type'][0])
        self.pulses[primary_pulse](instruction)

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
                    np.max([p['right'] for p in _pulses]) * 10)

        # Make new instructions for electron afterpulses, if requested
        if primary_pulse in ['s1', 's2']:
            if self.config.get('enable_electron_afterpulses', True):
                yield self.pulses['pi_el'].generate_instruction(
                    self.pulses[primary_pulse], instruction)
                if primary_pulse in ['s2']: # Only add gate ap to s2
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
            assert self.right - self.left < 200000, "Pulse cache too long"

            if self.left % 2 != 0: self.left -= 1 # Seems like a digizier effect

            # Use noise array to pave the fundation of the pulses
            #self._raw_data = self.get_real_noise(self.right - self.left + 1)
            self._raw_data = np.zeros((len(self.config['channels_in_detector']['tpc']),
                self.right - self.left + 1), dtype=('<i8'))

            for ix, _pulse in enumerate(self._pulses_cache):
                # Could round instead of trunc... no one cares!
                adc_wave = - np.trunc(_pulse['current'] * self.current_2_adc).astype(int)
                self._raw_data[_pulse['channel'],
                    _pulse['left'] - self.left:_pulse['right'] - self.left + 1] += adc_wave
                
            self._pulses_cache = [] # Memory control

            # Digitizers have finite number of bits per channel, so clip the signal.
            self._raw_data += self.config['digitizer_reference_baseline']
            self._raw_data[self._raw_data < 0] = 0
            # Hopefully (peak downward) waveform won't exceed upper limit
            # self._raw_data = np.clip(self._raw_data, 0, 2 ** (self.config['digitizer_bits']))

    def ZLE(self):
        """
        Modified software zero lengh encoding, coverting WFs into pulses (XENON definition)
        """
        # Ask for memory allocation just once
        if 'zle_intervals_buffer' not in self.__dict__:
            self.zle_intervals_buffer = -1 * np.ones((50000, 2), dtype=np.int64)            
        
        for ix, data in enumerate(self._raw_data):
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
                yield ix, self.left + itv[0], self.left + itv[1], data[itv[0]:itv[1]+1]

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
            # Set an endtime (if times has no length)
            tb['endtime'] = np.mean(instruction['time'])
            if len(times):
                tb[f'n_{quantum}'] = len(times)
                tb[f't_mean_{quantum}'] = np.mean(times)
                tb[f't_first_{quantum}'] = np.min(times)
                tb[f't_last_{quantum}'] = np.max(times)
                tb[f't_sigma_{quantum}'] = np.std(times)
                tb['endtime'] = tb['t_last_photon']
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

        channels = getattr(pulse, '_photon_channels', [])
        n_dpe = getattr(pulse, '_n_double_pe', 0)
        n_dpe_bot = getattr(pulse, '_n_double_pe_bot', 0)
        tb['n_photon'] += n_dpe
        tb['n_photon'] -= np.sum(np.isin(channels, getattr(pulse, 'turned_off_pmts', [])))

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

    def get_real_noise(self, length):
        """
        Get chunk(s) of noise sample from real noise data
        """
        # Randomly choose where in to start copying
        real_data_sample_size = len(self.resource.noise_data)
        id_t = np.random.randint(0, real_data_sample_size - length)
        data = self.resource.noise_data
        result = data[id_t:id_t + length]

        return result
