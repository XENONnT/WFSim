import logging
import numpy as np
from strax import exporter
from .pulse import Pulse
from .s2 import S2

export, __all__ = exporter()
logging.basicConfig(handlers=[logging.StreamHandler()])
log = logging.getLogger('wfsim.core')
log.setLevel('WARNING')


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

        ap_delay = delaytime_pmf_hist.get_random(n_electron)
        # Reasonably bin delay time that would be diffuse out together
        ap_delay_i, n_electron_i = self._reduce_instruction_timing(
            ap_delay,
            delaytime_pmf_hist)
        n_instruction = len(ap_delay_i)

        # Randomly select original photon as time zeros
        t_zeros = signal_pulse._photon_timings[np.random.randint(
            low=0, high=len(signal_pulse._photon_timings),
            size=n_instruction)]

        instruction = np.repeat(signal_pulse_instruction[0], n_instruction)

        instruction['type'] = 4  # pi_el
        instruction['time'] = t_zeros - self.config['drift_time_gate']
        instruction['x'], instruction['y'] = self._rand_position(n_instruction)
        instruction['z'] = - ap_delay_i * self.config['drift_velocity_liquid']
        instruction['amp'] = n_electron_i

        return instruction

    def _reduce_instruction_timing(self, ap_delay, delaytime_pmf_hist):
        # Binning the delay time, so electron timing within each
        # will be diffused to fill the whole bin
        
        delaytime_spread = np.sqrt(2 * self.config['diffusion_constant_longitudinal']\
                                   * delaytime_pmf_hist.bin_centers)
        delaytime_spread /= self.config['drift_velocity_liquid']

        coarse_time, coarse_time_i = [], 100 # Start at 100ns, as its smaller than single electron width
        while coarse_time_i < delaytime_pmf_hist.bin_centers[-1]:
            coarse_time.append(coarse_time_i)
            coarse_time_i += delaytime_spread[np.argmin(np.abs(coarse_time_i - delaytime_pmf_hist.bin_centers))]
        coarse_time = np.array(coarse_time)

        idx = np.digitize(ap_delay[ap_delay < coarse_time[-1]], coarse_time)
        idxs, n = np.unique(idx, return_counts=True)
        _ap_delay = coarse_time[idxs]
        return _ap_delay, n

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

            delaytime_bin_size = resource.uniform_to_pmt_ap[element]['delaytime_bin_size']
            amplitude_bin_size = resource.uniform_to_pmt_ap[element]['amplitude_bin_size']

            # Assign each photon FRIST random uniform number rU0 from (0, 1] for timing
            rU0 = 1 - np.random.rand(len(signal_pulse._photon_timings))

            # delaytime_cdf is intentionally not normalized to 1 but the probability of the AP 
            prob_ap = delaytime_cdf[signal_pulse._photon_channels, -1]
            if prob_ap.max() * config['pmt_ap_modifier'] > 0.5:
                prob = prob_ap.max() * config['pmt_ap_modifier']
                log.warning(f'PMT after pulse probability is {prob} larger than 0.5?')

            # Scaling down (up) rU0 effectivly increase (decrease) the ap rate
            rU0 /= config['pmt_ap_modifier']

            # Double the probability for those photon emitting dpe
            rU0[signal_pulse._photon_is_dpe] /= 2

            # Select those photons with U <= max of cdf of specific channel
            sel_photon_id = np.where(rU0 <= prob_ap)[0]
            if len(sel_photon_id) == 0:
                continue
            sel_photon_channel = signal_pulse._photon_channels[sel_photon_id]

            # Assign selected photon SECOND random uniform number rU1 from (0, 1] for amplitude
            rU1 = 1 - np.random.rand(len(sel_photon_channel))

            # The map is made so that the indices are delay time in unit of ns
            if 'Uniform' in element:
                ap_delay = (np.random.uniform(delaytime_cdf[sel_photon_channel, 0], 
                                             delaytime_cdf[sel_photon_channel, 1])
                            * delaytime_bin_size)
                ap_amplitude = np.ones_like(ap_delay)
            else:
                ap_delay = (np.argmin(
                    np.abs(
                        delaytime_cdf[sel_photon_channel]
                        - rU0[sel_photon_id][:, None]), axis=-1) * delaytime_bin_size
                            - config['pmt_ap_t_modifier'])
                if len(amplitude_cdf.shape) == 2:
                    ap_amplitude = np.argmin(
                        np.abs(
                            amplitude_cdf[sel_photon_channel]
                            - rU1[:, None]), axis=-1) * amplitude_bin_size
                else:
                    ap_amplitude = np.argmin(
                        np.abs(
                            amplitude_cdf[None, :]
                            - rU1[:, None]), axis=-1) * amplitude_bin_size

            _photon_timings.append(signal_pulse._photon_timings[sel_photon_id] + ap_delay)
            _photon_channels.append(signal_pulse._photon_channels[sel_photon_id])
            _photon_amplitude.append(np.atleast_1d(ap_amplitude))

        if len(_photon_timings) > 0:
            _photon_timings = np.hstack(_photon_timings)
            _photon_channels = np.hstack(_photon_channels).astype(np.int64)
            _photon_amplitude = np.hstack(_photon_amplitude)
            _photon_gains = np.array(config['gains'])[_photon_channels] * _photon_amplitude

            sorted_index = np.argsort(_photon_channels)
            return _photon_timings[sorted_index], _photon_channels[sorted_index], _photon_gains[sorted_index]

        else:
            return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0)
