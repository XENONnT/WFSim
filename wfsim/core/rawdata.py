import logging
from numba import njit
import numpy as np
from strax import exporter
from strax.utils import tqdm
from .afterpulse import PhotoIonization_Electron, PhotoElectric_Electron, PMT_Afterpulse
from .pulse import Pulse
from .s1 import S1
from .s2 import S2
from ..load_resource import load_config
from ..utils import find_intervals_below_threshold


export, __all__ = exporter()
logging.basicConfig(handlers=[logging.StreamHandler()])
log = logging.getLogger('wfsim.core')
log.setLevel('WARNING')


__all__.append('PULSE_TYPE_NAMES')
PULSE_TYPE_NAMES = ('RESERVED', 's1', 's2', 'unknown', 'pi_el', 'pmt_ap', 'pe_el')


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

        save_full_truth = self.config.get('save_full_truth', True)
        log.debug(f"save_full_truth : {save_full_truth}")
        
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
                         (instb[instb_indx]['z'] / v * (instb_type % 2 - 1)).astype(np.int64)
            instb_queue = np.argsort(instb_time, kind='stable')
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
                        if save_full_truth:
                            instb_run = np.split(instb_indx[ibqs[mask]], len(instb_indx[ibqs[mask]]))
                        else:
                            # Group S1 within 100 ns apart, truth info would be summarized within the group
                            instb_run = np.split(instb_indx[ibqs[mask]],
                                                 np.where(np.diff(instb_time[ibqs[mask]]) > 100)[0] + 1)
                    elif ptype == 2:
                        stop_at_this_group = True
                        if save_full_truth:
                            instb_run = np.split(instb_indx[ibqs[mask]], len(instb_indx[ibqs[mask]]))
                        else:
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
        
        self.digitize_pulse_cache()  # To make sure we digitize all data
        yield from self.ZLE()
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
                if primary_pulse in ['s2']:  # Our pi model is valid for only S2s
                    yield self.pulses['pi_el'].generate_instruction(
                        self.pulses[primary_pulse], instruction)
            if self.config.get('enable_gate_afterpulses', False):
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
                adc_wave = - np.around(_pulse['current'] * self.current_2_adc).astype(np.int64)
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
                               noise_data_length=len(self.resource.noise_data),
                               noise_data_channels=len(self.resource.noise_data[0]))

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
                
        self.get_mean_xy_electron(peak_type, instruction, tb)
        
        # Endtime is the end of the last pulse
        if np.isnan(tb['t_last_photon']):
            tb['endtime'] = instruction['time'][0]
        else:
            tb['endtime'] = tb['t_last_photon'] + \
                (self.config['samples_before_pulse_center'] + self.config['samples_after_pulse_center'] + 1) \
                * self.config['sample_duration']

        # Copy single valued fields directly from pulse class
        for field in ['n_pe', 'n_pe_trigger', 'n_photon', 'n_photon_trigger', 'raw_area', 'raw_area_trigger']:
            for suffix in ['', '_bottom']:
                tb[field+suffix] = getattr(pulse, '_' + field + suffix, 0)

        # Summarize the instruction cluster in one row of the truth file
        for field in instruction.dtype.names:
            value = instruction[field]
            if len(instruction) > 1 and field in 'xyz':
                tb[field] = np.mean(value)
            elif len(instruction) > 1 and field == 'amp':
                tb[field] = np.sum(value)
            else:
                # Cannot summarize intelligently: just take the first value
                tb[field] = value[0]

        # Signal this row is now filled, so it won't be overwritten
        tb['fill'] = True
        
    def get_mean_xy_electron(self, peak_type, instruction, tb):
        
        if peak_type == 's2' and self.config.get('field_distortion_model', "none") in ['comsol', 'inverse_fdc']:
            if self.config.get('field_distortion_model', "none") == 'comsol':
                _, xy_tmp = self.pulses['s2'].field_distortion_comsol(instruction['x'], instruction['y'], instruction['z'], self.resource)
            elif self.config.get('field_distortion_model', "none") == 'inverse_fdc':
                _, xy_tmp = self.pulses['s2'].inverse_field_distortion_correction(instruction['x'], instruction['y'], instruction['z'], self.resource)
            tb['x_mean_electron'] = np.mean(xy_tmp.T[0])
            tb['y_mean_electron'] = np.mean(xy_tmp.T[1])
        else:
            tb['x_mean_electron'] = np.nan
            tb['y_mean_electron'] = np.nan

    @staticmethod
    @njit
    def sum_signal(adc_wave, left, right, sum_template):
        sum_template[left:right] += adc_wave
        return sum_template

    @staticmethod
    @njit
    def add_noise(data, channel_mask, noise_data, noise_data_length, noise_data_channels):
        """
        Get chunk(s) of noise sample from real noise data
        """
        if channel_mask['mask'].sum() == 0:
            return

        left = np.min(channel_mask['left'][channel_mask['mask']])
        right = np.max(channel_mask['right'][channel_mask['mask']])

        if noise_data_length-right+left-1 < 0:
            ix_rand = np.random.randint(low=0, high=noise_data_length-1)
        else:
            ix_rand = np.random.randint(low=0, high=noise_data_length-right+left-1)

        for ch in range(data.shape[0]):
            # In case adding noise to he channels is not supported
            if ch >= noise_data_channels:
                continue

            if not channel_mask['mask'][ch]:
                continue

            left, right = channel_mask['left'][ch], channel_mask['right'][ch]
            for ix_data in range(left, right+1):
                ix_noise = ix_rand + ix_data - left
                if ix_data >= len(data[ch]):
                    # Don't create value-errors
                    continue

                if ix_noise >= noise_data_length:
                    ix_noise -= noise_data_length * (ix_noise // noise_data_length)

                data[ch, ix_data] += noise_data[ix_noise, ch]

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
