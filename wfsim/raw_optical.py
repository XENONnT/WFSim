import numpy as np
from tqdm import tqdm

from .load_resource import load_config
from strax import exporter
import wfsim
export, __all__ = exporter()


@export
class RawDataOptical(wfsim.RawData):

    def __init__(self, config):
        self.config = config
        self.pulses = wfsim.Pulse(config)
        self.resource = load_config(self.config)

    def __call__(self, instructions, channels, timings, truth_buffer=None, **kwargs):
        if truth_buffer is None:
            truth_buffer = []

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
        inst_time = instructions['time']
        inst_queue = np.argsort(inst_time)
        inst_queue = np.split(inst_queue, np.where(np.diff(inst_time[inst_queue]) > rext)[0]+1)

        # Instruction buffer
        instb = np.zeros(100000, dtype=instructions.dtype)  # size ~ 1% of size of primary
        instb_filled = np.zeros_like(instb, dtype=bool)  # Mask of where buffer is filled

        # ik those are illegible, messy logic. lmk if you have a better way
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
                pbar.update(1)
            except:
                pass

            # B) Cluster instructions again with gap size <= rext
            instb_indx = np.where(instb_filled)[0]
            instb_type = instb[instb_indx]['type']
            instb_time = instb[instb_indx]['time']
            instb_queue = np.argsort(instb_time, kind='stable')
            instb_queue = np.split(instb_queue,
                                   np.where(np.diff(instb_time[instb_queue]) > rext)[0] + 1)

            # C) Push pulse cache out first if nothing comes right after them
            if np.min(instb_time) - self.last_pulse_end_time > rext and not np.isinf(self.last_pulse_end_time):
                self.digitize_pulse_cache()
                yield from self.ZLE()

            # D) Run all clusters before the current source
            stop_at_this_group = False
            for ibqs in instb_queue:
                for ptype in [1, 2]:  # S1 S2 PI Gate
                    mask = instb_type[ibqs] == ptype
                    if np.sum(mask) == 0: continue  # No such instruction type
                    instb_run = instb_indx[ibqs[mask]]  # Take hold of todo list

                    if self.symtype(ptype) in ['s1', 's2']:
                        stop_at_this_group = True  # Stop group iteration
                        _instb_run = np.array_split(instb_run, len(instb_run))
                    else:
                        _instb_run = [instb_run]  # Small trick to make truth small

                    # Run pulse simulation for real
                    for instb_run in _instb_run:
                        for instb_secondary in self.sim_data(instb[instb_run],
                                                             channels[instb[instb_run]['event_number'][0]],
                                                             instb[instb_run]['time'] + timings[instb[instb_run]['event_number'][0]]):
                            ib = np.where(~instb_filled)[0][:len(instb_secondary)]
                            instb[ib] = instb_secondary
                            instb_filled[ib] = True

                        if len(truth_buffer):  # Extract truth info
                            self.get_truth(instb[instb_run], truth_buffer)

                        instb_filled[instb_run] = False  # Free buffer AFTER copyting into truth buffer

                if stop_at_this_group: break
                self.digitize_pulse_cache()  # from pulse cache to raw data
                yield from self.ZLE()

            self.source_finished = len(inst_queue) == 0 and np.sum(instb_filled) == 0
        pbar.close()

    def sim_data(self, instruction, channels, timings):
        """Simulate a pulse according to instruction, and yield any additional instructions
        for secondary electron afterpulses.
        """
        # Any additional fields in instruction correspond to temporary
        # configuration overrides. No need to restore the old config:
        # next instruction cannot but contain the same fields.

        # Simulate the primary pulse
        self.pulses._photon_channels = channels
        self.pulses._photon_timings = timings
        self.pulses()

        # Append pulses we just simulated to our cache
        _pulses = getattr(self.pulses, '_pulses')
        if len(_pulses) > 0:
            self._pulses_cache += _pulses
            self.last_pulse_end_time = max(
                self.last_pulse_end_time,
                np.max([p['right'] for p in _pulses]) * self.config['sample_duration'])
        return []

    def get_truth(self, instruction, truth_buffer):
        """Write truth in the first empty row of truth_buffer

        :param instruction: Array of instructions that were simulated as a
        single cluster, and should thus get one line in the truth info.
        :param truth_buffer: Truth buffer to write in.
        """
        ix = np.argmin(truth_buffer['fill'])
        tb = truth_buffer[ix]
        peak_type = self.symtype(instruction['type'][0])
        pulse = self.pulses

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
