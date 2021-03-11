import numpy as np
from tqdm import tqdm

from .load_resource import load_config
from strax import exporter
import wfsim
export, __all__ = exporter()


PULSE_TYPE_NAMES = ('RESERVED', 'pulse', 'pulse', 'unknown', 'pi_el', 'pmt_ap', 'pe_el')

@export
class RawDataOptical(wfsim.RawData):

    def __init__(self, config):
        self.config = config
        self.pulses = dict(pulse=wfsim.Pulse(config),
            pi_el=wfsim.PhotoIonization_Electron(config),
            pe_el=wfsim.PhotoElectric_Electron(config),
            pmt_ap=wfsim.PMT_Afterpulse(config))
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
                        for instb_secondary in self.sim_data(instruction=instb[instb_run],
                                                             channels=channels[instb[instb_run]['event_number'][0]],
                                                             timings=instb[instb_run]['time'] + timings[instb[instb_run]['event_number'][0]]):
                            ib = np.where(~instb_filled)[0][:len(instb_secondary)]
                            instb[ib] = instb_secondary
                            instb_filled[ib] = True

                        if len(truth_buffer): # Extract truth info
                            self.get_truth(instb[instb_run], truth_buffer)

                        instb_filled[instb_run] = False # Free buffer AFTER copyting into truth buffer

                if stop_at_this_group: 
                    break
                self.digitize_pulse_cache() # from pulse cache to raw data
                yield from self.ZLE()
                
            self.source_finished = len(inst_queue) == 0 and np.sum(instb_filled) == 0
        pbar.close()

    @staticmethod
    def symtype(ptype):
        '''Little stupid we need this guy twice, but we need the different values for PULSE_TYPE_NAMES'''
        return PULSE_TYPE_NAMES[ptype]

    def sim_primary(self, primary_pulse, instruction, channels,timings):
        self.pulses[primary_pulse]._photon_channels = channels
        self.pulses[primary_pulse]._photon_timings = timings
        self.pulses[primary_pulse]()
