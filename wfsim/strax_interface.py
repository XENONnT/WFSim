import gzip
import logging
import pickle
from tqdm import tqdm

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


import strax
from straxen.common import get_resource, get_to_pe

from utils import InterpolatingMap
from utils import get_resource as get_res
from core import Peak, RawRecord

export, __all__ = strax.exporter()
__all__ += ['inst_dtype','truth_extra_dtype']

log = logging.getLogger('SimulationCore')

inst_dtype = [('event_number', np.int), ('type', '<U2'), ('t', np.int), ('x', np.float32),
                                          ('y', np.float32), ('z', np.float32), ('amp', np.int), ('recoil', '<U2')]

truth_extra_dtype = [('n_photons', np.float),('t_mean_photons', np.float),('t_first_photons', np.float),
                     ('t_last_photons', np.float),('t_sigma_photons', np.float),('n_electrons', np.float),
                     ('t_mean_electrons', np.float),('t_first_electrons', np.float),
                     ('t_last_electrons', np.float),('t_sigma_electrons', np.float),]

def instruction_from_csv(file):
    instructions = np.genfromtxt(file, delimiter = ',', dtype = inst_dtype)
    return instructions

@export
def rand_instructions(n=10):
    nelectrons = 10 ** (np.random.uniform(1, 3, n))

    instructions = np.zeros(2 * n, dtype=  inst_dtype)

    instructions['event_number'] = np.repeat(np.arange(n), 2)
    instructions['type'] = np.tile(['s1', 's2'], n)
    instructions['t'] = np.repeat(5.e7+5.e7*np.arange(n), 2)
    r = np.sqrt(np.random.uniform(0, 2500, n))
    t = np.random.uniform(-np.pi, np.pi, n)

    instructions['x'] = np.repeat(r * np.cos(t), 2)
    instructions['y'] = np.repeat(r * np.sin(t), 2)
    instructions['z'] = np.repeat(np.random.uniform(-100, 0, n), 2)
    instructions['amp'] = np.vstack(
        [np.random.uniform(10, 500, n), nelectrons]).T.flatten().astype(int)
    instructions['recoil'] = ['er' for i in range(n * 2)]

    return instructions


@export
class PeakSimulator(object):

    def __init__(self, config):
        self.config = config
        self.peak = Peak(config)

    def __call__(self, master_instruction):
        m_inst = master_instruction

        p = np.zeros(1, dtype=strax.peak_dtype(len(self.config['to_pe'])))
        p['channel'] = -1
        p['dt'] = self.config['sample_duration']

        p_length = len(p['data'][0])
        i = 0
        ie = m_inst

        # print(ie)
        self.peak(ie)
        truth = self.peak._truth
        p[i]['time'] = ie['t'] + self.peak.start + self.peak.event_duration / 2.
        swv_buffer_size = self.peak.event_duration
        swv_buffer = np.zeros(swv_buffer_size,dtype=np.int32)
        # Now we need to get the sumwaveform out of the data, we just copy a large part of sum_waveform
        # Now we need to get the data out of the event into the peak
        p[i]['area_per_channel'] = np.sum(self.peak._raw_data['pulse'],axis=1)*self.config['to_pe']

        swv_buffer = np.sum(self.peak._raw_data['pulse']*
                            np.tile(self.config['to_pe'],(self.peak.event_duration,1)).T, axis=0)

        downs_f = int(np.ceil(swv_buffer_size / p_length))
        if downs_f > 1:
            #             # Compute peak length after downsampling.
            #             # We floor rather than ceil here, potentially cutting off
            #             # some samples from the right edge of the peak.
            #             # If we would ceil, the peak could grow larger and
            #             # overlap with a subsequent next peak, crashing strax later.
            new_ns = p[i]['length'] = int(np.floor(swv_buffer_size / downs_f))
            p[i]['data'][:new_ns] = \
                swv_buffer[:new_ns * downs_f].reshape(-1, downs_f).sum(axis=1)
            p[i]['dt'] *= downs_f
        else:
            p[i]['data'][:swv_buffer_size] = swv_buffer[:swv_buffer_size]
            p[i]['length'] = swv_buffer_size
        #
        #         # Store the total area and saturation count
        p[i]['area'] = np.sum(p[i]['area_per_channel'])
        swv_buffer[:] = 0

        strax.compute_widths(p)
        sort_key = p['time'] - p['time'].min()
        sort_i = np.argsort(sort_key)
        p = p[sort_i]

        return p, truth


@export
class RawRecordsSimulator(object):
    def __init__(self, config):
        self.config = config
        self.record = RawRecord(config)
        self.results = []
        self.pulse_buffer = []
        self.truth = []

    def __call__(self, m_inst):

        for ix, ie in enumerate(tqdm(m_inst, desc='Simulating raw records')):
            # First keep track of in which event we are:
            if not hasattr(self, 'event'):
                self.event = ie['event_number']

            if self.event != ie['event_number']:
                yield self.fill_records(self.pulse_buffer)
                self.event = ie['event_number']
                self.pulse_buffer = []
                self.truth = []

            self.record(ie)
            self.pulse_buffer.append(self.record._raw_data)
            self.truth.append(self.record._truth)
            if ix == len(m_inst)-1:
                yield self.fill_records(self.pulse_buffer)
                self.pulse_buffer = []
                self.truth = []

    def fill_records(self, p_b):
        samples_per_record = strax.DEFAULT_RECORD_LENGTH

        records_needed = np.sum([np.sum(pulses[i]['rec_needed']) for pulses in p_b for i in range(len(pulses))])
        rr = np.zeros(int(records_needed),dtype = strax.record_dtype())
        output_record_index = 0

        for pulses in p_b:
            for i in range(len(pulses)):
                for p in pulses[i]:
                    p_length = p['right'] - p['left'] + 2 * self.config['trigger_window']
                    n_records = int(np.ceil(p_length /  samples_per_record))
                    for rec_i in range(n_records):
                        if output_record_index == records_needed: #TODO this cannot be the right way to fix this.
                            print(
                                f'output_record_index: {output_record_index} is larger then the total records_needed: {records_needed}')
                            log.info(f'output_record_index: {output_record_index} is larger then the total records_needed: {records_needed}')
                            continue                                #Why is output_record_index, sometimes, larger then records needed?
                        r = rr[output_record_index]
                        r['channel'] = p['channel']
                        r['dt'] = self.config['sample_duration']
                        r['pulse_length'] = p_length
                        r['record_i'] = rec_i
                        r['time'] = (p['left']-self.config['trigger_window']) * r['dt'] + rec_i * samples_per_record * r['dt']
                        if rec_i != n_records  -  1:
                            #The pulse doesn't fit on one record, so store a full chunk
                            n_store = samples_per_record
                            assert p_length > samples_per_record * (rec_i + 1)

                        else:
                            n_store = p_length - samples_per_record * rec_i

                        assert 0 <= n_store <= samples_per_record
                        r['length'] = n_store
                        offset = rec_i  * samples_per_record

                        r['data'][:n_store] = p['pulse'][offset: offset + n_store]
                        output_record_index +=1

        self.results.append(rr)
        y = self.finish_results()
        return y

    def finish_results(self):
        records = np.concatenate(self.results)
        truth = np.concatenate(self.truth)
        # In strax data, records are always stored
        # sorted, baselined and integrated
        records = strax.sort_by_time(records)
        strax.baseline(records)
        strax.integrate(records)
        # print("Returning %d records" % len(records))
        self.results = []
        return records, truth



@strax.takes_config(
    strax.Option('fax_file', default=None, track=True,
                 help="Directory with fax instructions"),
    strax.Option('nevents',default = 100,track=False,
                help="Number of random events to generate if no instructions are provided"),
    strax.Option('digitizer_trigger',default = 15,track=True,
                 help="Minimum current in ADC to be measured by the digitizer in order to be written"),
    strax.Option('trigger_window', default = 50, track=True,
                 help='Digitizer trigger window'),
    strax.Option('to_pe_file',
        default='https://raw.githubusercontent.com/XENONnT/strax_auxiliary_files/master/to_pe.npy',
        help='link to the to_pe conversion factors'),
    strax.Option('fax_config', default='/project2/lgrandi/zhut/sim/WFSimDev/fax_config.json'),
)
class FaxSimulatorPlugin(strax.Plugin):
    depends_on = tuple()

    # Cannot arbitrarily rechunk records inside events
    rechunk_on_save = False

    # Simulator uses iteration semantics, so the plugin has a state
    # TODO: this seems avoidable...
    parallel = False

    # TODO: this state is needed for sorting checks,
    # but it prevents prevent parallelization
    last_chunk_time = -999999999999999

    def setup(self):
        c = self.config
        c.update(get_res(c['fax_config'], fmt='json'))

        # Gains
        c['to_pe'] = self.to_pe = get_to_pe(self.run_id, c['to_pe_file'])

        if c['fax_file']:
            self.instructions = instruction_from_csv(c['fax_file'])
            c['nevents'] = len(self.instructions)
        else:
            self.instructions = rand_instructions(c['nevents'])

        self._setup_simulator()

    def _sort_check(self, result):
        if result['time'][0] < self.last_chunk_time + 5000:
            raise RuntimeError(
                "Simulator returned chunks with insufficient spacing. "
                f"Last chunk's max time was {result['time'][0]}, "
                f"this chunk's first time is {self.last_chunk_time}.")
        if np.diff(result['time']).min() < 0:
            raise RuntimeError("Simulator returned non-sorted records!")
        self.last_chunk_time = result['time'].max()

    def source_finished(self):
        return True


@export
class RawRecordsFromFax(FaxSimulatorPlugin):
    provides = ('raw_records','truth')

    data_kind = dict(
        raw_records='raw_records',
        truth='truth',)

    dtype = dict(
        raw_records = strax.record_dtype(),
        truth = inst_dtype + truth_extra_dtype)

    def _setup_simulator(self):
        self.sim_iter = RawRecordsSimulator(self.config)(self.instructions)

    # This lets strax figure out when we are done making chunks
    def is_ready(self, chunk_i):
        return chunk_i < self.config['nevents']

    def compute(self, chunk_i):
        try:
            result, t = next(self.sim_iter)
        except StopIteration:
            raise RuntimeError("Bug in chunk count computation")
        self._sort_check(result)
        return dict(raw_records = result,
                    truth = t)


@export
class PeaksFromFax(FaxSimulatorPlugin):
    provides = ('peaks','truth')
    data_kind = dict(
        peaks = 'peaks',
        truth = 'truth',)

    def infer_dtype(self):
        self.to_pe = get_to_pe(self.run_id, self.config['to_pe_file'])
        truth_dtype = inst_dtype + truth_extra_dtype
        return dict(
            peaks = strax.peak_dtype(len(self.to_pe)),
            truth = truth_dtype)

    def _setup_simulator(self):
        self.simulator = PeakSimulator(self.config)

    # In this simulator 1 instruction = 1 chunk... not sure if this makes sense
    def is_ready(self, chunk_i):
        return chunk_i < len(self.instructions)

    def compute(self, chunk_i):
        result, t = self.simulator(self.instructions[chunk_i])
        # self._sort_check(result)
        return dict(peaks = result,
                    truth = t)
