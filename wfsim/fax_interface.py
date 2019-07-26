import gzip
import logging
import pickle
from tqdm import tqdm

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

import strax
from straxen.common import get_resource, get_to_pe

from .itp_map import InterpolatingMap
from .fax import Peak, RawRecord

export, __all__ = strax.exporter()
log = logging.getLogger('SimulationCore')

inst_dtype = [('event_number', np.int), ('type', '<U2'), ('t', np.int), ('x', np.float32),
                                          ('y', np.float32), ('z', np.float32), ('amp', np.int), ('recoil', '<U2')]

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
def init_spe_scaling_factor_distributions(file):
    # Extract the spe pdf from a csv file into a pandas dataframe
    spe_shapes = pd.read_csv(file)

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
        return uniform_to_pe_arr


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

        return p


@export
class RawRecordsSimulator(object):
    def __init__(self, config):
        self.config = config
        self.record = RawRecord(config)
        self.results = []
        self.pulse_buffer = []

    def __call__(self, m_inst):

        for ix, ie in enumerate(tqdm(m_inst, desc='Simulating raw records')):
            # First keep track of in which event we are:
            if not hasattr(self, 'event'):
                self.event = ie['event_number']

            if self.event != ie['event_number']:
                yield self.fill_records(self.pulse_buffer)
                self.event = ie['event_number']
                self.pulse_buffer = []

            self.record(ie)
            self.pulse_buffer.append(self.record._raw_data)
            if ix == len(m_inst)-1:
                yield self.fill_records(self.pulse_buffer)
                self.pulse_buffer = []

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
        # In strax data, records are always stored
        # sorted, baselined and integrated
        records = strax.sort_by_time(records)
        strax.baseline(records)
        strax.integrate(records)
        # print("Returning %d records" % len(records))
        self.results = []
        return records



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
    strax.Option('s1_light_yield_map',default='https://raw.githubusercontent.com/XENONnT/strax_auxiliary_files/master/'
                                              'fax_files/XENON1T_s1_xyz_ly_kr83m_SR1_pax-680_fdc-3d_v0.json'),
    strax.Option('s1_pattern_map',default='https://github.com/XENONnT/strax_auxiliary_files/blob/master/'
                                          'fax_files/XENON1T_s1_xyz_patterns_interp_corrected_MCv2.1.0.json.gz?raw=true'),
    strax.Option('s2_light_yield_map',default='https://raw.githubusercontent.com/XENONnT/strax_auxiliary_files/master/'
                                              'fax_files/XENON1T_s2_xy_ly_SR1_v2.2.json'),
    strax.Option('s2_pattern_map',default='https://github.com/XENONnT/strax_auxiliary_files/blob/master/'
                                          'fax_files/XENON1T_s2_xy_patterns_top_corrected_MCv2.1.0.json.gz?raw=true'),
    strax.Option('fax_config', default='https://raw.githubusercontent.com/XENONnT/strax_auxiliary_files/master'
                                       '/fax_files/fax_config.json'),
    strax.Option('phase', default='liquid'),
    strax.Option('ele_afterpulse_file',
                 default='https://raw.githubusercontent.com/XENONnT/strax_auxiliary_files/'
                         'b5ddb62b7f8308181b5c82a33de5982bac1835df/fax_files/x1t_se_afterpulse_delaytime.pkl.gz'),
    strax.Option('pmt_afterpulse_file',
                 default='https://raw.githubusercontent.com/XENONnT/strax_auxiliary_files/'
                         'b5ddb62b7f8308181b5c82a33de5982bac1835df/fax_files/x1t_pmt_afterpulse_config.pkl.gz'),
    strax.Option('spe_file',
                 default='https://github.com/XENONnT/strax_auxiliary_files/blob/master/fax_files/XENON1T_spe_distributions.csv?raw=true'),
    strax.Option('noise_file',
                 default='https://raw.githubusercontent.com/XENONnT/strax_auxiliary_files/'
                         'b5ddb62b7f8308181b5c82a33de5982bac1835df/fax_files/170203_0850_00_small.npz'),
    strax.Option('kr83m_map',
                 default='https://raw.githubusercontent.com/XENONnT/strax_auxiliary_files/'
                         '4874cb458afcdc8f230464f1aa5bbe86cc1bc6ca/fax_files/Kr83m_Ddriven_per_pmt_params_dataframe.pkl'),
)
class FaxSimulatorPlugin(strax.Plugin):
    depends_on = tuple()

    # Cannot arbitrarily rechunk records inside events
    rechunk_on_save = False

    # Simulator uses iteration semantics, so the plugin has a state
    # TODO: this seems avoidable...
    parallel = False

    def setup(self):
        c = self.config
        c.update(get_resource(c['fax_config'], fmt='json'))

        # Gains
        c['to_pe'] = self.to_pe = get_to_pe(self.run_id, c['to_pe_file'])

        # Per-PMT single-PE distributions
        c['uniform_to_pe_arr'] = init_spe_scaling_factor_distributions(
            c['spe_file'])

        # PMT afterpulses
        c['uniform_to_pmt_ap'] = pickle.loads(gzip.decompress(get_resource(
            c['pmt_afterpulse_file'], fmt='binary')))

        # Single-electron "afterpulses" (photoionization / delayed extraction)
        c['uniform_to_ele_ap'] = pickle.loads(gzip.decompress(get_resource(
            c['ele_afterpulse_file'], fmt='binary')))

        # Real per-channel noise data
        c['noise_data'] = get_resource(
            c['noise_file'], fmt='npy')['arr_0'].flatten()

        # Something?
        all_params = get_resource(c['kr83m_map'], fmt='npy_pickle')
        c['params'] = all_params.loc[
            all_params.kr_run_id == 10,   # ???
            ['amp0', 'amp1', 'tao0', 'tao1', 'pmtx', 'pmty']].values.T

        # Light distribution maps
        for si in [1, 2]:
            c[f's{si}_light_map'] = InterpolatingMap(
                get_resource(c[f's{si}_light_yield_map']))
            c[f's{si}_pattern_map'] = InterpolatingMap(
                get_resource(c[f's{si}_pattern_map'],
                             fmt='binary'))

        if c['fax_file']:
            self.instructions = instruction_from_csv(c['fax_file'])
            c['nevents'] = len(self.instructions)
        else:
            self.instructions = rand_instructions(c['nevents'])

        self._setup_simulator()

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
        truth= inst_dtype)



    def _setup_simulator(self):
        self.sim_iter = RawRecordsSimulator(self.config)(self.instructions)

    # This lets strax figure out when we are done making chunks
    # [Jelle: 1 chunk = 1 event currently, right? Then why is the +1 needed?]
    # TODO: why does source_finished not get called?? Something is wrong.
    def is_ready(self, chunk_i):
        return chunk_i < self.config['nevents'] + 1

    def compute(self, chunk_i):
        return dict(raw_records = next(self.sim_iter),
                    truth = self.instructions)


@export
class PeaksFromFax(FaxSimulatorPlugin):
    provides = 'peaks'

    def infer_dtype(self):
        self.to_pe = get_to_pe(self.run_id, self.config['to_pe_file'])
        return strax.peak_dtype(len(self.to_pe))

    def _setup_simulator(self):
        self.simulator = PeakSimulator(self.config)

    # In this simulator 1 instruction = 1 chunk... not sure if this makes sense
    def is_ready(self, chunk_i):
        return chunk_i < len(self.instructions)

    def compute(self, chunk_i):
        return self.simulator(self.instructions[chunk_i])

