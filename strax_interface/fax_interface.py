import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from numba import int64, float64, guvectorize

from tqdm import tqdm
import strax
from straxen.common import get_resource, get_to_pe
from straxen import InterpolatingMap
from straxen.plugins.fax import Peak, RawRecord

export, __all__ = strax.exporter()

def rand_instructions(n=10):
    nelectrons = 10 ** (np.random.uniform(4, 5, n))

    instructions = np.zeros(2 * n, dtype=[('event_number', np.int), ('type', '<U2'), ('t', np.int), ('x', np.float32),
                                          ('y', np.float32), ('z', np.float32), ('amp', np.int), ('recoil', '<U2')])

    instructions['event_number'] = np.repeat(np.arange(n), 2)
    instructions['type'] = np.tile(['s1', 's2'], n)
    instructions['t'] = np.repeat(1.e3+5.e6*np.arange(n), 2)
    r = np.sqrt(np.random.uniform(0, 2500, n))
    t = np.random.uniform(-np.pi, np.pi, n)

    instructions['x'] = np.repeat(r * np.cos(t), 2)
    instructions['y'] = np.repeat(r * np.sin(t), 2)
    instructions['z'] = np.repeat(np.random.uniform(-100, 0, n), 2)
    instructions['amp'] = np.vstack(
        [np.random.uniform(3000, 3001, n), nelectrons]).T.flatten().astype(int)
    instructions['recoil'] = ['er' for i in range(n * 2)]

    return instructions



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


class PeakSimulator(object):

    def __init__(self, config):
        self.config = config
        self.peak = Peak(config)

    def __call__(self, master_instruction):
        m_inst = master_instruction
        p = np.zeros(len(m_inst), dtype=strax.peak_dtype(len(self.config['to_pe'])))
        p['channel'] = -1
        p['dt'] = self.config['sample_duration']

        p_length = len(p['data'][0])
        for i, ie in tqdm(enumerate(m_inst)):
            print(ie)
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


class RawRecordsSimulator(object):
    def __init__(self, config):
        self.config = config
        self.record = RawRecord(config)
        self.results = []
        self.pulse_buffer = dict()

    def __call__(self, m_inst):

        for ix, ie in tqdm(enumerate(m_inst)):
            # First keep track of in which event we are:
            try:
                self.event
            except:
                self.event = ie['event_number']
                self.t = ie['t']

            if self.event != ie['event_number']:
                yield self.store_buffer(self.pulse_buffer,self.t)

                self.t = ie['t']
                self.event = ie['event_number']
                self.pulse_buffer = dict()

            print(ie)
            self.record(ie)

            self.pulse_buffer[ie['type']] = self.record._raw_data

            if ix == len(m_inst)-1:
                yield self.store_buffer(self.pulse_buffer, ie['t'])

    def store_buffer(self, p_b, t):
        samples_per_record = strax.DEFAULT_RECORD_LENGTH
        event_seperation = int(2e6)

        r_l = self.config['sample_duration'] * samples_per_record

        for ch in np.arange(len(self.config['to_pe'])):
            rr = np.zeros(int(np.floor(event_seperation / r_l)), dtype=strax.record_dtype())
            rr['data'] = self.get_real_noise(rr)

            rr['dt'] = self.config['sample_duration']
            rr['channel'] = ch
            rr['time'] = np.arange(len(rr)) * r_l + t

            for s in p_b.keys():
                p_ = p_b[s]
                p = p_[p_['channel'] == ch]
                if np.sum(p['pulse']) >= 0:
                    continue
                s_r, o_r = int(np.floor(p['left'][0] / samples_per_record)), np.mod(p['left'][0], samples_per_record)
                p_length = p['right'][0] - p['left'][0] + 1
                n_records = int(np.ceil((o_r + p_length) / samples_per_record))

                o_p = 0
                n_store = 0
                for rec_i in range(n_records):
                    # Since for some reason indexing doesn't work
                    # So get the time and channel indexes to find the index of the right
                    # record in the array
                    rr[s_r + rec_i]['pulse_length'] = p_length
                    rr[s_r + rec_i]['record_i'] = rec_i
                    if rec_i >= 1:
                        o_r = 0

                    o_p += n_store
                    if rec_i != n_records - 1:
                        # More stuff comming
                        n_store = samples_per_record
                        if rec_i == 0:
                            n_store = samples_per_record - o_r
                    else:
                        n_store = p_length - o_p
                    rr[s_r + rec_i]['length'] = n_store
                    rr[s_r + rec_i]['data'][o_r: o_r + n_store] = p['pulse'][0][o_p: o_p + n_store]


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
        print("Returning %d records" % len(records))
        self.results = []
        return records

    def get_real_noise(self, records):
        """
        Get chunk(s) of noise sample from real noise data
        """
        duration = strax.DEFAULT_RECORD_LENGTH * len(records)

        # Randomly choose where in to start copying
        real_data_sample_size = len(self.config['noise_data'])
        id_t = np.random.randint(0, real_data_sample_size - duration)
        data = self.config['noise_data']
        result = data[id_t:id_t + duration].reshape(len(records),strax.DEFAULT_RECORD_LENGTH)

        return result

@export
@strax.takes_config(
    strax.Option('fax_file', default=None, track=False,
                 help="Directory with fax instructions"),
    strax.Option('nevents',default = 100,track=False,
                help="Number of random events to generate if no instructions are provided"),
    strax.Option('max_pulse_length',default = 20000,
                help='Maximum length of a pulse generated by a PMT in the Pulse class'),
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
    strax.Option('ele_afterpulse_file',default = 'https://github.com/XENONnT/strax_auxiliary_files/blob/master/'
                                                 'fax_files/ele_after_pulse.npy?raw=true'),
    strax.Option('pmt_afterpulse_file',default = '/Users/petergaemers/Desktop/python/WFSimDev/pmt_after_pulse.npy'),
    strax.Option('spe_file',default = 'https://github.com/XENONnT/strax_auxiliary_files/blob/master/'
                                      'fax_files/XENON1T_spe_distributions.csv?raw=true'),
)
class PeaksFromFax(strax.Plugin):
    provides = 'peaks'
    data_kind = 'peaks'
    compressor = 'zstd'
    depends_on = tuple()
    parallel = False
    rechunk_on_save = False

    def infer_dtype(self):
        self.to_pe = get_to_pe(self.run_id,self.config['to_pe_file'])
        return strax.peak_dtype(len(self.to_pe))

    def setup(self):
        spe_dist = init_spe_scaling_factor_distributions(self.config['spe_file'])
        self.config.update(get_resource(self.config['fax_config'],fmt='json'))
        self.config.update({'to_pe':self.to_pe,
                            's1_light_map': InterpolatingMap(get_resource(self.config['s1_light_yield_map'],fmt='json')),
                            's2_light_map': InterpolatingMap(get_resource(self.config['s2_light_yield_map'],fmt='json')),
                            's1_pattern_map': InterpolatingMap(get_resource(self.config['s1_pattern_map'],fmt='json.gz')),
                            's2_pattern_map': InterpolatingMap(get_resource(self.config['s2_pattern_map'],fmt='json.gz')),
                            'uniform_to_pmt_ap': np.load(self.config['pmt_afterpulse_file'],allow_pickle=True).item(),
                            'uniform_to_ele_ap': np.load(self.config['ele_afterpulse_file'],allow_pickle=True).item(),
                            'uniform_to_pe_arr': spe_dist,
                               })

    def iter(self, *args, **kwargs):
        sim = PeakSimulator(self.config)
        instructions = rand_instructions(self.config['nevents'])
        yield sim(instructions)


@export
@strax.takes_config(
    strax.Option('fax_file', default=None, track=False,
                 help="Directory with fax instructions"),
    strax.Option('nevents',default = 100,track=False,
                help="Number of random events to generate if no instructions are provided"),
    strax.Option('max_pulse_length',default = 20000,
                help='Maximum length of a pulse generated by a PMT in the Pulse class'),
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
    strax.Option('ele_afterpulse_file',default = 'https://github.com/XENONnT/strax_auxiliary_files/blob/master/'
                                                 'fax_files/ele_after_pulse.npy?raw=true'),
    strax.Option('pmt_afterpulse_file',default = '/Users/petergaemers/Desktop/python/WFSimDev/pmt_after_pulse.npy'),
    strax.Option('spe_file',default = 'https://github.com/XENONnT/strax_auxiliary_files/blob/master/'
                                      'fax_files/XENON1T_spe_distributions.csv?raw=true'),
    strax.Option('noise_file',default = '/Users/petergaemers/Desktop/python/WFSimDev/real_noise_sample/'
                                        'real_noise_sample/170203_0850_00.npz')
)
class RawRecordsFromFax(strax.Plugin):
    provides = 'raw_records'
    depends_on = tuple()
    dtype = strax.record_dtype()
    # compressor = 'zstd'
    parallel = False
    rechunk_on_save = False

    def setup(self):
        spe_dist = init_spe_scaling_factor_distributions(self.config['spe_file'])
        noise_data = get_resource(self.config['noise_file'], fmt='npy')['arr_0'].flatten()
        self.to_pe = get_to_pe(self.run_id,self.config['to_pe_file'])
        self.config.update(get_resource(self.config['fax_config'],fmt='json'))
        self.config.update({'to_pe':self.to_pe,
                            's1_light_map': InterpolatingMap(get_resource(self.config['s1_light_yield_map'],
                                                                          fmt='json')),
                            's2_light_map': InterpolatingMap(get_resource(self.config['s2_light_yield_map'],
                                                                          fmt='json')),
                            's1_pattern_map': InterpolatingMap(get_resource(self.config['s1_pattern_map'],
                                                                            fmt='json.gz')),
                            's2_pattern_map': InterpolatingMap(get_resource(self.config['s2_pattern_map'],
                                                                            fmt='json.gz')),
                            'uniform_to_pmt_ap': np.load(self.config['pmt_afterpulse_file'],
                                                         allow_pickle=True).item(),
                            'uniform_to_ele_ap': np.load(self.config['ele_afterpulse_file'],
                                                         allow_pickle=True).item(),
                            'uniform_to_pe_arr': spe_dist,
                            'noise_data': noise_data,
                            })

    def iter(self, *args, **kwargs):
        sim = RawRecordsSimulator(self.config)
        instructions = rand_instructions(self.config['nevents'])
        yield from sim(instructions)
