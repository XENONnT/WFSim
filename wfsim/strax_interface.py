import logging

import numpy as np
import pandas as pd
import uproot

import strax, straxen
from .core import RawData, RawDataOptical
from .load_resource import load_config
from immutabledict import immutabledict
from scipy.interpolate import interp1d
from copy import deepcopy
export, __all__ = strax.exporter()
__all__ += ['instruction_dtype', 'optical_extra_dtype', 'truth_extra_dtype']
_cached_wavelength_to_qe_arr = {}


instruction_dtype = [(('Waveform simulator event number.', 'event_number'), np.int32),
             (('Quanta type (S1 photons or S2 electrons)', 'type'), np.int8),
             (('Time of the interaction [ns]', 'time'), np.int64),
             (('X position of the cluster[cm]', 'x'), np.float32),
             (('Y position of the cluster[cm]', 'y'), np.float32),
             (('Z position of the cluster[cm]', 'z'), np.float32),
             (('Number of quanta', 'amp'), np.int32),
             (('Recoil type of interaction.', 'recoil'), np.int8),
             (('Energy deposit of interaction', 'e_dep'), np.float32),
             (('Eventid like in geant4 output rootfile', 'g4id'), np.int32),
             (('Volume id giving the detector subvolume', 'vol_id'), np.int32)
             ]

optical_extra_dtype = [(('Index of first optical input', '_first'), np.int32),
    (('Index of last optical input (including the last)', '_last'), np.int32),]

truth_extra_dtype = [
    (('End time of the interaction [ns]', 'endtime'), np.int64),
    ('n_electron', np.float),
    ('n_photon', np.float), ('n_photon_bottom', np.float),
    ('t_first_photon', np.float), ('t_last_photon', np.float),
    ('t_mean_photon', np.float), ('t_sigma_photon', np.float),
    ('t_first_electron', np.float), ('t_last_electron', np.float),
    ('t_mean_electron', np.float), ('t_sigma_electron', np.float)]


log = logging.getLogger('SimulationCore')


def rand_instructions(c):
    """Random instruction generator function. This will be called by wfsim if you do not specify 
    specific instructions.
    :params c: wfsim configuration dict"""
    n = c['nevents'] = c['event_rate'] * c['chunk_size'] * c['nchunk']
    c['total_time'] = c['chunk_size'] * c['nchunk']

    instructions = np.zeros(2 * n, dtype=instruction_dtype)
    uniform_times = c['total_time'] * (np.arange(n) + 1.) / n
    instructions['time'] = np.repeat(uniform_times, 2) * int(1e9)
    instructions['event_number'] = np.digitize(instructions['time'],
         1e9 * np.arange(c['nchunk']) * c['chunk_size']) - 1
    instructions['type'] = np.tile([1, 2], n)
    instructions['recoil'] = [7 for i in range(n * 2)]  # Use nest ids for  ER

    r = np.sqrt(np.random.uniform(0, c['tpc_radius']**2, n))
    t = np.random.uniform(-np.pi, np.pi, n)
    instructions['x'] = np.repeat(r * np.cos(t), 2)
    instructions['y'] = np.repeat(r * np.sin(t), 2)
    instructions['z'] = np.repeat(np.random.uniform(-c['tpc_length'], 0, n), 2)

    nphotons = np.random.uniform(2000, 2050, n)
    nelectrons = 10 ** (np.random.uniform(3, 4, n))
    instructions['amp'] = np.vstack([nphotons, nelectrons]).T.flatten().astype(int)

    return instructions


def _read_optical_nveto(config, events, mask):
    """Helper function for nveto to read photon channels and timings from G4 and apply QE's
    :params config: dict, wfsim configuration
    :params events: g4 root file
    :params mask: 1d bool array to select events
    
    returns two flatterned nested arrays of channels and timings, 
    """
    channels = np.hstack(events["pmthitID"].array(library="np")[mask])
    timings = np.hstack(events["pmthitTime"].array(library="np")[mask] * 1e9)

    constant_hc = 1239.841984 # (eV*nm) to calculate (wavelength lambda) = h * c / energy
    wavelengths = np.hstack(constant_hc / events["pmthitEnergy"].array(library="np")[mask])

    # Caching a 2d array of interpolated value of (channel, wavelength [every nm]) 
    h = strax.deterministic_hash(config)
    nveto_channels = np.arange(config['channel_map']['nveto'][0], config['channel_map']['nveto'][1] + 1)
    if h not in _cached_wavelength_to_qe_arr:
        resource = load_config(config)
        if getattr(resource, 'nv_pmt_qe_data', None) is None:
            log.warning('nv pmt qe data not specified all qe default to 100 %')
            _cached_wavelength_to_qe_arr[h] = np.ones([len(nveto_channels), 1000]) * 100
        else:
            qe_data = resource.nv_pmt_qe_data
            wavelength_to_qe_arr = np.zeros([len(nveto_channels), 1000])
            for ich, channel in enumerate(nveto_channels):
                wavelength_to_qe_arr[ich] = interp1d(qe_data['nv_pmt_qe_wavelength'],
                   qe_data['nv_pmt_qe'][str(channel)],
                   bounds_error=False,
                   kind='linear',
                   fill_value=(0, 0))(np.arange(1000))
            _cached_wavelength_to_qe_arr[h] = wavelength_to_qe_arr

    # retrieving qe by index
    hit_mask = (channels >= nveto_channels[0]) & (channels <= nveto_channels[-1])
    channels[~hit_mask] = nveto_channels[0]

    qes = _cached_wavelength_to_qe_arr[h][channels - nveto_channels[0],
        np.around(wavelengths).astype(np.int64)]
    hit_mask &= np.random.rand(len(qes)) <= qes * config.get('nv_pmt_ce_factor', 1.0) / 100

    amplitudes, og_offset = [], 0
    for tmp in events["pmthitID"].array(library="np"):
        og_length = len(tmp)
        amplitudes.append(hit_mask[og_offset:og_offset + og_length].sum())
        og_offset += og_length
    
    return channels[hit_mask], timings[hit_mask], np.array(amplitudes, int)


def read_optical(config):
    """Function will be executed when wfsim in run in optical mode. This function expects c['fax_file'] 
    to be a root file from optical mc
    :params config: wfsim configuration dict"""
    data = uproot.open(config['fax_file'])
    try:
        events = data.get('events')
    except:
        raise Exception("Are you using mc version >4?")

    # Slightly weird here. Entry_stop is not in the regular config, so if it's not skip this part
    g4id = events['eventid'].array(library="np")
    if config.get('entry_stop', -1) == -1:
        config['entry_stop'] = np.max(g4id) + 1

    mask = ((g4id < config.get('entry_stop', int(2**63-1))) 
        & (g4id >= config.get('entry_start', 0)))
    n_events = len(g4id[mask])

    if config['neutron_veto']:
        channels, timings, amplitudes = _read_optical_nveto(config, events, mask)
        # Still need to shift nveto channel for simulation code to work
        channels -= config['channel_map']['nveto'][0]
    else:
        # TPC
        channels = np.hstack(events["pmthitID"].array(library="np")[mask])
        timings = np.hstack(events["pmthitTime"].array(library="np")[mask]) * 1e9
        amplitudes = np.array([len(tmp) for tmp in events["pmthitID"].array(library="np")[mask]])

    # Events should be in the TPC
    ins = np.zeros(n_events, dtype=instruction_dtype + optical_extra_dtype)
    ins['x'] = events["xp_pri"].array(library="np").flatten()[mask] / 10.
    ins['y'] = events["yp_pri"].array(library="np").flatten()[mask] / 10.
    ins['z'] = events["zp_pri"].array(library="np").flatten()[mask] / 10.
    ins['time']= int(1e7) * np.arange(1, n_events + 1).flatten()[mask]  # Separate the events by a constant dt
    ins['event_number'] = np.arange(n_events)
    ins['g4id'] = events['eventid'].array(library="np")[mask]
    ins['type'] = np.repeat(1, n_events)
    ins['recoil'] = np.repeat(1, n_events)
    ins['_first'] = np.cumsum(amplitudes) - amplitudes
    ins['_last'] = np.cumsum(amplitudes)

    return ins, channels, timings


def instruction_from_csv(filename):
    """
    Return wfsim instructions from a csv
    :param filename: Path to csv file
    """
    df = pd.read_csv(filename)

    recs = np.zeros(len(df), dtype=instruction_dtype)
    for column in df.columns:
        recs[column]=df[column]

    expected_dtype = np.dtype(instruction_dtype)
    assert recs.dtype == expected_dtype, \
        f"CSV {filename} produced wrong dtype. Got {recs.dtype}, expected {expected_dtype}."
    return recs


@export
class ChunkRawRecords(object):
    def __init__(self, config, rawdata_generator=RawData, **kwargs):
        log.debug(f'Starting {self.__class__.__name__}')
        self.config = config
        log.debug(f'Setting raw data with {rawdata_generator.__class__.__name__}')
        self.rawdata = rawdata_generator(self.config, **kwargs)
        self.record_buffer = np.zeros(5000000,
            dtype=strax.raw_record_dtype(samples_per_record=strax.DEFAULT_RECORD_LENGTH))  # 2*250 ms buffer
        self.truth_buffer = np.zeros(10000, dtype=instruction_dtype + truth_extra_dtype + [('fill', bool)])

        self.blevel = buffer_filled_level = 0
        log.debug(f'Starting {self.__class__.__name__} initiated')

    def __call__(self, instructions, **kwargs):
        log.debug(f'{self.__class__.__name__} called')
        samples_per_record = strax.DEFAULT_RECORD_LENGTH
        dt = self.config['sample_duration']
        buffer_length = len(self.record_buffer)
        rext = int(self.config['right_raw_extension'])
        cksz = int(self.config['chunk_size'] * 1e9)

        # Save the constants as privates
        self.blevel = buffer_filled_level = 0
        self.chunk_time_pre = np.min(instructions['time']) - rext
        self.chunk_time = self.chunk_time_pre + cksz # Starting chunk
        self.current_digitized_right = self.last_digitized_right = 0
        for channel, left, right, data in self.rawdata(instructions=instructions,
                                                       truth_buffer=self.truth_buffer,
                                                       **kwargs):
            pulse_length = right - left + 1
            records_needed = int(np.ceil(pulse_length / samples_per_record))

            if self.rawdata.left * dt > self.chunk_time:
                self.chunk_time = (self.last_digitized_right + 1) * dt
                yield from self.final_results()
                self.chunk_time_pre = self.chunk_time
                self.chunk_time += cksz

            if self.blevel + records_needed > buffer_length:
                log.warning('Chunck size too large, insufficient record buffer')
                yield from self.final_results()

            if self.blevel + records_needed > buffer_length:
                log.warning('Pulse length too large, insufficient record buffer, skipping pulse')
                continue

            # WARNING baseline and area fields are zeros before finish_results
            s = slice(self.blevel, self.blevel + records_needed)
            self.record_buffer[s]['channel'] = channel
            self.record_buffer[s]['dt'] = dt
            self.record_buffer[s]['time'] = dt * (left + samples_per_record * np.arange(records_needed))
            self.record_buffer[s]['length'] = [min(pulse_length, samples_per_record * (i+1))
                - samples_per_record * i for i in range(records_needed)]
            self.record_buffer[s]['pulse_length'] = pulse_length
            self.record_buffer[s]['record_i'] = np.arange(records_needed)
            self.record_buffer[s]['data'] = np.pad(data,
                (0, records_needed * samples_per_record - pulse_length), 'constant').reshape((-1, samples_per_record))
            self.blevel += records_needed
            if self.rawdata.right != self.current_digitized_right:
                self.last_digitized_right = self.current_digitized_right
                self.current_digitized_right = self.rawdata.right

        self.last_digitized_right = self.current_digitized_right
        self.chunk_time = max((self.last_digitized_right + 1) * dt, self.chunk_time_pre + dt)
        yield from self.final_results()

    def final_results(self):
        records = self.record_buffer[:self.blevel] # No copying the records from buffer
        maska = records['time'] <= self.last_digitized_right * self.config['sample_duration']
        records = records[maska]

        records = strax.sort_by_time(records) # Do NOT remove this line

        # Yield an appropriate amount of stuff from the truth buffer
        # and mark it as available for writing again

        maskb = (
            self.truth_buffer['fill'] &
            # This condition will always be false if self.truth_buffer['t_first_photon'] == np.nan
            ((self.truth_buffer['t_first_photon']
             <= self.last_digitized_right * self.config['sample_duration']) |
             # Hence, we need to use this trick to also save these cases (this
             # is what we set the end time to for np.nans)
            (np.isnan(self.truth_buffer['t_first_photon']) &
             (self.truth_buffer['time']
              <= self.last_digitized_right * self.config['sample_duration'])
            )))
        truth = self.truth_buffer[maskb]   # This is a copy, not a view!

        # Careful here: [maskb]['fill'] = ... does not work
        # numpy creates a copy of the array on the first index.
        # The assignment then goes to the (unused) copy.
        # ['fill'][maskb] leads to a view first, then the advanced
        # assignment works into the original array as expected.
        self.truth_buffer['fill'][maskb] = False

        truth.sort(order='time')
        # Return truth without 'fill' field
        _truth = np.zeros(len(truth), dtype=instruction_dtype + truth_extra_dtype)
        for name in _truth.dtype.names:
            _truth[name] = truth[name]
        _truth['time'][~np.isnan(_truth['t_first_photon'])] = \
            _truth['t_first_photon'][~np.isnan(_truth['t_first_photon'])].astype(int)
        _truth.sort(order='time')

        #Oke this will be a bit ugly but it's easy
        if self.config['detector']=='XENON1T':
            yield dict(raw_records=records,
                       truth=_truth)
        elif self.config['detector']=='XENONnT':
            yield dict(raw_records=records[records['channel'] < self.config['channel_map']['he'][0]],
                       raw_records_he=records[(records['channel'] >= self.config['channel_map']['he'][0]) &
                                              (records['channel'] <= self.config['channel_map']['he'][-1])],
                       raw_records_aqmon=records[records['channel'] == 800],
                       truth=_truth)

        self.record_buffer[:np.sum(~maska)] = self.record_buffer[:self.blevel][~maska]
        self.blevel = np.sum(~maska)

    def source_finished(self):
        return self.rawdata.source_finished


@strax.takes_config(
    strax.Option('optical', default=False, track=True,
                 help="Flag for using optical mc for instructions"),
    strax.Option('seed', default=False, track=True,
                 help="Option for setting the seed of the random number generator used for"
                      "generation of the instructions"),
    strax.Option('fax_file', default=None, track=False,
                 help="Directory with fax instructions"),
    strax.Option('fax_config_override', default=None,
                 help="Dictionary with configuration option overrides"),
    strax.Option('event_rate', default=1000, track=False,
                 help="Average number of events per second"),
    strax.Option('chunk_size', default=100, track=False,
                 help="Duration of each chunk in seconds"),
    strax.Option('nchunk', default=10, track=False,
                 help="Number of chunks to simulate"),
    strax.Option('right_raw_extension', default=50000),
    strax.Option('timeout', default=1800,
                 help="Terminate processing if any one mailbox receives "
                      "no result for more than this many seconds"),
    strax.Option('fax_config', default='fax_config_nt_design.json'),
    strax.Option('gain_model', default=('to_pe_per_run', 'to_pe_nt.npy'),
                 help='PMT gain model. Specify as (model_type, model_config).'),
    strax.Option('detector', default='XENONnT', track=True),
    strax.Option('channel_map', track=False, type=immutabledict,
                 help="immutabledict mapping subdetector to (min, max) "
                      "channel number. Provided by context"),
    strax.Option('n_tpc_pmts', track=False,
                 help="Number of pmts in tpc. Provided by context"),
    strax.Option('n_top_pmts', track=False,
                 help="Number of pmts in top array. Provided by context"),
    strax.Option('neutron_veto', default=False, track=False,
                 help="Flag for nVeto optical simulation instead of TPC"),
)
class SimulatorPlugin(strax.Plugin):
    depends_on = tuple()

    # Cannot arbitrarily rechunk records inside events
    rechunk_on_save = False

    # Simulator uses iteration semantics, so the plugin has a state
    # this seems avoidable...
    parallel = False

    # this state is needed for sorting checks,
    # but it prevents prevent parallelization
    last_chunk_time = -999999999999999

    # A very very long input timeout, our simulator takes time
    input_timeout = 3600 # as an hour

    def setup(self):
        self.set_config()

        c=self.config
        
        # Update gains to the nT defaults
        self.to_pe = straxen.get_to_pe(self.run_id, c['gain_model'],
                              c['channel_map']['tpc'][1]+1)

        c['gains'] = np.divide((1e-8 * 2.25 / 2**14) / (1.6e-19 * 10 * 50),
                               self.to_pe,
                               out=np.zeros_like(self.to_pe, ), 
                               where=self.to_pe!=0)

        if c['seed'] != False:
            np.random.seed(c['seed'])

        # We hash the config to load resources. Channel map is immutable and cannot be hashed
        self.config['channel_map'] = dict(self.config['channel_map'])
        self.config['channel_map']['sum_signal'] = 800
        self.config['channels_bottom'] = np.arange(self.config['n_top_pmts'],self.config['n_tpc_pmts'])

        self.get_instructions()
        self.check_instructions()
        self._setup()

    def set_config(self,):
        c = self.config
        c.update(straxen.get_resource(c['fax_config'], fmt='json'))
        overrides = self.config['fax_config_override']
        if overrides is not None:
            c.update(overrides)

    def _setup(self):
        # Set in inheriting class
        pass

    def get_instructions(self):
        # Set in inheriting class
        pass

    def check_instructions(self):
        # Set in inheriting class
        pass

    def _sort_check(self, result):
        if len(result) == 0: return
        if result['time'][0] < self.last_chunk_time + 1000:
            raise RuntimeError(
                "Simulator returned chunks with insufficient spacing. "
                f"Last chunk's max time was {self.last_chunk_time}, "
                f"this chunk's first time is {result['time'][0]}.")
        if np.diff(result['time']).min() < 0:
            raise RuntimeError("Simulator returned non-sorted records!")
        self.last_chunk_time = result['time'].max()

    def is_ready(self,chunk_i):
        """Overwritten to mimic online input plugin.
        Returns False to check source finished;
        Returns True to get next chunk.
        """
        if 'ready' not in self.__dict__: self.ready = False
        self.ready ^= True # Flip
        return self.ready

    def source_finished(self):
        """Return whether all instructions has been used."""
        return self.sim.source_finished()


@export
class RawRecordsFromFaxNT(SimulatorPlugin):
    provides = ('raw_records', 'raw_records_he', 'raw_records_aqmon', 'truth')
    data_kind = immutabledict(zip(provides, provides))

    def _setup(self):
        self.sim = ChunkRawRecords(self.config)
        self.sim_iter = self.sim(self.instructions)

    def get_instructions(self):
        if self.config['fax_file']:
            assert not self.config['fax_file'].endswith('root'), 'None optical g4 input is deprecated use EPIX instead'
            assert self.config['fax_file'].endswith('csv'), 'Only csv input is supported'
            self.instructions = instruction_from_csv(self.config['fax_file'])
        else:
            self.instructions = rand_instructions(self.config)

    def check_instructions(self):
        # Let below cathode S1 instructions pass but remove S2 instructions
        m = (self.instructions['z'] < - self.config['tpc_length']) & (self.instructions['type'] == 2)
        self.instructions = self.instructions[~m]

        assert np.all(self.instructions['x']**2 + self.instructions['y']**2 < self.config['tpc_radius']**2), \
                "Interation is outside the TPC"
        assert np.all(self.instructions['z'] < 0.25), \
                "Interation is outside the TPC"
        assert np.all(self.instructions['amp'] > 0), \
                "Interaction has zero size"

    def infer_dtype(self):
        dtype = {data_type:strax.raw_record_dtype(samples_per_record=strax.DEFAULT_RECORD_LENGTH)
                for data_type in self.provides if data_type != 'truth'}
        dtype['truth'] = instruction_dtype + truth_extra_dtype
        return dtype

    def compute(self):
        try:
            result = next(self.sim_iter)
        except StopIteration:
            raise RuntimeError("Bug in chunk count computation")
        self._sort_check(result[self.provides[0]])
        # To accomodate nveto raw records, should be the first in provide.

        return {data_type:self.chunk(
            start=self.sim.chunk_time_pre,
            end=self.sim.chunk_time,
            data=result[data_type],
            data_type=data_type) for data_type in self.provides}


@export
class RawRecordsFromFax1T(RawRecordsFromFaxNT):
    provides = ('raw_records', 'truth')


@export
@strax.takes_config(
    strax.Option('epix_config', track=False, default={},
                help='Path to epix configuration'),
    strax.Option('entry_start', default=0, track=False,),
    strax.Option('entry_stop', default=-1, track=False,
                help='G4 id event number to stop at. If -1 process the entire file'),
    )
class RawRecordsFromMcChain(SimulatorPlugin):
    provides = ('raw_records', 'raw_records_he', 'raw_records_aqmon', 'raw_records_nv', 'truth', 'truth_nv')
    data_kind = immutabledict(zip(provides, provides))

    def set_timing(self,):
        """Set timing information in such a way to synchronize instructions for the TPC and nVeto"""
        logging.info("Setting timings")

        # If no event_stop is specified set it to the maximum (-1=do all events)
        if self.config['entry_stop'] == -1:
            self.config['entry_start'] = np.min(self.g4id)
            self.config['entry_stop'] = np.max(self.g4id) + 1

        # Convert rate from Hz to ns^-1
        rate = self.config['event_rate'] / 1000000000
        # Add half interval to avoid time 0
        timings = np.random.uniform((self.config['entry_start'] + 0.5) / rate, 
                                    (self.config['entry_stop'] + 0.5) / rate, 
                                    self.config['entry_stop'] - self.config['entry_start'])
        timings = np.sort(timings).astype(np.int64)

        # For tpc we have multiple instructions per g4id.
        if 'raw_records' in self.provides:
            i_timings = np.searchsorted(np.unique(self.g4id), self.instructions_epix['g4id'])
            print(len(timings), len(i_timings))
            print(timings[i_timings])
            self.instructions_epix['time'] += timings[i_timings]
        # nveto instruction doesn't carry physical time delay, so the time is overwritten
        if 'raw_records_nv' in self.provides:
            i_timings = np.searchsorted(np.unique(self.g4id), self.instructions_nveto['g4id'])
            self.instructions_nveto['time'] = timings[i_timings]

    def check_instructions(self):
        if 'raw_records' in self.provides:
            # Let below cathode S1 instructions pass but remove S2 instructions
            m = (self.instructions_epix['z'] < - self.config['tpc_length']) & (self.instructions_epix['type'] == 2)
            self.instructions_epix = self.instructions_epix[~m]

            assert np.all(self.instructions_epix['x']**2 + self.instructions_epix['y']**2 < self.config['tpc_radius']**2), \
                    "Interation is outside the TPC"
            assert np.all(self.instructions_epix['z'] < 0.25), \
                    "Interation is outside the TPC"
            assert np.all(self.instructions_epix['amp'] > 0), \
                    "Interaction has zero size"
            assert all(self.instructions_epix['g4id'] >= self.config['entry_start'])
            assert all(self.instructions_epix['g4id'] < self.config['entry_stop'])

        if 'raw_records_nv' in self.provides:
            assert all(self.instructions_nveto['g4id'] >= self.config['entry_start'])
            assert all(self.instructions_nveto['g4id'] < self.config['entry_stop'])

    def get_instructions(self):
        """
        Run epix and save instructions as self.instructions_epix
        epix_config with all the epix run arguments is passed as a dictionary, where
        source_rate must be set to 0 (epix default), since time handling is done outside epix.
        
        epix needs to be imported in here to avoid circle imports
        """
        logging.info("Getting instructions from epix")

        self.g4id = []
        if 'raw_records' in self.provides:
            import epix
            epix_config = deepcopy(self.config['epix_config'])  # dictionary directly fed to context
            epix_config.update({
                'detector': self.config['detector'],
                'entry_start': self.config['entry_start'],
                'entry_stop': self.config['entry_stop'],
                'input_file': self.config['fax_file'],})
            self.instructions_epix = epix.run_epix.main(
                epix.run_epix.setup(epix_config),
                return_wfsim_instructions=True)
            self.g4id.append(self.instructions_epix['g4id'])

        if 'raw_records_nv' in self.provides:
            self.config['nv_pmt_ce_factor'] = 1.0
            self.instructions_nveto, self.nveto_channels, self.nveto_timings =\
                read_optical(self.config)
            self.g4id.append(self.instructions_nveto['g4id'])

        self.g4id = np.unique(np.concatenate(self.g4id))
        self.set_timing()

    def _setup(self):
        if 'raw_records' in self.provides:
            self.sim = ChunkRawRecords(self.config)
            self.sim_iter = self.sim(self.instructions_epix)
        if 'raw_records_nv' in self.provides:
            self.sim_nv = ChunkRawRecords(self.config,
                                          rawdata_generator=RawDataOptical,
                                          channels=self.nveto_channels,
                                          timings=self.nveto_timings,)
            self.sim_nv.truth_buffer = np.zeros(10000, dtype=instruction_dtype + optical_extra_dtype
                                                + truth_extra_dtype + [('fill', bool)])

            assert '_first' in self.instructions_nveto.dtype.names, 'Require indexing info in optical instruction see optical extra dtype'
            assert all(self.instructions_nveto['type'] == 1), 'Only s1 type is supported for generating rawdata from optical input'
            self.sim_nv_iter = self.sim_nv(self.instructions_nveto,)

    def infer_dtype(self):
        dtype = dict([(data_type, instruction_dtype + truth_extra_dtype) if 'truth' in data_type
            else (data_type, strax.raw_record_dtype(samples_per_record=strax.DEFAULT_RECORD_LENGTH))
            for data_type in self.provides])
        return dtype

    def compute(self):
        if 'raw_records' in self.provides:
            try:
                result = next(self.sim_iter)
                self._sort_check(result['raw_records'])
            except StopIteration:
                raise RuntimeError("Bug in chunk count computation")

        if 'raw_records_nv' in self.provides:
            try:
                result_nv = next(self.sim_nv_iter)
                self._sort_check(result_nv['raw_records'])
                result_nv['raw_records']['channel'] += config['channel_map']['nveto'][0]
            except StopIteration:
                raise RuntimeError("Bug in chunk count computation")            

        out = {}
        for data_type in self.provides:
            if 'nv' in data_type:
                out[data_type] = self.chunk(
                    start=self.sim_nv.chunk_time_pre,
                    end=self.sim_nv.chunk_time,
                    data=result_nv[data_type.strip('_nv')],
                    data_type=data_type)
            else:
                out[data_type] = self.chunk(
                    start=self.sim.chunk_time_pre,
                    end=self.sim.chunk_time,
                    data=result[data_type],
                    data_type=data_type)

        return out

    def source_finished(self):
        """Return whether all instructions has been used."""
        source_finished = True
        if 'raw_records' in self.provides:
            source_finished &= self.sim.source_finished()
        if 'raw_records_nv' in self.provides:
            source_finished &= self.sim_nv.source_finished()
        return source_finished
