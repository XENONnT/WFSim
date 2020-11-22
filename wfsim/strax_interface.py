import logging

import numpy as np
import pandas as pd
import uproot

import strax
from straxen.common import get_resource
from straxen import get_to_pe
import wfsim
from immutabledict import immutabledict

export, __all__ = strax.exporter()
__all__ += ['instruction_dtype', 'truth_extra_dtype']

instruction_dtype = [
    ('event_number', np.int32),
    ('type', np.int8),
    ('time', np.int64),
    ('x', np.float32),
    ('y', np.float32),
    ('z', np.float32),
    ('amp', np.int32),
    ('recoil', '<U2')]

truth_extra_dtype = [
    ('n_electron', np.float),
    ('n_photon', np.float), ('n_photon_bottom', np.float),
    ('t_first_photon', np.float), ('t_last_photon', np.float), 
    ('t_mean_photon', np.float), ('t_sigma_photon', np.float), 
    ('t_first_electron', np.float), ('t_last_electron', np.float), 
    ('t_mean_electron', np.float), ('t_sigma_electron', np.float), ('endtime',np.int64)]

log = logging.getLogger('SimulationCore')


@export
def rand_instructions(c):
    n = c['nevents'] = c['event_rate'] * c['chunk_size'] * c['nchunk']
    c['total_time'] = c['chunk_size'] * c['nchunk']

    instructions = np.zeros(2 * n, dtype=instruction_dtype)
    uniform_times = c['total_time'] * (np.arange(n) + 0.5) / n
    instructions['time'] = np.repeat(uniform_times, 2) * int(1e9)
    instructions['event_number'] = np.digitize(instructions['time'],
         1e9 * np.arange(c['nchunk']) * c['chunk_size']) - 1
    instructions['type'] = np.tile([1, 2], n)
    instructions['recoil'] = ['er' for i in range(n * 2)]

    r = np.sqrt(np.random.uniform(0, c['tpc_radius']**2, n))
    t = np.random.uniform(-np.pi, np.pi, n)
    instructions['x'] = np.repeat(r * np.cos(t), 2)
    instructions['y'] = np.repeat(r * np.sin(t), 2)
    instructions['z'] = np.repeat(np.random.uniform(-c['tpc_length'], 0, n), 2)

    nphotons = np.random.uniform(2000, 2050, n)
    nelectrons = 10 ** (np.random.uniform(3, 4, n))
    instructions['amp'] = np.vstack([nphotons, nelectrons]).T.flatten().astype(int)

    return instructions


@export
def read_optical(file):
    data = uproot.open(file)
    all_ttrees = dict(data.allitems(filterclass=lambda cls: issubclass(cls, uproot.tree.TTreeMethods)))
    e = all_ttrees[next(iter(all_ttrees))]

    n_events = len(e.array('eventid'))
    # lets separate the events in time by a constant time difference
    time = np.arange(1, n_events+1)

    # Events should be in the TPC
    xp = e.array("xp_pri") / 10
    yp = e.array("yp_pri") / 10
    zp = e.array("zp_pri") / 10

    channels = e.array("pmthitID")
    # TODO: It should be fixed following PMT mapping.
    # In our MC, TPC PMT IDs are assigned at 10000-- -> 0--
    # nVeto ones are assigned at 20000-- -> 2000--
    channels = [[channel if channel < 20000 else channel - 18000 for channel in array] for array in channels]
    timings = e.array("pmthitTime")*1e9

    ins = np.zeros(n_events, dtype=instruction_dtype)

    ins['x'], ins['y'], ins['z'], ins['time'] = xp.flatten()  / 10, \
                                                       yp.flatten() / 10, \
                                                       zp.flatten() / 10, \
                                                       1e7 * time.flatten()

    ins['event_number'] = np.arange(n_events)
    ins['type'] = np.repeat(1, n_events)
    ins['recoil'] = np.repeat('er', n_events)
    ins['amp'] = [len(t) for t in timings]

    # cut interactions without electrons or photons
    ins = ins[ins["amp"] > 0]

    return ins, channels, timings


@export
def instruction_from_csv(filename):
    """Return wfsim instructions from a csv

    :param filename: Path to csv file
    """
    # Pandas does not grok the <U2 field 'recoil' correctly.
    # Probably it loads it as some kind of string instead...
    # we'll get it into the right format in the next step.
    dtype_dict = dict(instruction_dtype)
    df = pd.read_csv(filename,
                     names=list(dtype_dict.keys()),
                     skiprows=1,
                     dtype={k: v for k, v in dtype_dict.items()
                            if k != 'recoil'})

    # Convert to records and check format
    recs = df.to_records(index=False, column_dtypes=dtype_dict)
    expected_dtype = np.dtype(instruction_dtype)
    assert recs.dtype == expected_dtype, \
        f"CSV {filename} produced wrong dtype. Got {recs.dtype}, expected {expected_dtype}."
    return recs


@export
class ChunkRawRecords(object):
    def __init__(self, config):
        self.config = config
        self.rawdata = wfsim.RawData(self.config)
        self.record_buffer = np.zeros(5000000, dtype=strax.record_dtype()) # 2*250 ms buffer
        self.truth_buffer = np.zeros(10000, dtype=instruction_dtype + truth_extra_dtype + [('fill', bool)])

    def __call__(self, instructions):
        # Save the constants as privates
        samples_per_record = strax.DEFAULT_RECORD_LENGTH
        buffer_length = len(self.record_buffer)
        dt = self.config['sample_duration']
        rext = int(self.config['right_raw_extension'])
        cksz = int(self.config['chunk_size'] * 1e9)

        self.blevel = buffer_filled_level = 0
        self.chunk_time_pre = np.min(instructions['time']) - rext
        self.chunk_time = self.chunk_time_pre + cksz # Starting chunk
        self.current_digitized_right = self.last_digitized_right = 0
        for channel, left, right, data in self.rawdata(instructions, self.truth_buffer):
            pulse_length = right - left + 1
            records_needed = int(np.ceil(pulse_length / samples_per_record))

            if self.rawdata.left * self.config['sample_duration'] > self.chunk_time:
                self.chunk_time = self.last_digitized_right * self.config['sample_duration']
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

        yield from self.final_results()

    def final_results(self):
        records = self.record_buffer[:self.blevel] # No copying the records from buffer
        maska = records['time'] <= self.last_digitized_right * self.config['sample_duration']
        records = records[maska]

        records = strax.sort_by_time(records) # Do NOT remove this line
        # strax.baseline(records) Will be done w/ pulse processing
        strax.integrate(records)

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
        if self.config['detector']=='XENON1T':
            yield dict(raw_records=records,
                       truth=_truth)
        else:
            yield dict(raw_records=records[records['channel'] < self.config['channels_top_high_energy'][0]],
                       raw_records_he=records[(records['channel'] >= self.config['channels_top_high_energy'][0]) &
                                              (records['channel'] <= self.config['channels_top_high_energy'][-1])],
                       raw_records_aqmon=records[records['channel']==800],
                       truth=_truth)
        self.record_buffer[:np.sum(~maska)] = self.record_buffer[:self.blevel][~maska]
        self.blevel = np.sum(~maska)

    def source_finished(self):
        return self.rawdata.source_finished


@export
class ChunkRawRecordsOptical(ChunkRawRecords):
    def __init__(self, config):
        self.config = config
        self.rawdata = wfsim.RawDataOptical(self.config)
        self.record_buffer = np.zeros(5000000, dtype=strax.record_dtype()) # 2*250 ms buffer
        self.truth_buffer = np.zeros(10000, dtype=instruction_dtype + truth_extra_dtype + [('fill', bool)])

    def __call__(self, instructions, channels, timings):
        # Save the constants as privates
        samples_per_record = strax.DEFAULT_RECORD_LENGTH
        buffer_length = len(self.record_buffer)
        dt = self.config['sample_duration']
        rext = int(self.config['right_raw_extension'])
        cksz = int(self.config['chunk_size'] * 1e9)

        self.blevel = buffer_filled_level = 0
        self.chunk_time_pre = np.min(instructions['time']) - rext
        self.chunk_time = self.chunk_time_pre + cksz # Starting chunk
        self.current_digitized_right = self.last_digitized_right = 0
        for channel, left, right, data in self.rawdata(instructions, channels, timings, self.truth_buffer):
            pulse_length = right - left + 1
            records_needed = int(np.ceil(pulse_length / samples_per_record))

            if self.rawdata.left * self.config['sample_duration'] > self.chunk_time:
                self.chunk_time = self.last_digitized_right * self.config['sample_duration']
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

        yield from self.final_results()




@strax.takes_config(
    strax.Option('optical',default=False, track=True,
                 help="Flag for using optical mc for instructions"),
    strax.Option('seed',default=False, track=True,
                 help="Option for setting the seed of the random number generator used for"
                      "generation of the instructions"),
    strax.Option('fax_file', default=None, track=False,
                 help="Directory with fax instructions"),
    strax.Option('fax_config_override', default=None,
                 help="Dictionary with configuration option overrides"),
    strax.Option('event_rate', default=5, track=False,
                 help="Average number of events per second"),
    strax.Option('chunk_size', default=5, track=False,
                 help="Duration of each chunk in seconds"),
    strax.Option('nchunk', default=4, track=False,
                 help="Number of chunks to simulate"),
    strax.Option('right_raw_extension', default=50000),
    strax.Option('timeout', default=1800,
                 help="Terminate processing if any one mailbox receives "
                      "no result for more than this many seconds"),
    strax.Option('fax_config',
                 default='https://raw.githubusercontent.com/XENONnT/'
                 'strax_auxiliary_files/master/fax_files/fax_config_nt.json'),
    strax.Option('gain_model',
                 default=('to_pe_per_run', 'https://raw.githubusercontent.com/XENONnT/'
                 'strax_auxiliary_files/master/fax_files/to_pe_nt.npy'),
                 help='PMT gain model. Specify as (model_type, model_config)'),
    strax.Option('detector', default='XENONnT', track=True),)
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

    # A very very long input timeout, our simulator takes time
    input_timeout = 3600 # as an hour

    def setup(self):
        c = self.config
        c.update(get_resource(c['fax_config'], fmt='json'))
        # Update gains to the nT defaults
        self.to_pe = get_to_pe(self.run_id, c['gain_model'],
                              len(c['channels_in_detector']['tpc']))
        c['gains'] = 1 / self.to_pe * (1e-8 * 2.25 / 2**14) / (1.6e-19 * 10 * 50)
        c['gains'][self.to_pe==0] = 0
        if c['seed'] != False:
            np.random.seed(c['seed'])

        overrides = self.config['fax_config_override']
        if overrides is not None:
            c.update(overrides)

        if c['optical']:
            self.instructions, self.channels, self.timings = read_optical(c['fax_file'])
            c['nevents']=len(self.instructions['event_number'])

        elif c['fax_file']:
            assert c['fax_file'][-5:] != '.root', 'None optical g4 input is deprecated use EPIX instead'
            self.instructions = instruction_from_csv(c['fax_file'])
            c['nevents'] = np.max(self.instructions['event_number'])

        else:
            self.instructions = rand_instructions(c)

        # Let below cathode S1 instructions pass but remove S2 instructions
        m = (self.instructions['z'] < -c['tpc_length']) & (self.instructions['type'] == 2)
        self.instructions = self.instructions[~m]

        assert np.all(self.instructions['x']**2 + self.instructions['y']**2 < c['tpc_radius']**2), \
                "Interation is outside the TPC"
        assert np.all(self.instructions['z'] < 0.25), \
                "Interation is outside the TPC"
        assert np.all(self.instructions['amp'] > 0), \
                "Interaction has zero size"

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


@export
class RawRecordsFromFaxNT(FaxSimulatorPlugin):
    provides = ('raw_records', 'raw_records_he', 'raw_records_aqmon', 'truth')
    data_kind = immutabledict(zip(provides, provides))

    def setup(self):
        super().setup()
        self.sim = ChunkRawRecords(self.config)
        self.sim_iter = self.sim(self.instructions)

    def infer_dtype(self):
        dtype = dict(raw_records=strax.record_dtype(),
                     raw_records_he=strax.record_dtype(),
                     raw_records_aqmon=strax.record_dtype(),
                     truth=instruction_dtype + truth_extra_dtype)
        return dtype

    def is_ready(self, chunk_i):
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

    def compute(self, chunk_i):
        try:
            result = next(self.sim_iter)
        except StopIteration:
            raise RuntimeError("Bug in chunk count computation")
        self._sort_check(result['raw_records'])

        return {data_type:self.chunk(
            start=self.sim.chunk_time_pre,
            end=self.sim.chunk_time,
            data=result[data_type],
            data_type=data_type) for data_type in self.provides}

@export
class RawRecordsFromFax1T(RawRecordsFromFaxNT):
    provides = ('raw_records', 'truth')
    data_kind = immutabledict(zip(provides, provides))

    def infer_dtype(self):
        dtype = dict(raw_records=strax.record_dtype(),
                     truth=instruction_dtype + truth_extra_dtype)
        return dtype

@export
class RawRecordsFromFaxOptical(RawRecordsFromFaxNT):
    def setup(self):
        super().setup()
        self.sim = ChunkRawRecordsOptical(self.config)
        self.sim_iter = self.sim(self.instructions, self.channels, self.timings)
