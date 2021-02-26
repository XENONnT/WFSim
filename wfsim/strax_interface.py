import logging

import numpy as np
import pandas as pd
import uproot

import strax
from straxen.common import get_resource
from straxen import get_to_pe
import wfsim
from immutabledict import immutabledict
import os.path as osp
import json
from collections import Counter

export, __all__ = strax.exporter()
__all__ += ['instruction_dtype', 'truth_extra_dtype']


#recoil refers to 1:ER, 2=NR, 3=Alpha
instruction_dtype = [(('Waveform simulator event number.', 'event_number'), np.int32),
             (('Quanta type (S1 photons or S2 electrons)', 'type'), np.int8),
             (('Time of the interaction [ns]', 'time'), np.int64),
             (('End time of the interaction [ns]', 'endtime'), np.int64),
             (('X position of the cluster[cm]', 'x'), np.float32),
             (('Y position of the cluster[cm]', 'y'), np.float32),
             (('Z position of the cluster[cm]', 'z'), np.float32),
             (('Number of quanta', 'amp'), np.int32),
             (('Recoil type of interaction.', 'recoil'), np.int8),
             (('Energy deposit of interaction', 'e_dep'), np.float32),
             (('Eventid like in geant4 output rootfile', 'g4id'), np.int32),
             (('Volume id giving the detector subvolume', 'vol_id'), np.int32)
             ]

truth_extra_dtype = [
    ('n_electron', np.float),
    ('n_photon', np.float), ('n_photon_bottom', np.float),
    ('t_first_photon', np.float), ('t_last_photon', np.float), 
    ('t_mean_photon', np.float), ('t_sigma_photon', np.float), 
    ('t_first_electron', np.float), ('t_last_electron', np.float), 
    ('t_mean_electron', np.float), ('t_sigma_electron', np.float)]

log = logging.getLogger('SimulationCore')

def rand_instructions(c):
    n = c['nevents'] = c['event_rate'] * c['chunk_size'] * c['nchunk']
    c['total_time'] = c['chunk_size'] * c['nchunk']

    instructions = np.zeros(2 * n, dtype=instruction_dtype)
    uniform_times = c['total_time'] * (np.arange(n) + 0.5) / n
    instructions['time'] = np.repeat(uniform_times, 2) * int(1e9)
    instructions['event_number'] = np.digitize(instructions['time'],
         1e9 * np.arange(c['nchunk']) * c['chunk_size']) - 1
    instructions['type'] = np.tile([1, 2], n)
    instructions['recoil'] = [7 for i in range(n * 2)] #Use nest ids for  ER

    r = np.sqrt(np.random.uniform(0, c['tpc_radius']**2, n))
    t = np.random.uniform(-np.pi, np.pi, n)
    instructions['x'] = np.repeat(r * np.cos(t), 2)
    instructions['y'] = np.repeat(r * np.sin(t), 2)
    instructions['z'] = np.repeat(np.random.uniform(-c['tpc_length'], 0, n), 2)

    nphotons = np.random.uniform(2000, 2050, n)
    nelectrons = 10 ** (np.random.uniform(3, 4, n))
    instructions['amp'] = np.vstack([nphotons, nelectrons]).T.flatten().astype(int)

    return instructions

def read_optical(c):
    file = c['fax_file']
    data = uproot.open(file)
    try:
        e = data.get('events/events')
    except:
        raise Exception("Are you using mc version >4?")

    event_id = e['eventid'].array(library="np")
    n_events = len(event_id)
    # lets separate the events in time by a constant time difference
    time = np.arange(1, n_events+1)

    if c['neutron_veto']:
        nV_pmt_id_offset = 20000
        channels = [[channel - nV_pmt_id_offset for channel in array if channel >=20000] for array in e["pmthitID"].array(library="np")]
        timings = e["pmthitTime"].array(library="np")*1e9
    else:
        # TPC
        channels = e["pmthitID"].array(library="np")
        timings = e["pmthitTime"].array(library="np")*1e9

    # Events should be in the TPC
    ins = np.zeros(n_events, dtype=instruction_dtype)
    ins['x'] = e["xp_pri"].array(library="np").flatten() / 10.
    ins['y'] = e["yp_pri"].array(library="np").flatten() / 10.
    ins['z'] = e["zp_pri"].array(library="np").flatten() / 10.
    ins['time']= 1e7 * time.flatten()
    ins['event_number'] = np.arange(n_events)
    ins['g4id'] = event_id
    ins['type'] = np.repeat(1, n_events)
    ins['recoil'] = np.repeat(1, n_events)
    ins['amp'] = [len(t) for t in timings]

    # cut interactions without electrons or photons
    ins = ins[ins["amp"] > 0]
    if (c['event_start']&c['event_stop']):
        mask = np.where((ins['g4id']<c['event_stop'])&(ins['g4id']>=c['event_start']))
        return ins[mask], channels[mask],timings[mask]

    return ins, channels, timings

def instruction_from_csv(filename):
    """
    Return wfsim instructions from a csv
    
    :param filename: Path to csv file
    """
    df = pd.read_csv(filename)
    
    recs = np.zeros(len(df),
                    dtype=instruction_dtype
                   )
    for column in df.columns:
        recs[column]=df[column]
        
    expected_dtype = np.dtype(instruction_dtype)
    assert recs.dtype == expected_dtype, \
        f"CSV {filename} produced wrong dtype. Got {recs.dtype}, expected {expected_dtype}."
    return recs


@export
class McChainSimulator(object):
    def __init__(self,strax_context,run_id) -> None:
        ''' Epix needs to be imported in here to avoid circle imports'''
        self.context = strax_context
        self.file = self.context.config['fax_file']
        self.targets = ('raw_records')
        self.run_id = run_id
        if "RawRecordsFromFaxNVeto" in self.context._plugin_class_registry:
            self.targets = ('raw_records','raw_records_nv')

    def instructions_from_epix(self,):
        '''Run epix and save instructions as self.instruction.'''
        logging.info("Getting instructions from epix")
        import epix
        epix_config=dict(
            input_file='/Volumes/Extreme SSD/XENONnT_TPC_neutron_50000-001.root',
            detector='XENONnT',
            epix_config_override='',
            entry_stop=self.context.config['event_stop'],
            micro_separation=0.05,
            micro_separation_time=10,
            tag_cluster_by='time',
            max_delay=1e7, #ns
            source_rate=-1,
            output_path="",
            debug=False,)
        epix_config = epix.run_epix.setup(epix_config)
        self.instructions_epix=epix.run_epix.main(epix_config,return_wfsim_instructions=True)

    def instructions_from_nveto(self,):
        logging.info("Getting nveto instructions")
        self.instructions_nveto=read_optical(self.context.config)

    def set_timing(self,):
        logging.info("Setting timings")
        rate = self.context.config['event_rate']
        n_events = self.instructions_epix['g4id'][-1]-self.instructions_epix['g4id'][0]+1

        timings= self.context.config['event_start']/rate + wfsim.units.s*np.random.random(n_events)*(n_events)/rate
        timings.sort()

        #For tpc we have multiple instructions per g4id. For nveto we only have one
        counter=Counter(self.instructions_epix['g4id'])
        _min=list(counter.keys())[0]
        _max=list(counter.keys())[-1]+1 #We need to include the last vlaue
        i_per_id = [counter[k] for k in range(_min,_max)]
        timings_tpc=np.repeat(timings,i_per_id)
        self.instructions_epix['time']=timings_tpc
        if self.context.config['neutron_veto']:
            self.instructions_nveto['time']=timings
    
    def set_configuration(self,):
        chunk_size = np.ceil(500/self.context.config['event_rate'])
        nchunk = np.ceil((self.context.config['event_stop']-self.context.config['event_start']+1)/ (chunk_size*self.context.config['event_rate']))
        self.context.set_config(dict(chunk_size=chunk_size, nchunk=nchunk,))
        self.context.set_config(dict(wfsim_instructions=self.instructions_epix,))
        if self.context.config['neutron_veto']:
            self.context.set_config(dict(wfsim_nveto_instruction=self.instructions_nveto,
                                         wfsim_nveto_channels=self.nveto_channels,
                                         wfsim_nveto_timings=self.nveto_timings))

    def set_instructions(self,):
        self.instructions_from_epix()
        if self.context.config['neutron_veto']:
            self.instructions_from_nveto()
        self.set_timing()
        self.set_configuration()

    def run_strax(self,run_id):
        '''Runs wfsim up to raw records for tpc and if requisted the nveto'''
        self.context.make(str(run_id),'raw_records')

        #nveto doesn't have proper afterpulses or noise implementation
        if self.context.config['neutron_veto']:
            self.context.set_config(dict(fax_config_override=dict(enable_pmt_afterpulses=False,
                                                              enable_electron_afterpulses=False,
                                                              enable_noise=False,)))
            self.context.make(run_id,'raw_records_nv')
        
    def _sync_meta(self,):
        '''Sync metadata between tpc and nveto. The start and end times of the metadata must match for
            something strax wants. Check which ever one starts first, make that the start of both. Same for end'''
        logging.debug('Syncing metadata')
        keys = ('start','stop')
        for key in keys:
            if self.tpc_meta[key]>self.nveto_meta[key]:
                self.tpc_meta[key]=self.nveto_meta[key]
            else:
                self.nveto_meta[key]=self.tpc_meta[key]

    def _store_meta(self,):
        ''''Store metadata after matching'''
        logging.debug('Storing synced metadata')
        json_options = dict(sort_keys=True, indent=4)
        metas=(self.tpc_meta,self.nveto_meta)

        for target,ix in enumerate(self.targets):
            prefix=f"{self.context.key_for(self.run_id,target)}-metadata.json"
            path = self.context.storage[0].path
            md_path = osp.join(path,prefix)
            with open(md_path, mode='w') as f:
                f.write(json.dumps(metas[ix], **self.json_options))
    
    def sync_meta(self,run_id):
        '''Sync metadata outputs from the tpc and the nveto.'''
        self.tpc_meta=self.context.get_meta(run_id,'raw_records')
        self.nveto_meta=self.context.get_meta(run_id,'raw_records_nv')
        self._sync_meta()
        self._store_meta()

    def run_chain(self):
        logging.info(f"Setting instructions")
        self.set_instructions()
        logging.info(f"Simulating data")
        self.run_strax(run_id=self.run_id)
        if self.context.config['neutron_veto']:
            logging.info(f"Syncing metadata")
            self.sync_meta()


@export
class ChunkRawRecords(object):
    def __init__(self, config):
        self.config = config
        self.rawdata = wfsim.RawData(self.config)
        self.record_buffer = np.zeros(5000000,
                                      dtype=strax.raw_record_dtype(samples_per_record=strax.DEFAULT_RECORD_LENGTH)) # 2*250 ms buffer
        self.truth_buffer = np.zeros(10000, dtype=instruction_dtype + truth_extra_dtype + [('fill', bool)])

        self.blevel = buffer_filled_level = 0

    def __call__(self, instructions, **kwargs):
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
                self.chunk_time = self.last_digitized_right * dt
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
        if self.config['neutron_veto']:
            yield dict(raw_records_nv=records[records['channel'] < self.config['channel_map']['he'][0]],
                       truth=_truth)
        elif self.config['detector']=='XENONnT':
            yield dict(raw_records=records[records['channel'] < self.config['channel_map']['he'][0]],
                       raw_records_he=records[(records['channel'] >= self.config['channel_map']['he'][0]) &
                                              (records['channel'] <= self.config['channel_map']['he'][-1])],
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
        self.record_buffer = np.zeros(5000000, dtype=strax.raw_record_dtype()) # 2*250 ms buffer
        self.truth_buffer = np.zeros(10000, dtype=instruction_dtype + truth_extra_dtype + [('fill', bool)])


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
    strax.Option('chunk_size', default=100, track=False,
                 help="Duration of each chunk in seconds"),
    strax.Option('nchunk', default=10, track=False,
                 help="Number of chunks to simulate"),
    strax.Option('right_raw_extension', default=50000),
    strax.Option('timeout', default=1800,
                 help="Terminate processing if any one mailbox receives "
                      "no result for more than this many seconds"),
    strax.Option('fax_config',
                 default='https://raw.githubusercontent.com/XENONnT/private_nt_aux_files/master/sim_files/fax_config_nt.json?token=AHCU5AZMPZABYSGVRLDACR3ABAZUA'),
    strax.Option('gain_model',
                 default=('to_pe_per_run', 'https://github.com/XENONnT/private_nt_aux_files/blob/master/sim_files/to_pe_nt.npy?raw=true'),
                 help='PMT gain model. Specify as (model_type, model_config).'),
    strax.Option('detector', default='XENONnT', track=True),
    strax.Option('channel_map', track=False, type=immutabledict,
                 help="immutabledict mapping subdetector to (min, max) "
                      "channel number. Provided by context"),
    strax.Option('n_tpc_pmts', track=False,
                 help="Number of pmts in tpc. Provided by context"),
    strax.Option('n_top_pmts', track=False,
                 help="Number of pmts in top array. Provided by context"),
    strax.Option('neutron_veto', default=False, track=True,
                 help="Flag for nVeto optical simulation instead of TPC"),
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

    # A very very long input timeout, our simulator takes time
    input_timeout = 3600 # as an hour

    def setup(self):
        c = self.config
        c.update(get_resource(c['fax_config'], fmt='json'))
        # Update gains to the nT defaults
        self.to_pe = get_to_pe(self.run_id, c['gain_model'],
                              c['channel_map']['tpc'][1]+1)
        c['gains'] = 1 / self.to_pe * (1e-8 * 2.25 / 2**14) / (1.6e-19 * 10 * 50)
        c['gains'][self.to_pe==0] = 0
        if c['seed'] != False:
            np.random.seed(c['seed'])

        overrides = self.config['fax_config_override']
        if overrides is not None:
            c.update(overrides)

        #We hash the config to load resources. Channel map is immutable and cannot be hashed
        self.config['channel_map'] = dict(self.config['channel_map'])
        self.config['channel_map']['sum_signal']=800
        self.config['channels_bottom'] = np.arange(self.config['n_top_pmts'],self.config['n_tpc_pmts'])
        
        self.get_instructions()
        self.check_instructions()
        self._setup()
    
    def _setup(self):
        #Set in inheriting class
        pass

    def get_instructions(self):
        #Set in inheriting class
        pass

    def check_instructions(self):
        #Set in inheriting class
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


@export
class RawRecordsFromFaxNT(FaxSimulatorPlugin):
    provides = ('raw_records', 'raw_records_he', 'raw_records_aqmon', 'truth')
    data_kind = immutabledict(zip(provides, provides))

    def _setup(self):
        self.sim = ChunkRawRecords(self.config)
        self.sim_iter = self.sim(self.instructions)

    def get_instructions(self):
        if self.config['fax_file']:
            assert self.config['fax_file'][-5:] != '.root', 'None optical g4 input is deprecated use EPIX instead'
            self.instructions = instruction_from_csv(self.config['fax_file'])
            self.config['nevents'] = np.max(self.instructions['event_number'])

        else:
            self.instructions = rand_instructions(self.config)

    def check_instructions(self):
        # Let below cathode S1 instructions pass but remove S2 instructions
        m = (self.instructions['z'] < -self.config['tpc_length']) & (self.instructions['type'] == 2)
        self.instructions = self.instructions[~m]

        assert np.all(self.instructions['x']**2 + self.instructions['y']**2 < self.config['tpc_radius']**2), \
                "Interation is outside the TPC"
        assert np.all(self.instructions['z'] < 0.25), \
                "Interation is outside the TPC"
        assert np.all(self.instructions['amp'] > 0), \
                "Interaction has zero size"


    def infer_dtype(self):
        dtype = {data_type:strax.raw_record_dtype(samples_per_record=strax.DEFAULT_RECORD_LENGTH) 
                for data_type in self.provides if data_type is not 'truth'}
        dtype['truth']=instruction_dtype + truth_extra_dtype
        return dtype


    def compute(self):
        try:
            result = next(self.sim_iter)
        except StopIteration:
            raise RuntimeError("Bug in chunk count computation")
        self._sort_check(result[self.provides[0]])
        #To accomodate nveto raw records, should be the first in provide.

        return {data_type:self.chunk(
            start=self.sim.chunk_time_pre,
            end=self.sim.chunk_time,
            data=result[data_type],
            data_type=data_type) for data_type in self.provides}


@export
@strax.takes_config(
    strax.Option('wfsim_instructions',track=False,default=False),)
class RawRecordsFromFaxEpix(RawRecordsFromFaxNT):

    # def _setup(self):
    #     self.sim = ChunkRawRecords(self.config)
    #     self.sim_iter = self.sim(self.instructions)
        
    # def compute(self):
    #     try:
    #         result = next(self.sim_iter)
    #     except StopIteration:
    #         raise RuntimeError("Bug in chunk count computation")
    #     self._sort_check(result['raw_records'])

    #     return {data_type:result[data_type] for data_type in self.provides}

    def get_instructions(self):
        if self.config['wfsim_instructions'] !=False:
            self.instructions=self.config['wfsim_instructions']  
        else:
            self.instructions=epix.run_epix(file=self.config['fax_file'],
                                        config=self.config,
                                        return_wfsim_instructions=True)

    def check_instructions(self):
        pass

    # def is_ready(self,chuck_i):
    #     """Overwritten to mimic online input plugin.
    #     Returns False to check source finished;
    #     Returns True to get next chunk.
    #     """
    #     return True


@export
class RawRecordsFromFax1T(RawRecordsFromFaxNT):
    provides = ('raw_records', 'truth')


@export
class RawRecordsFromFaxOptical(RawRecordsFromFaxNT):
    def _setup(self):
        self.sim = ChunkRawRecordsOptical(self.config)
        self.sim_iter = self.sim(instructions=self.instructions, 
                                 channels=self.channels, 
                                 timings=self.timings)

    def get_instructions(self):
        self.instructions, self.channels, self.timings = read_optical(self.config)
        self.config['nevents']=len(self.instructions['event_number'])


@export
@strax.takes_config(
    strax.Option('wfsim_nveto_instructions',track=False,default=False),
    strax.Option('nveto_channels',track=False,default=False),
    strax.Option('nveto_timings',track=False,default=False),
    )
class RawRecordsFromFaxnVeto(RawRecordsFromFaxOptical):
    provides = ('raw_records_nv', 'truth')
    data_kind = immutabledict(zip(provides, provides))
    # Why does the data_kind need to be repeated?? So the overriding of the 
    # provides doesn't work in the setting of the data__kind?

    def compute(self):
        result = super().compute()
        result['raw_records_nv'].data['channel'] += 2000  # nVeto PMT ID offset
        return result

    def get_instructions(self):
        if (self.config['wfsim_nveto_instructions']& self.config['nveto_channels'] &  self.config['nveto_timings']):
            self.instructions=self.config['instructions']
            self.channels=self.config['channels']
            self.timings=self.config['timings']
        else:
            super().get_instructions()

    def check_instructions(self):
        # Are there some nveto boundries we need to include?
        pass
