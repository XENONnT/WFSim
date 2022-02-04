from copy import deepcopy
from immutabledict import immutabledict
import logging
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import uproot
import typing as ty
import strax
import straxen
import wfsim
from strax.utils import tqdm

export, __all__ = strax.exporter()
logging.basicConfig(handlers=[logging.StreamHandler()])
log = logging.getLogger('wfsim.interface')
log.setLevel('WARNING')


__all__ += ['instruction_dtype', 'optical_extra_dtype', 'truth_extra_dtype']
_cached_wavelength_to_qe_arr = {}


instruction_dtype = [(('Waveform simulator event number.', 'event_number'), np.int32),
                     (('Quanta type (S1 photons or S2 electrons)', 'type'), np.int8),
                     (('Time of the interaction [ns]', 'time'), np.int64),
                     (('X position of the cluster [cm]', 'x'), np.float32),
                     (('Y position of the cluster [cm]', 'y'), np.float32),
                     (('Z position of the cluster [cm]', 'z'), np.float32),
                     (('Number of quanta', 'amp'), np.int32),
                     (('Recoil type of interaction.', 'recoil'), np.int8),
                     (('Energy deposit of interaction', 'e_dep'), np.float32),
                     (('Eventid like in geant4 output rootfile', 'g4id'), np.int32),
                     (('Volume id giving the detector subvolume', 'vol_id'), np.int32),
                     (('Local field [ V / cm ]', 'local_field'), np.float64),
                     (('Number of excitons', 'n_excitons'), np.int32),
]


optical_extra_dtype = [(('first optical input index', '_first'), np.int32),
                       (('last optical input index +1', '_last'), np.int32)]


truth_extra_dtype = [
    (('End time of the interaction [ns]', 'endtime'), np.int64),
    (('Number of simulated electrons', 'n_electron'), np.int32),
    (('Number of photons reaching PMT', 'n_photon'), np.int32),
    (('Number of photons + dpe passing', 'n_pe'), np.int32),
    (('Number of photons passing trigger', 'n_photon_trigger'), np.int32),
    (('Number of photons + dpe passing trigger', 'n_pe_trigger'), np.int32),
    (('Raw area in pe', 'raw_area'), np.float64),
    (('Raw area in pe passing trigger', 'raw_area_trigger'), np.float64),
    (('Number of photons reaching PMT (bottom)', 'n_photon_bottom'), np.int32),
    (('Number of photons + dpe passing (bottom)', 'n_pe_bottom'), np.int32),
    (('Number of photons passing trigger (bottom)', 'n_photon_trigger_bottom'), np.int32),
    (('Number of photons + dpe passing trigger (bottom)', 'n_pe_trigger_bottom'), np.int32),
    (('Raw area in pe (bottom)', 'raw_area_bottom'), np.float64),
    (('Raw area in pe passing trigger (bottom)', 'raw_area_trigger_bottom'), np.float64),
    (('Arrival time of the first photon [ns]', 't_first_photon'), np.float64),
    (('Arrival time of the last photon [ns]', 't_last_photon'), np.float64),
    (('Mean time of the photons [ns]', 't_mean_photon'), np.float64),
    (('Standard deviation of photon arrival times [ns]', 't_sigma_photon'), np.float64),
    (('X field-distorted mean position of the electrons [cm]', 'x_mean_electron'), np.float32),
    (('Y field-distorted mean position of the electrons [cm]', 'y_mean_electron'), np.float32),
    (('Arrival time of the first electron [ns]', 't_first_electron'), np.float64),
    (('Arrival time of the last electron [ns]', 't_last_electron'), np.float64),
    (('Mean time of the electrons [ns]', 't_mean_electron'), np.float64),
    (('Standard deviation of electron arrival times [ns]', 't_sigma_electron'), np.float64),]


def rand_instructions(c):
    """Random instruction generator function. This will be called by wfsim if you do not specify 
    specific instructions.
    :params c: wfsim configuration dict"""
    log.warning('rand_instructions is deprecated, please use wfsim.random_instructions')
    if 'drift_field' not in c:
        log.warning('drift field not specified!')
    # Do the conversion for now, don't break everything all at once
    return _rand_instructions(event_rate=c.get('event_rate', 10),
                              chunk_size=c.get('chunk_size', 5),
                              n_chunk=c.get('n_chunk', 2),
                              energy_range=[1, 100],  # keV
                              drift_field=c.get('drift_field', 100),  # V/cm
                              tpc_radius=c.get('tpc_radius', straxen.tpc_r),
                              tpc_length=c.get('tpc_length', straxen.tpc_z),
                              nest_inst_types=[7]
                              )


def random_instructions(**kwargs) -> np.ndarray:
    """
    Generate instructions to run WFSim
    :param event_rate: # events per second
    :param chunk_size: the size of each chunk
    :param n_chunk: the number of chunks
    :param drift_field:
    :param energy_range: the energy range (in keV) of the recoil type
    :param tpc_length: the max depth of the detector
    :param tpc_radius: the max radius of the detector
    :param nest_inst_types: the nestpy type of the interaction see:
        github.com/NESTCollaboration/nestpy/blob/8eb79414e5f834eb6cf6ddba5d6c433c6b0cbc70/src/nestpy/helpers.py#L27
    :return: Instruction array that can be used for simulation in wfsim
    """
    return _rand_instructions(**kwargs)


def _rand_instructions(
        event_rate: int,
        chunk_size: int,
        n_chunk: int,
        drift_field: float,
        energy_range: ty.Union[tuple, list, np.ndarray],
        tpc_length: float = straxen.tpc_z,
        tpc_radius: float = straxen.tpc_r,
        nest_inst_types: ty.Union[ty.List[int], ty.Tuple[ty.List], np.ndarray, None] = None,
) -> np.ndarray:
    import nestpy
    if nest_inst_types is None:
        nest_inst_types = [7]

    n_events = event_rate * chunk_size * n_chunk
    total_time = chunk_size * n_chunk

    inst = np.zeros(2 * n_events, dtype=instruction_dtype)
    inst[:] = -1

    uniform_times = total_time * (np.arange(n_events) + 0.5) / n_events

    inst['time'] = np.repeat(uniform_times, 2) * int(1e9)
    inst['event_number'] = np.digitize(inst['time'],
                                       1e9 * np.arange(n_chunk) *
                                       chunk_size) - 1
    inst['type'] = np.tile([1, 2], n_events)

    r = np.sqrt(np.random.uniform(0, tpc_radius ** 2, n_events))
    t = np.random.uniform(-np.pi, np.pi, n_events)
    inst['x'] = np.repeat(r * np.cos(t), 2)
    inst['y'] = np.repeat(r * np.sin(t), 2)
    inst['z'] = np.repeat(np.random.uniform(-tpc_length, 0, n_events), 2)

    # Here we'll define our XENON-like detector
    nest_calc = nestpy.NESTcalc(nestpy.VDetector())
    nucleus_A = 131.293
    nucleus_Z = 54.
    lxe_density = 2.862  # g/cm^3   #SR1 Value

    energy = np.random.uniform(*energy_range, n_events)
    quanta = []
    exciton = []
    recoil = []
    e_dep = []
    for energy_deposit in tqdm(energy, desc='generating instructions from nest'):
        interaction_type = np.random.choice(nest_inst_types)
        interaction = nestpy.INTERACTION_TYPE(interaction_type)
        y = nest_calc.GetYields(interaction,
                                energy_deposit,
                                lxe_density,
                                drift_field,
                                nucleus_A,
                                nucleus_Z,
                                )
        q = nest_calc.GetQuanta(y, lxe_density)
        quanta.append(q.photons)
        quanta.append(q.electrons)
        exciton.append(q.excitons)
        exciton.append(0)
        # both S1 and S2
        recoil += [interaction_type, interaction_type]
        e_dep += [energy_deposit, energy_deposit]

    inst['amp'] = quanta
    inst['local_field'] = drift_field
    inst['n_excitons'] = exciton
    inst['recoil'] = recoil
    inst['e_dep'] = e_dep
    for field in inst.dtype.names:
        if np.any(inst[field] == -1):
            log.warning(f'{field} is not (fully) filled')
    return inst


def _read_optical_nveto(config, events, mask):
    """Helper function for nveto to read photon channels and timings from G4 and apply QE's
    :params config: dict, wfsim configuration
    :params events: g4 root file
    :params mask: 1d bool array to select events
    
    returns two flatterned nested arrays of channels and timings, 
    """
    channels = np.hstack(events["pmthitID"].array(library="np")[mask])
    timings = np.hstack(events["pmthitTime"].array(library="np")[mask] * 1e9).astype(np.int64)

    constant_hc = 1239.841984  # (eV*nm) to calculate (wavelength lambda) = h * c / energy
    wavelengths = np.hstack(constant_hc / events["pmthitEnergy"].array(library="np")[mask])

    # Caching a 2d array of interpolated value of (channel, wavelength [every nm]) 
    h = strax.deterministic_hash(config)
    nveto_channels = np.arange(config['channel_map']['nveto'][0], config['channel_map']['nveto'][1] + 1)
    if h not in _cached_wavelength_to_qe_arr:
        resource = wfsim.load_config(config)
        if getattr(resource, 'nv_pmt_qe', None) is None:
            log.warning('nv pmt qe data not specified all qe default to 100 %')
            _cached_wavelength_to_qe_arr[h] = np.ones([len(nveto_channels), 1000]) * 100
        else:
            qe_data = resource.nv_pmt_qe
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

    wavelengths[(wavelengths < 0) | (wavelengths >= 999)] = 0
    qes = _cached_wavelength_to_qe_arr[h][channels - nveto_channels[0],
                                          np.around(wavelengths).astype(np.int64)]
    hit_mask &= np.random.rand(len(qes)) <= qes * config.get('nv_pmt_ce_factor', 1.0) / 100

    amplitudes, og_offset = [], 0
    for tmp in events["pmthitID"].array(library="np")[mask]:
        og_length = len(tmp)
        amplitudes.append(hit_mask[og_offset:og_offset + og_length].sum())
        og_offset += og_length
    
    return channels[hit_mask], timings[hit_mask], np.array(amplitudes, int)


@export
def read_optical(config):
    """Function will be executed when wfsim in run in optical mode. This function expects c['fax_file'] 
    to be a root file from optical mc
    :params config: wfsim configuration dict"""
    data = uproot.open(config['fax_file'])
    try:
        events = data.get('events')
    except AttributeError:
        raise Exception("Are you using mc version >4?")

    # Slightly weird here. Entry_stop is not in the regular config, so if it's not skip this part
    g4id = events['eventid'].array(library="np")
    if config.get('entry_stop', None) is None:
        config['entry_stop'] = np.max(g4id) + 1

    mask = ((g4id < config.get('entry_stop', int(2**63-1)))
            & (g4id >= config.get('entry_start', 0)))
    n_events = len(g4id[mask])

    if config['detector'] == 'XENONnT_neutron_veto':
        channels, timings, amplitudes = _read_optical_nveto(config, events, mask)
        # Still need to shift nveto channel for simulation code to work
        channels -= config['channel_map']['nveto'][0]
    else:
        # TPC
        channels = np.hstack(events["pmthitID"].array(library="np")[mask])
        timings = np.hstack(events["pmthitTime"].array(library="np")[mask] * 1e9).astype(np.int64)
        amplitudes = np.array([len(tmp) for tmp in events["pmthitID"].array(library="np")[mask]])

    # Events should be in the TPC
    ins = np.zeros(n_events, dtype=instruction_dtype + optical_extra_dtype)
    ins['x'] = events["xp_pri"].array(library="np").flatten()[mask] / 10.
    ins['y'] = events["yp_pri"].array(library="np").flatten()[mask] / 10.
    ins['z'] = events["zp_pri"].array(library="np").flatten()[mask] / 10.
    ins['time'] = np.zeros(n_events, np.int64)
    ins['event_number'] = np.arange(n_events)
    ins['g4id'] = events['eventid'].array(library="np")[mask]
    ins['type'] = np.repeat(1, n_events)
    ins['recoil'] = np.repeat(1, n_events)
    ins['_first'] = np.cumsum(amplitudes) - amplitudes
    ins['_last'] = np.cumsum(amplitudes)

    # Need to shift the timing and split long pulses
    ins = wfsim.optical_adjustment(ins, timings, channels)
    return ins, channels, timings


def instruction_from_csv(filename):
    """
    Return wfsim instructions from a csv
    :param filename: Path to csv file
    """
    df = pd.read_csv(filename)

    recs = np.zeros(len(df), dtype=instruction_dtype)
    for column in df.columns:
        recs[column] = df[column]

    expected_dtype = np.dtype(instruction_dtype)
    assert recs.dtype == expected_dtype, \
        f"CSV {filename} produced wrong dtype. Got {recs.dtype}, expected {expected_dtype}."
    return recs


@export
class ChunkRawRecords(object):
    def __init__(self, config, rawdata_generator=wfsim.RawData, **kwargs):
        log.debug(f'Starting {self.__class__.__name__}')
        self.config = config
        log.debug(f'Setting raw data with {rawdata_generator.__name__}')
        self.rawdata = rawdata_generator(self.config, **kwargs)
        self.record_buffer = np.zeros(5000000,
                                      dtype=strax.raw_record_dtype(samples_per_record=strax.DEFAULT_RECORD_LENGTH))
        self.truth_buffer = np.zeros(10000, dtype=instruction_dtype + truth_extra_dtype + [('fill', bool)])

        self.blevel = 0  # buffer_filled_level

    def __call__(self, instructions, time_zero=None, **kwargs):
        """
        :param instructions: Structured array with instruction dtype in strax_interface module
        :param time_zero: Starting time of the first chunk
        """
        samples_per_record = strax.DEFAULT_RECORD_LENGTH
        if len(instructions) == 0: # Empty
            yield from np.array([], dtype=strax.raw_record_dtype(samples_per_record=samples_per_record))
            self.rawdata.source_finished = True
            return
        dt = self.config['sample_duration']
        buffer_length = len(self.record_buffer)
        rext = int(self.config['right_raw_extension'])
        cksz = int(self.config['chunk_size'] * 1e9)

        # Save the constants as privates
        self.blevel = 0  # buffer_filled_level
        self.chunk_time_pre = time_zero - rext if time_zero else np.min(instructions['time']) - rext
        self.chunk_time = self.chunk_time_pre + cksz  # Starting chunk
        self.current_digitized_right = self.last_digitized_right = 0
        for channel, left, right, data in self.rawdata(instructions=instructions,
                                                       truth_buffer=self.truth_buffer,
                                                       **kwargs):
            pulse_length = right - left + 1
            records_needed = int(np.ceil(pulse_length / samples_per_record))

            if self.rawdata.right != self.current_digitized_right:
                self.last_digitized_right = self.current_digitized_right
                self.current_digitized_right = self.rawdata.right

            if self.rawdata.left * dt > self.chunk_time + rext:
                next_left_time = self.rawdata.left * dt
                log.debug(f'Pause sim loop at {self.chunk_time}, next pulse start at {next_left_time}')
                if (self.last_digitized_right + 1) * dt > self.chunk_time:
                    extend = (self.last_digitized_right + 1) * dt - self.chunk_time
                    self.chunk_time += extend 
                    log.debug(f'Chunk happenned during event, extending {extend} ns')
                yield from self.final_results()
                self.chunk_time_pre = self.chunk_time
                self.chunk_time += cksz

            if self.blevel + records_needed > buffer_length:
                log.warning('Chunck size too large, insufficient record buffer \n'
                            'No longer in sync if simulating nVeto with TPC \n'
                            'Consider reducing the chunk size')
                next_left_time = self.rawdata.left * dt
                self.chunk_time = (self.last_digitized_right + 1) * dt
                log.debug(f'Pause sim loop at {self.chunk_time}, next pulse start at {next_left_time}')
                yield from self.final_results()
                self.chunk_time_pre = self.chunk_time
                self.chunk_time += cksz

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
                                                   (0, records_needed * samples_per_record - pulse_length),
                                                   'constant').reshape((-1, samples_per_record))
            self.blevel += records_needed

        self.last_digitized_right = self.current_digitized_right
        self.chunk_time = max((self.last_digitized_right + 1) * dt, self.chunk_time_pre + dt)
        yield from self.final_results()

    def final_results(self):
        records = self.record_buffer[:self.blevel]  # No copying the records from buffer
        log.debug(f'Yielding chunk from {self.rawdata.__class__.__name__} '
                  f'between {self.chunk_time_pre} - {self.chunk_time}')
        maska = records['time'] <= self.chunk_time
        if self.blevel >= 1:
            max_r_time = records['time'].max()
            log.debug(f'Truncating data at sample time {self.chunk_time}, last record time {max_r_time}')
        else:
            log.debug(f'Truncating data at sample time {self.chunk_time}, no record is produced')
        records = records[maska]
        records = strax.sort_by_time(records)  # Do NOT remove this line

        # Yield an appropriate amount of stuff from the truth buffer
        # and mark it as available for writing again

        maskb = (
            self.truth_buffer['fill'] &
            # This condition will always be false if self.truth_buffer['t_first_photon'] == np.nan
            ((self.truth_buffer['t_first_photon'] <= self.chunk_time) |
             # Hence, we need to use this trick to also save these cases (this
             # is what we set the end time to for np.nans)
             (np.isnan(self.truth_buffer['t_first_photon']) &
              (self.truth_buffer['time'] <= self.chunk_time)))
        )
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

        # Oke this will be a bit ugly but it's easy
        if self.config['detector'] == 'XENON1T' or self.config['detector'] == 'XENONnT_neutron_veto':
            yield dict(raw_records=records,
                       truth=_truth)
        elif self.config['detector'] == 'XENONnT':
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
    strax.Option('detector', default='XENONnT', track=True, infer_type=False),
    strax.Option('event_rate', default=1000, track=False, infer_type=False,
                 help="Average number of events per second"),
    strax.Option('chunk_size', default=100, track=False, infer_type=False,
                 help="Duration of each chunk in seconds"),
    strax.Option('n_chunk', default=10, track=False, infer_type=False,
                 help="Number of chunks to simulate"),
    strax.Option('fax_file', default=None, track=False, infer_type=False,
                 help="Directory with fax instructions"), 
    strax.Option('fax_config', default='fax_config_nt_design.json'),
    strax.Option('fax_config_override', default=None, infer_type=False,
                 help="Dictionary with configuration option overrides"),
    strax.Option('fax_config_override_from_cmt', default=None, infer_type=False,
                 help="Dictionary of fax parameter names (key) mapped to CMT config names (value)"
                      "where the fax parameter values will be replaced by CMT"),
    strax.Option('gain_model_mc', default=('to_pe_per_run', 'to_pe_nt.npy'), infer_type=False,
                 help='PMT gain model. Specify as (model_type, model_config).'),
    strax.Option('channel_map', track=False, type=immutabledict,
                 help="immutabledict mapping subdetector to (min, max) "
                      "channel number. Provided by context"),
    strax.Option('n_tpc_pmts', track=False, infer_type=False,
                 help="Number of pmts in tpc. Provided by context"),
    strax.Option('n_top_pmts', track=False, infer_type=False,
                 help="Number of pmts in top array. Provided by context"),
    strax.Option('right_raw_extension', default=100000, infer_type=False),
    strax.Option('seed', default=False, track=False, infer_type=False,
                 help="Option for setting the seed of the random number generator used for"
                      "generation of the instructions"),
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
    input_timeout = 3600  # as an hour

    def setup(self):
        self.set_config()
        self.get_instructions()
        self.check_instructions()
        self._setup()

    def set_config(self,):
        self.config.update(straxen.get_resource(self.config['fax_config'], fmt='json'))
        overrides = self.config['fax_config_override']
        if overrides is not None:
            self.config.update(overrides)
            
        # backwards compatibility
        if 'field_distortion_on' in self.config and not 'field_distortion_model' in self.config:
            self.config.update({'field_distortion_model': "inverse_fdc" if self.config['field_distortion_on'] else "none"})

        # Update gains to the nT defaults
        self.to_pe = straxen.get_correction_from_cmt(self.run_id,
                               self.config['gain_model_mc'])

        adc_2_current = (self.config['digitizer_voltage_range']
                         / 2 ** (self.config['digitizer_bits'])
                         / self.config['pmt_circuit_load_resistor'])

        self.config['gains'] = np.divide(adc_2_current,
                                         self.to_pe,
                                         out=np.zeros_like(self.to_pe),
                                         where=self.to_pe != 0)

        if self.config['seed']:
            np.random.seed(self.config['seed'])

        # We hash the config to load resources. Channel map is immutable and cannot be hashed
        self.config['channel_map'] = dict(self.config['channel_map'])
        self.config['channel_map']['sum_signal'] = 800
        self.config['channels_bottom'] = np.arange(self.config['n_top_pmts'], self.config['n_tpc_pmts'])

        # Update some values stored in CMT
        if self.config['fax_config_override_from_cmt'] is not None:
            for fax_field, cmt_option in self.config['fax_config_override_from_cmt'].items():
                if (fax_field in ['fdc_3d', 's1_lce_correction_map']
                    and self.config.get('default_reconstruction_algorithm', False)):
                    cmt_option = tuple(['suffix',
                                        self.config['default_reconstruction_algorithm'],
                                        *cmt_option])

                cmt_value = straxen.get_correction_from_cmt(self.run_id, cmt_option)
                log.warning(f'Replacing {fax_field} with CMT option {cmt_option} to {cmt_value}')
                self.config[fax_field] = cmt_value

    def _setup(self):
        # Set in inheriting class
        pass

    def get_instructions(self):
        # Set in inheriting class
        pass

    def check_instructions(self):
        # Set in inheriting class
        pass

    def _sort_check(self, results):
        if not isinstance(results, list):
            results = [results]
        last_chunk_time = self.last_chunk_time
        for result in results:
            if len(result) == 0:
                continue
            if result['time'][0] < self.last_chunk_time + 1000:
                raise RuntimeError(
                    "Simulator returned chunks with insufficient spacing. "
                    f"Last chunk's max time was {self.last_chunk_time}, "
                    f"this chunk's first time is {result['time'][0]}.")
            if len(result) == 1:
                continue
            if np.diff(result['time']).min() < 0:
                raise RuntimeError("Simulator returned non-sorted records!")
            
            last_chunk_time = max(result['time'].max(), self.last_chunk_time)
        self.last_chunk_time = last_chunk_time

    def is_ready(self, chunk_i):
        """Overwritten to mimic online input plugin.
        Returns False to check source finished;
        Returns True to get next chunk.
        """
        if 'ready' not in self.__dict__:
            self.ready = False
        self.ready ^= True  # Flip
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
        dtype = {data_type: strax.raw_record_dtype(samples_per_record=strax.DEFAULT_RECORD_LENGTH)
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

        return {data_type: self.chunk(
            start=self.sim.chunk_time_pre,
            end=self.sim.chunk_time,
            data=result[data_type],
            data_type=data_type) for data_type in self.provides}


@export
class RawRecordsFromFax1T(RawRecordsFromFaxNT):
    provides = ('raw_records', 'truth')


@export
class RawRecordsFromFaxOpticalNT(RawRecordsFromFaxNT):
    
    def _setup(self):
        self.sim = ChunkRawRecords(self.config,
                                   rawdata_generator=wfsim.RawDataOptical,
                                   channels=self.channels,
                                   timings=self.timings,)
        self.sim.truth_buffer = np.zeros(10000, dtype=instruction_dtype + optical_extra_dtype
                                                + truth_extra_dtype + [('fill', bool)])
        self.sim_iter = self.sim(self.instructions)
        
        
    def get_instructions(self):
        assert self.config['fax_file'].endswith('.root'), 'You need to supply a root file for optical simulation!'
        self.instructions, self.channels, self.timings = read_optical(self.config)
        
    
@export
@strax.takes_config(
    strax.Option('epix_config', track=False, default={}, infer_type=False,
                 help='Dict with epix configuration'),
    strax.Option('entry_start', default=0, track=False, infer_type=False,),
    strax.Option('entry_stop', default=None, track=False, infer_type=False,
                 help='G4 id event number to stop at. If -1 process the entire file'),
    strax.Option('fax_config_nveto', default=None, track=True, infer_type=False,),
    strax.Option('fax_config_override_nveto', default=None, track=True, infer_type=False,
                 help='Dictionary with configuration option overrides'),
    strax.Option('gain_model_nv', track=True, infer_type=False,
                 help='nveto gain model, provided by context'),
    strax.Option('targets', default=('tpc',), track=False, infer_type=False,
                 help='tuple with what data to simulate (tpc, nveto or both)')
)
class RawRecordsFromMcChain(SimulatorPlugin):
    provides = ('raw_records', 'raw_records_he', 'raw_records_aqmon', 'raw_records_nv', 'truth', 'truth_nv')
    data_kind = immutabledict(zip(provides, provides))

    def set_config(self,):
        super().set_config()

        if 'nveto' in self.config['targets']:
            self.config_nveto = deepcopy(self.config)
            self.config_nveto.update(straxen.get_resource(self.config_nveto['fax_config_nveto'], fmt='json'))
            self.config_nveto['detector'] = 'XENONnT_neutron_veto'
            self.config_nveto['channel_map'] = dict(self.config_nveto['channel_map'])
            overrides = self.config['fax_config_override_nveto']
            if overrides is not None:
                self.config_nveto.update(overrides)

            self.to_pe_nveto = straxen.get_correction_from_cmt(self.run_id,
                               self.config['gain_model_nv'])

            self.config_nveto['gains'] = np.divide((2e-9 * 2 / 2**14) / (1.6e-19 * 1 * 50),
                                                   self.to_pe_nveto,
                                                   out=np.zeros_like(self.to_pe_nveto),
                                                   where=self.to_pe_nveto != 0)
            self.config_nveto['channels_bottom'] = np.array([], np.int64)

    def get_instructions(self):
        """
        Run epix and save instructions as self.instructions_epix
        epix_config with all the epix run arguments is passed as a dictionary, where
        source_rate must be set to 0 (epix default), since time handling is done outside epix.
        
        epix needs to be imported in here to avoid circle imports
        """

        self.g4id = []
        if 'tpc' in self.config['targets']:
            import epix
            epix_config = deepcopy(self.config['epix_config'])  # dictionary directly fed to context
            epix_config.update({
                'detector': self.config['detector'],
                'entry_start': self.config['entry_start'],
                'entry_stop': self.config['entry_stop'],
                'input_file': self.config['fax_file']})
            self.instructions_epix = epix.run_epix.main(
                epix.run_epix.setup(epix_config),
                return_wfsim_instructions=True)

            self.g4id.append(self.instructions_epix['g4id'])
            log.debug("Epix produced %d instructions in tpc" % (len(self.instructions_epix)))

        if 'nveto' in self.config['targets']:
            self.instructions_nveto, self.nveto_channels, self.nveto_timings =\
                read_optical(self.config_nveto)
            if len(self.g4id) > 0:
                nv_inst_to_keep = (self.instructions_nveto['_last'] - self.instructions_nveto['_first']) >= 0

            self.instructions_nveto = self.instructions_nveto[nv_inst_to_keep]
            self.g4id.append(self.instructions_nveto['g4id'])
            log.debug("%d instructions were produced in nv" % (len(self.instructions_nveto)))

        self.g4id = np.unique(np.concatenate(self.g4id))
        self.set_timing()

    def set_timing(self,):
        """Set timing information in such a way to synchronize instructions for the TPC and nVeto"""

        # If no event_stop is specified set it to the maximum (-1=do all events)
        if self.config['entry_stop'] is None:
            self.config['entry_start'] = np.min(self.g4id)
            self.config['entry_stop'] = np.max(self.g4id) + 1
        log.debug('Entry stop set at %d, g4id min %d max %d'
                  % (self.config['entry_stop'], np.min(self.g4id), np.max(self.g4id)))

        # Convert rate from Hz to ns^-1
        rate = self.config['event_rate'] / 1e9
        # Add half interval to avoid time 0
        timings = np.random.uniform((self.config['entry_start'] + 0.5) / rate, 
                                    (self.config['entry_stop'] + 0.5) / rate, 
                                    self.config['entry_stop'] - self.config['entry_start'])
        timings = np.sort(timings).astype(np.int64)
        max_time = int((self.config['entry_stop'] + 0.5) / rate)

        # For tpc we have multiple instructions per g4id.
        if 'tpc' in self.config['targets']:
            i_timings = np.searchsorted(np.arange(self.config['entry_start'], self.config['entry_stop']),
                                        self.instructions_epix['g4id'])
            self.instructions_epix['time'] += timings[i_timings]

            extra_long = self.instructions_epix['time'] > max_time
            self.instructions_epix = self.instructions_epix[~extra_long]
            log.warning('Found and removing %d epix instructions later than maximum time %d'
                        % (extra_long.sum(), max_time))

        # nveto instruction doesn't carry physical time delay, so the time is overwritten
        if 'nveto' in self.config['targets']:
            i_timings = np.searchsorted(np.arange(self.config['entry_start'], self.config['entry_stop']),
                                        self.instructions_nveto['g4id'])
            self.instructions_nveto['time'] += timings[i_timings]

            extra_long = self.instructions_nveto['time'] > max_time
            self.instructions_nveto = self.instructions_nveto[~extra_long]
            log.warning('Found and removing %d nveto instructions later than maximum time %d'
                        % (extra_long.sum(), max_time))

    def check_instructions(self):
        if 'tpc' in self.config['targets']:
            # Let below cathode S1 instructions pass but remove S2 instructions
            m = (self.instructions_epix['z'] < - self.config['tpc_length']) & (self.instructions_epix['type'] == 2)
            self.instructions_epix = self.instructions_epix[~m]

            assert np.all(self.instructions_epix['x']**2 + self.instructions_epix['y']**2 <
                          self.config['tpc_radius']**2), \
                "Interation is outside the TPC"
            assert np.all(self.instructions_epix['z'] < 0.25), \
                "Interation is outside the TPC"
            assert np.all(self.instructions_epix['amp'] > 0), \
                "Interaction has zero size"
            assert all(self.instructions_epix['g4id'] >= self.config['entry_start'])
            assert all(self.instructions_epix['g4id'] < self.config['entry_stop'])

        if 'nveto' in self.config['targets']:
            assert all(self.instructions_nveto['g4id'] >= self.config['entry_start'])
            assert all(self.instructions_nveto['g4id'] < self.config['entry_stop'])

    def _setup(self):
        if 'tpc' in self.config['targets']:
            self.sim = ChunkRawRecords(self.config)
            self.sim_iter = self.sim(
                self.instructions_epix,
                time_zero=int((self.config['entry_start'] + 0.5) / self.config['event_rate'] * 1e9),
                progress_bar=True)

        if 'nveto' in self.config['targets']:
            self.sim_nv = ChunkRawRecords(self.config_nveto,
                                          rawdata_generator=wfsim.RawDataOptical,
                                          channels=self.nveto_channels,
                                          timings=self.nveto_timings,)
            self.sim_nv.truth_buffer = np.zeros(10000, dtype=instruction_dtype + optical_extra_dtype
                                                + truth_extra_dtype + [('fill', bool)])

            assert '_first' in self.instructions_nveto.dtype.names, 'Require indexing info in optical ' \
                                                                    'instruction see optical extra dtype'
            assert all(self.instructions_nveto['type'] == 1), 'Only s1 type ' \
                                                              'is supported for generating rawdata from optical input'
            self.sim_nv_iter = self.sim_nv(
                self.instructions_nveto,
                time_zero=int((self.config['entry_start'] + 0.5) / self.config['event_rate'] * 1e9),
                progress_bar=True)

    def infer_dtype(self):
        dtype = dict([(data_type, instruction_dtype + truth_extra_dtype) if 'truth' in data_type
                      else (data_type, strax.raw_record_dtype(samples_per_record=strax.DEFAULT_RECORD_LENGTH))
                      for data_type in self.provides])
        return dtype

    def compute(self):
        log.debug('Full chain plugin calling compute')
        if 'tpc' in self.config['targets']:
            try:
                result = next(self.sim_iter)
            except StopIteration:
                if self.sim.source_finished():
                    log.debug('TPC instructions are already depleted')
                    result = dict([(data_type, np.zeros(0, self.dtype_for(data_type)))
                                   for data_type in self.provides if 'nv' not in data_type])

                    num_of_results = sum([len(v) for _, v in result.items()])
                    if num_of_results != 0:
                        self.sim.chunk_time = self.sim_nv.chunk_time
                        self.sim.chunk_time_pre = self.sim_nv.chunk_time_pre
                else:
                    raise RuntimeError("Bug in getting source finished")

        if 'nveto' in self.config['targets']:
            try:
                result_nv = next(self.sim_nv_iter)
                result_nv['raw_records']['channel'] += self.config['channel_map']['nveto'][0]
            except StopIteration:
                if self.sim_nv.source_finished():
                    log.debug('nVeto instructions are already depleted')
                    result_nv = dict([(data_type.strip('_nv'), np.zeros(0, self.dtype_for(data_type)))
                                      for data_type in self.provides if 'nv' in data_type])
                    self.sim_nv.chunk_time = self.sim.chunk_time
                    self.sim_nv.chunk_time_pre = self.sim.chunk_time_pre
                else:
                    raise RuntimeError("Bug in getting source finished")

        exist_tpc_result, exist_nveto_result = False, False
        for data_type in self.provides:
            if 'nv' in data_type:
                if 'nveto' in self.config['targets']:
                    if len(result_nv[data_type.strip('_nv')]) > 0:
                        exist_nveto_result = True
            else:
                if len(result[data_type]) > 0:
                    exist_tpc_result = True
        chunk = {}
        for data_type in self.provides:
            if 'nv' in data_type:
                if exist_nveto_result:
                    chunk[data_type] = self.chunk(start=self.sim_nv.chunk_time_pre,
                                                end=self.sim_nv.chunk_time,
                                                data=result_nv[data_type.strip('_nv')],
                                                data_type=data_type)
                # If nv is not one of the targets just return an empty chunk
                # If there is TPC event, set TPC time for the start and end
                else:
                    dummy_dtype = wfsim.truth_extra_dtype if 'truth' in data_type else strax.raw_record_dtype()
                    if exist_tpc_result:
                        chunk[data_type] = self.chunk(start=self.sim.chunk_time_pre,
                                                end=self.sim.chunk_time,
                                                data=np.array([], dtype=dummy_dtype),
                                                data_type=data_type)
                    else:
                        chunk[data_type] = self.chunk(start=0, end=0, data=np.array([], dtype=dummy_dtype),
                                                  data_type=data_type)
            else:
                if exist_tpc_result:
                    chunk[data_type] = self.chunk(start=self.sim.chunk_time_pre,
                                                  end=self.sim.chunk_time,
                                              data=result[data_type],
                                              data_type=data_type)
                else:
                    dummy_dtype = wfsim.truth_extra_dtype if 'truth' in data_type else strax.raw_record_dtype()
                    if exist_nveto_result:
                        chunk[data_type] = self.chunk(start=self.sim_nv.chunk_time_pre,
                                                      end=self.sim_nv.chunk_time,
                                                  data=np.array([], dtype=dummy_dtype),
                                                  data_type=data_type)
                    else:
                        chunk[data_type] = self.chunk(start=0, end=0, data=np.array([], dtype=dummy_dtype),
                                                  data_type=data_type)

        self._sort_check([chunk[data_type].data for data_type in self.provides])

        return chunk

    def source_finished(self):
        """Return whether all instructions has been used."""
        source_finished = True
        if 'tpc' in self.config['targets']:
            source_finished &= self.sim.source_finished()
        if 'nveto' in self.config['targets']:
            source_finished &= self.sim_nv.source_finished()
        return source_finished


@export
class RawRecordsFromFaxnVeto(RawRecordsFromMcChain):
    provides = ('raw_records_nv', 'truth_nv')
    data_kind = immutabledict(zip(provides, provides))


@export
class RawRecordsFromMcChain1T(RawRecordsFromMcChain):
    provides = ('raw_records', 'truth')
    data_kind = immutabledict(zip(provides, provides))
