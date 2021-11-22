import logging
from numba import njit
import numpy as np
from strax import exporter
from .pulse import Pulse


export, __all__ = exporter()
logging.basicConfig(handlers=[logging.StreamHandler()])
log = logging.getLogger('wfsim.core')
log.setLevel('WARNING')


try:
    import nestpy
except (ModuleNotFoundError, ImportError):
    log.warning('Nestpy is not found, + nest mode will not work!')


@export
class NestId:
    """
    Nest ids for referring to different scintillation models, only ER is actually validated
    See github.com/NESTCollaboration/nestpy/blob/8eb79414e5f834eb6cf6ddba5d6c433c6b0cbc70/src/nestpy/helpers.py#L27
    """
    NR = [0]
    ALPHA = [6]
    ER = [7, 8, 11, 12]
    LED = [20]
    _ALL = NR + ALPHA + ER + LED


@export
class S1(Pulse):
    """
    Given temperal inputs as well as number of photons
    Random generate photon timing and channel distribution.
    """
    nestpy_calc = None

    def __init__(self, config):
        super().__init__(config)
        self.phase = 'liquid'  # To distinguish singlet/triplet time delay.
        if 'nest' in self.config['s1_model_type'] and (self.nestpy_calc is None):
            log.info('Using NEST for scintillation time without set calculator\n'
                     'Creating new nestpy calculator')
            self.nestpy_calc = nestpy.NESTcalc(nestpy.DetectorExample_XENON10())

        # Check if user specified s1 model type exist
        S1VALIDTYPE = ['', 'simple', 'custom', 'optical_propagation', 'nest']
        def s1_valid_type(s, c='+ ,'):
            if len(c) > 0:
                for k in s.split(c[0]):  
                    s1_valid_type(k, c[1:])
            else:
                assert s in S1VALIDTYPE, f'Model type "{s}" not in {S1VALIDTYPE}'

        s1_valid_type(self.config['s1_model_type'])

    def __call__(self, instruction):
        """Main s1 simulation function. Called by RawData for s1 simulation. 
        Generates first number of photons in the s1, then timings and channels.
        These arrays are fed to Pulse to generate the data.

        param instructions: Array with dtype wfsim.instruction_dtype """
        if len(instruction.shape) < 1:
            # shape of recarr is a bit strange
            instruction = np.array([instruction])

        # _, _, t, x, y, z, n_photons, recoil_type, *rest = [
        #    np.array(v).reshape(-1) for v in zip(*instruction)]
        t = instruction['time']
        x = instruction['x']
        y = instruction['y']
        z = instruction['z']
        n_photons = instruction['amp']
        recoil_type = instruction['recoil']
        positions = np.array([x, y, z]).T  # For map interpolation
        n_photon_hits = self.get_n_photons(n_photons=n_photons,
                                           positions=positions,
                                           s1_lce_correction_map=self.resource.s1_lce_correction_map,
                                           config=self.config)

        # The new way interpolation is written always require a list
        self._photon_channels = self.photon_channels(positions=positions,
                                                     n_photon_hits=n_photon_hits,
                                                     config=self.config, 
                                                     s1_pattern_map=self.resource.s1_pattern_map)

        extra_targs = {}
        if 'nest' in self.config['s1_model_type']:
            extra_targs['n_photons_emitted'] = n_photons
            extra_targs['n_excitons'] = instruction['n_excitons']
            extra_targs['local_field'] = instruction['local_field']
            extra_targs['e_dep'] = instruction['e_dep']
            extra_targs['nestpy_calc'] = self.nestpy_calc

        self._photon_timings = self.photon_timings(t=t,
                                                   n_photon_hits=n_photon_hits, 
                                                   recoil_type=recoil_type,
                                                   config=self.config,
                                                   phase=self.phase,
                                                   channels=self._photon_channels,
                                                   positions=positions,
                                                   resource=self.resource,
                                                   **extra_targs)

        # Sorting times according to the channel, as non-explicit sorting
        # is performed later and this breaks timing of individual channels/arrays
        sortind = np.argsort(self._photon_channels)

        self._photon_channels = self._photon_channels[sortind]
        self._photon_timings = self._photon_timings[sortind]
        super().__call__()

    @staticmethod
    def get_n_photons(n_photons, positions, s1_lce_correction_map, config):
        """Calculates number of detected photons based on number of photons in total and the positions
        :param n_photons: 1d array of ints with number of emitted S1 photons:
        :param positions: 2d array with xyz positions of interactions
        :param s1_lce_correction_map: interpolator instance of s1 light yield map
        :param config: dict wfsim config 
        
        return array with number photons"""
        ly = s1_lce_correction_map(positions)
        # depending on if you use the data driven or mc pattern map for light yield 
        #the shape of n_photon_hits will change. Mc needs a squeeze
        if len(ly.shape) != 1:
            ly = np.squeeze(ly, axis=-1)
        ly /= 1 + config['p_double_pe_emision']
        ly *= config['s1_detection_efficiency']

        n_photon_hits = np.random.binomial(n=n_photons, p=ly)
        
        return n_photon_hits

    @staticmethod
    def photon_channels(positions, n_photon_hits, config, s1_pattern_map):
        """Calculate photon arrival channels
        :params positions: 2d array with xy positions of interactions
        :params n_photon_hits: 1d array of ints with number of photon hits to simulate
        :params config: dict wfsim config
        :params s1_pattern_map: interpolator instance of the s1 pattern map

        returns nested array with photon channels   
        """
        channels = np.arange(config['n_tpc_pmts'])  # +1 for the channel map
        p_per_channel = s1_pattern_map(positions)
        p_per_channel[:, np.in1d(channels, config['turned_off_pmts'])] = 0

        _photon_channels = np.array([]).astype(np.int64)
        for ppc, n in zip(p_per_channel, n_photon_hits):
            _photon_channels = np.append(_photon_channels,
                                         np.random.choice(
                                             channels,
                                             size=n,
                                             p=ppc / np.sum(ppc),
                                             replace=True))
        return _photon_channels

    @staticmethod
    def photon_timings(t, n_photon_hits, recoil_type, config, phase, 
                       channels=None, positions=None, e_dep=None,
                       n_photons_emitted=None, n_excitons=None, 
                       local_field=None, resource=None, nestpy_calc=None):
        """Calculate distribution of photon arrival timnigs
        :param t: 1d array of ints
        :param n_photon_hits: number of photon hits, 1d array of ints
        :param recoil_type: 1d array of ints
        :param config: dict wfsim config
        :param phase: str "liquid"
        :param channels: list of photon hit channels 
        :param positions: nx3 array of true XYZ positions from instruction
        :param e_dep: energy of the deposit, 1d float array
        :param n_photons_emitted: number of orignally emitted photons/quanta, 1d int array
        :param n_excitons: number of exctions in deposit, 1d int array
        :param local_field: local field in the point of the deposit, 1d array of floats
        :param resource: pointer to resources class of wfsim that contains s1 timing splines
        returns photon timing array"""
        _photon_timings = np.repeat(t, n_photon_hits)
        _n_hits_total = len(_photon_timings)

        if len(_photon_timings) == 0:
            return _photon_timings.astype(np.int64)

        if 'optical_propagation' in config['s1_model_type']:
            z_positions = np.repeat(positions[:, 2], n_photon_hits)
            _photon_timings += S1.optical_propagation(channels, z_positions, config,
                                                      spline=resource.s1_optical_propagation_spline).astype(np.int64)

        if 'simple' in config['s1_model_type']:
            # Simple S1 model enabled: use it for ER and NR.
            _photon_timings += np.random.exponential(config['s1_decay_time'], _n_hits_total).astype(np.int64)
            _photon_timings += np.random.normal(0, config['s1_decay_spread'], _n_hits_total).astype(np.int64)

        if 'nest' in config['s1_model_type'] or 'custom' in config['s1_model_type']:
            # Pulse model depends on recoil type
            counts_start = 0
            for i, counts in enumerate(n_photon_hits):

                if 'custom' in config['s1_model_type']:
                    for k in vars(NestId):
                        if k.startswith('_'):
                            continue
                        if recoil_type[i] in getattr(NestId, k):
                            str_recoil_type = k
                    try:
                        _photon_timings[counts_start: counts_start + counts] += \
                            getattr(S1, str_recoil_type.lower())(
                            size=counts,
                            config=config,
                            phase=phase).astype(np.int64)
                    except AttributeError:
                        raise AttributeError(f"Recoil type must be ER, NR, alpha or LED, "
                                             f"not {recoil_type}. Check nest ids")

                if 'nest' in config['s1_model_type']:
                    scint_time = nestpy_calc.GetPhotonTimes(
                        nestpy.INTERACTION_TYPE(recoil_type[i]),
                        n_photons_emitted[i],
                        n_excitons[i],
                        local_field[i],
                        e_dep[i])

                    scint_time = np.clip(scint_time, 0, config.get('maximum_recombination_time', 10000))
                    _photon_timings[counts_start: counts_start + counts] += np.array(scint_time[:counts], np.int64)

                counts_start += counts

        return _photon_timings

    @staticmethod
    def optical_propagation(channels, z_positions, config, spline):
        """Function gettting times from s1 timing splines:

        :param channels: The channels of all s1 photon
        :param z_positions: The Z positions of all s1 photon
        :param config: current configuration of wfsim
        :param spline: pointer to s1 optical propagation splines from resources
        """
        assert len(z_positions) == len(channels), 'Give each photon a z position'

        prop_time = np.zeros_like(channels)
        z_rand = np.array([z_positions, np.random.rand(len(channels))]).T

        is_top = channels < config['n_top_pmts']
        prop_time[is_top] = spline(z_rand[is_top], map_name='top')

        is_bottom = channels >= config['n_top_pmts']
        prop_time[is_bottom] = spline(z_rand[is_bottom], map_name='bottom')

        return prop_time

    @staticmethod
    def alpha(size, config, phase):
        """  Calculate S1 photon timings for an alpha decay. Neglible recombination time, not validated
        :param size: 1d array of ints, number of photons
        :param config: dict wfsim config
        :param phase: str "liquid"
        
        return 1d array of photon timings"""
        return Pulse.singlet_triplet_delays(size, config['s1_ER_alpha_singlet_fraction'], config, phase)

    @staticmethod
    def led(size, config, **kwargs):
        """  distribute photons uniformly within the LED pulse length, not validated
        :param size: 1d array of ints, number of photons
        :param config: dict wfsim config

        return 1d array of photon timings"""
        return np.random.uniform(0, config['led_pulse_length'], size)

    @staticmethod
    def er(size, config, phase):
        """Complex ER model, not validated
        :param size: 1d array of ints, number of photons
        :param config: dict wfsim config
        :param phase: str "liquid"
        return 1d array of photon timings
        """

        # How many of these are primary excimers? Others arise through recombination.
        # This config is not set for the nT fax config todo
        config.setdefault('liquid_density', 1.872452802978054e+30)
        density = config['liquid_density'] / (units.g / units.cm ** 3)
        excfrac = 0.4 - 0.11131 * density - 0.0026651 * density ** 2    # primary / secondary excimers
        excfrac = 1 / (1 + excfrac)                                     # primary / all excimers
        # primary / all excimers that produce a photon:
        excfrac /= 1 - (1 - excfrac) * (1 - config['s1_ER_recombination_fraction'])
        config['s1_ER_primary_excimer_fraction'] = excfrac
        log.debug('Inferred s1_ER_primary_excimer_fraction %s' % excfrac)

        # Recombination time from NEST 2014
        # 3.5 seems fishy, they fit an exponential to data, but in the code they use a non-exponential distribution...
        efield = (config['drift_field'] / (units.V / units.cm))
        reco_time = 3.5 / \
            0.18 * (1 / 20 + 0.41) * np.exp(-0.009 * efield)
        config['s1_ER_recombination_time'] = reco_time
        log.debug('Inferred s1_ER_recombination_time %s' % reco_time)

        timings = np.random.choice([0, reco_time], size, replace=True,
                                   p=[excfrac, 1 - excfrac])
        primary = timings == 0
        size_primary = len(timings[primary])

        timings[primary] += Pulse.singlet_triplet_delays(
            size_primary, config['s1_ER_primary_singlet_fraction'], config, phase)

        # Correct for the recombination time
        # For the non-exponential distribution: see Kubota 1979, solve eqn 2 for n/n0.
        # Alternatively, see Nest V098 source code G4S1Light.cc line 948
        timings[~primary] *= 1 / (-1 + 1 / np.random.uniform(0, 1, size - size_primary))
        # Update max recombine time in the nT fax config
        config['maximum_recombination_time'] = 1000
        timings[~primary] = np.clip(timings[~primary], 0, config['maximum_recombination_time'])
        timings[~primary] += Pulse.singlet_triplet_delays(
            size - size_primary, config['s1_ER_secondary_singlet_fraction'], config, phase)

        return timings

    @staticmethod
    def nr(size, config, phase):
        """NR model model, not validated
        :param size: 1d array of ints, number of photons
        :param config: dict wfsim config
        :param phase: str "liquid"
        return 1d array of photon timings
        """
        return Pulse.singlet_triplet_delays(size, config['s1_NR_singlet_fraction'], config, phase)
