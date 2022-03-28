import logging
from numba import njit
import numpy as np
from scipy.stats import skewnorm
from scipy import interpolate
from strax import exporter
from .pulse import Pulse
from .. import units
from ..load_resource import DummyMap

export, __all__ = exporter()
logging.basicConfig(handlers=[logging.StreamHandler()])
log = logging.getLogger('wfsim.core')
log.setLevel('WARNING')


@export
class S2(Pulse):
    """
    Given temporal inputs as well as number of electrons
    Random generate photon timing and channel distribution.
    """

    def __init__(self, config):
        super().__init__(config)

        self.phase = 'gas'  # To distinguish singlet/triplet time delay.
        self.luminescence_switch_threshold = 100  # When to use simplified model (NOT IN USE)

    @staticmethod
    def inverse_field_distortion_correction(x, y, z, resource):
        """For 1T the pattern map is a data driven one so we need to reverse engineer field distortion correction
        into the simulated positions
        :param x: 1d array of float
        :param y: 1d array of float
        :param z: 1d array of float
        :param resource: instance of resource class
        returns z: 1d array, postions 2d array 
        """
        positions = np.array([x, y, z]).T
        for i_iter in range(6):  # 6 iterations seems to work
            dr = resource.fdc_3d(positions)
            if i_iter > 0:
                dr = 0.5 * dr + 0.5 * dr_pre  # Average between iter
            dr_pre = dr

            r_obs = np.sqrt(x**2 + y**2) - dr
            x_obs = x * r_obs / (r_obs + dr)
            y_obs = y * r_obs / (r_obs + dr)
            z_obs = - np.sqrt(z**2 + dr**2)
            positions = np.array([x_obs, y_obs, z_obs]).T

        positions = np.array([x_obs, y_obs]).T 
        return z_obs, positions
    
    @staticmethod
    def field_distortion_comsol(x, y, z, resource):
        """Field distortion from the COMSOL simulation for the given electrode configuration:
        :param x: 1d array of float
        :param y: 1d array of float
        :param z: 1d array of float
        :param resource: instance of resource class
        returns z: 1d array, postions 2d array 
        """
        positions = np.array([np.sqrt(x**2 + y**2), z]).T
        theta = np.arctan2(y, x)
        r_obs = resource.fd_comsol(positions, map_name='r_distortion_map')
        x_obs = r_obs * np.cos(theta)
        y_obs = r_obs * np.sin(theta)
        
        positions = np.array([x_obs, y_obs]).T 
        return z, positions

    def __call__(self, instruction):
        if len(instruction.shape) < 1:
            # shape of recarr is a bit strange
            instruction = np.array([instruction])

        _, _, t, x, y, z, n_electron, recoil_type, *rest = [
            np.array(v).reshape(-1) for v in zip(*instruction)]
        
        # Reverse engineering FDC
        if self.config['field_distortion_model'] == 'inverse_fdc':
            z_obs, positions = self.inverse_field_distortion_correction(x, y, z, resource=self.resource)
        # Reverse engineering FDC
        elif self.config['field_distortion_model'] == 'comsol':
            z_obs, positions = self.field_distortion_comsol(x, y, z, resource=self.resource)
        else:
            z_obs, positions = z, np.array([x, y]).T

        n_electron = self.get_electron_yield(n_electron=n_electron,
                                             xy_int= np.array([x, y]).T, # maps are in R_true, so orginal position should be here
                                             z_int=z, # maps are in Z_true, so orginal position should be here
                                             config=self.config,
                                             resource=self.resource)

        sc_gain = self.get_s2_light_yield(positions=positions,
                                          config=self.config,
                                          resource=self.resource)


        n_photons_per_xy, n_photons_per_ele, self._electron_timings = self.get_n_photons(t=t,
                                                                                         n_electron=n_electron,
                                                                                         z_obs=z_obs,
                                                                                         positions=positions,
                                                                                         sc_gain=sc_gain, 
                                                                                         config=self.config,
                                                                                         resource=self.resource)

        self._instruction = np.repeat(np.arange(len(n_electron)), n_photons_per_xy)
        self._photon_channels = self.photon_channels(n_electron=n_electron,
                                                     z_obs=z_obs,
                                                     positions=positions,
                                                     _instruction=self._instruction,
                                                     config=self.config,
                                                     resource=self.resource)

        # Second generate photon timing
        self._photon_timings = self.photon_timings(positions=positions, 
                                                   n_photons_per_xy=n_photons_per_xy,
                                                   _electron_timings=self._electron_timings,
                                                   n_photons_per_ele=n_photons_per_ele,
                                                   _photon_channels=self._photon_channels,
                                                   phase=self.phase,
                                                   config=self.config,
                                                   resource=self.resource,)

        # Sorting times according to the channel, as non-explicit sorting
        # is performed later and this breaks timing of individual channels/arrays
        sortind = np.argsort(self._photon_channels)

        self._photon_channels = self._photon_channels[sortind]
        self._photon_timings = self._photon_timings[sortind]

        super().__call__()

    @staticmethod
    def get_avg_drift_velocity(z, xy, config, resource):
        """Calculate s2 drift time mean and spread

        :param positions: 1d array of z (floats)
        :param xy: 2d array of xy positions (floats)
        :param config: dict with wfsim config
        :param resource: instance of the resource class

        returns array of floats corresponding to average drift velocities from given point to the gate
        """
        drift_v_LXe=None
        if config['enable_field_dependencies']['drift_speed_map']:
            drift_v_LXe = resource.field_dependencies_map(z, xy, map_name='drift_speed_map')  # mm/µs
            drift_v_LXe *= 1e-4  # cm/ns
            drift_v_LXe *= resource.drift_velocity_scaling
        else:
            drift_v_LXe=config['drift_velocity_liquid']
        return(drift_v_LXe)

    @staticmethod
    def get_s2_drift_time_params(z_int, xy_int, config, resource):
        """Calculate s2 drift time mean and spread

        :param z_int: 1d array of true z (floats) 
        :param xy_int: 2d array of true xy positions (floats)
        :param config: dict with wfsim config
        :param resource: instance of the resource class

        returns two arrays of floats (mean drift time, drift time spread) 
        """
        drift_velocity_liquid = S2.get_avg_drift_velocity(z_int, xy_int, config, resource)
        if config['enable_field_dependencies']['diffusion_longitudinal_map']:
            diffusion_constant_longitudinal = resource.field_dependencies_map(z_int, xy_int, map_name='diffusion_longitudinal_map')  # cm²/s
            diffusion_constant_longitudinal *= 1e-9  # cm²/ns
        else:
            diffusion_constant_longitudinal = config['diffusion_constant_longitudinal']

        drift_time_mean = - z_int / \
            drift_velocity_liquid + config['drift_time_gate']
        drift_time_mean = np.clip(drift_time_mean, 0, np.inf)
        drift_time_spread = np.sqrt(2 * diffusion_constant_longitudinal * drift_time_mean)
        drift_time_spread /= drift_velocity_liquid
        return drift_time_mean, drift_time_spread

    @staticmethod
    def get_s2_light_yield(positions, config, resource):
        """Calculate s2 light yield...
        
        :param positions: 2d array of positions (floats)
        :param config: dict with wfsim config
        :param resource: instance of the resource class
        
        returns array of floats (mean expectation) 
        """
        sc_gain = resource.s2_correction_map(positions)
        # depending on if you use the data driven or mc pattern map for light yield for S2 
        # the shape of n_photon_hits will change. Mc needs a squeeze
        if len(sc_gain.shape) != 1:
            sc_gain=np.squeeze(sc_gain, axis=-1)

        # sc gain should has the unit of pe / electron, here we divide 1 + dpe to get nphoton / electron
        sc_gain /= 1 + config['p_double_pe_emision']
        sc_gain *= config['s2_secondary_sc_gain']

        # data driven map contains nan, will be set to 0 here
        sc_gain[np.isnan(sc_gain)] = 0

        return sc_gain

    @staticmethod
    def get_electron_yield(n_electron, xy_int, z_int, config, resource):
        """Drift electrons up to the gas interface and absorb them

        :param n_electron: 1d array with ints as number of electrons
        :param xy_int: 2d array of xy interaction positions (floats)
        :param z_int: 1d array of floats with the z interaction positions (floats)
        :param config: dict with wfsim config

        returns 1d array ints with number of electrons
        """
        # Average drift time of the electrons
        drift_time_mean, drift_time_spread = S2.get_s2_drift_time_params(z_int, xy_int, config, resource)

        # Absorb electrons during the drift
        electron_lifetime_correction = np.exp(- 1 * drift_time_mean /
                                              config['electron_lifetime_liquid'])
        cy = config['electron_extraction_yield'] * electron_lifetime_correction

        # Remove electrons in insensitive volume
        if config['enable_field_dependencies']['survival_probability_map']:
            p_surv = resource.field_dependencies_map(z_int, xy_int, map_name='survival_probability_map')
            if np.any(p_surv<0) or np.any(p_surv>1):
                # FIXME: this is necessary due to map artefacts, such as negative or values >1
                p_surv=np.clip(p_surv, a_min = 0, a_max = 1)
            cy *= p_surv
        n_electron = np.random.binomial(n=n_electron, p=cy)
        return n_electron

    @staticmethod
    @njit
    def electron_timings(t, n_electron, drift_time_mean, drift_time_spread, sc_gain, timings, gains,
                         electron_trapping_time):
        """Calculate arrival times of the electrons. Data is written to the timings and gains arrays
        :param t: 1d array of ints
        :param n_electron:1 d array of ints
        :param drift_time_mean: 1d array of floats
        :param drift_time_spread: 1d array of floats
        :param sc_gain: secondary scintillation gain       
        :param timings: empty array with length sum(n_electron)
        :param gains: empty array with length sum(n_electron)
        :param electron_trapping_time: configuration values
        """
        assert len(timings) == np.sum(n_electron)
        assert len(gains) == np.sum(n_electron)
        assert len(sc_gain) == len(t)

        i_electron = 0
        for i in np.arange(len(t)):
            # Calculate electron arrival times in the ELR region
            for _ in np.arange(n_electron[i]):
                _timing = np.random.exponential(electron_trapping_time)
                _timing += np.random.normal(drift_time_mean[i], drift_time_spread[i])
                timings[i_electron] = t[i] + int(_timing)

                # add manual fluctuation to sc gain
                gains[i_electron] = sc_gain[i]
                i_electron += 1

    @staticmethod
    def get_n_photons(t, n_electron, z_obs, positions, sc_gain, config, resource):
        """Generates photon timings for S2s.
        Returns a list of photon timings and instructions repeated for original electron
        
        :param t: 1d int array time of s2
        :param n_electron: 1d float array number of electrons to simulate
        :param z_obs: float array. Z positions of s2
        :param positions: 2d float array, xy positions of s2
        :param sc_gain: float, secondary s2 gain
        :param config: dict of the wfsim config
        :param resource: instance of the resource class """
        # Get electron timings
        drift_time_mean, drift_time_spread = S2.get_s2_drift_time_params(z_obs, positions, config, resource)
        _electron_timings = np.zeros(np.sum(n_electron), np.int64)
        _electron_gains = np.zeros(np.sum(n_electron), np.float64)
        S2.electron_timings(t, n_electron, drift_time_mean, drift_time_spread, sc_gain,
                            _electron_timings, _electron_gains, config['electron_trapping_time'])

        # Populate with photons per e/ per position
        n_photons_per_ele = np.random.poisson(_electron_gains)
        n_photons_per_ele += np.random.normal(0, config.get('s2_gain_spread', 0), len(n_photons_per_ele)).astype(np.int64)
        n_photons_per_ele[n_photons_per_ele < 0] = 0
        #
        n_photons_per_xy = np.cumsum(np.pad(n_photons_per_ele, [1, 0]))[np.cumsum(n_electron)]
        n_photons_per_xy = np.diff(np.pad(n_photons_per_xy, [1, 0]))

        return n_photons_per_xy, n_photons_per_ele, _electron_timings

    @staticmethod
    @njit
    def _luminescence_timings_simple(n, dG, E0, r, dr, rr, alpha, uE, p, n_photons):
        """
        Luminescence time distribution computation, calculates emission timings of photons from the excited electrons
        return 1d nested array with ints
        """
        emission_time = np.zeros(np.sum(n_photons), np.int64)

        ci = 0
        for i in range(n):
            npho = n_photons[i]
            dt = dr / (alpha * E0[i] * rr)
            dy = E0[i] * rr / uE - 0.8 * p  # arXiv:physics/0702142
            avgt = np.sum(np.cumsum(dt) * dy) / np.sum(dy)

            j = np.argmax(r <= dG[i])
            t = np.cumsum(dt[j:]) - avgt
            y = np.cumsum(dy[j:])

            probabilities = np.random.rand(npho)
            emission_time[ci:ci+npho] = np.interp(probabilities, y / y[-1], t).astype(np.int64)
            ci += npho

        return emission_time

    @staticmethod
    def luminescence_timings_simple(xy, n_photons, config, resource):
        """
        Luminescence time distribution computation according to simple s2 model (many many many single electrons)
        :param xy: 1d array with positions
        :param n_photons: 1d array with ints for number of xy positions
        :param config: dict wfsim config
        :param resource: instance of wfsim resource
        returns _luminescence_timings_simple
        """
        assert len(n_photons) == len(xy), 'Input number of n_photons should have same length as positions'

        number_density_gas = config['pressure'] / \
            (units.boltzmannConstant * config['temperature'])
        alpha = config['gas_drift_velocity_slope'] / number_density_gas
        uE = units.kV / units.cm
        pressure = config['pressure'] / units.bar

        if config.get('enable_gas_gap_warping', True):
            dG = resource.gas_gap_length(xy)
        else:
            dG = np.ones(len(xy)) * config['elr_gas_gap_length']
        rA = config['anode_field_domination_distance']
        rW = config['anode_wire_radius']
        dL = config['gate_to_anode_distance'] - dG

        VG = config['anode_voltage'] / (1 + dL / dG / config['lxe_dielectric_constant'])
        E0 = VG / ((dG - rA) / rA + np.log(rA / rW))  # V / cm

        dr = 0.0001  # cm
        r = np.arange(np.max(dG), rW, -dr)
        rr = np.clip(1 / r, 1 / rA, 1 / rW)

        return S2._luminescence_timings_simple(len(xy), dG, E0, 
                                               r, dr, rr, alpha, uE,
                                               pressure, n_photons)

    @staticmethod
    def luminescence_timings_garfield(xy, n_photons, config, resource, confine_position=None):
        """
        Luminescence time distribution computation according to garfield scintillation maps
        :param xy: 1d array with positions
        :param n_photons: 1d array with ints for number of xy positions
        :param config: dict wfsim config
        :param resource: instance of wfsim resource
        :param confine_position: if float, confine extraction region +/- this position around anode wires

        returns 2d array with ints for photon timings of input param 'shape'
        """
        assert 's2_luminescence' in resource.__dict__, 's2_luminescence model not found'
        assert len(n_photons) == len(xy), 'Input number of n_electron should have same length as positions'
        assert len(resource.s2_luminescence['t'].shape) == 2, 'Timing data is expected to have D2'

        if type(confine_position)==float:
            distance = np.random.uniform(-confine_position, confine_position, len(xy))
        else:
            tilt = config.get('anode_xaxis_angle', np.pi / 4)
            pitch = config.get('anode_pitch', 0.5)
            rotation_mat = np.array(((np.cos(tilt), -np.sin(tilt)), (np.sin(tilt), np.cos(tilt))))
            jagged = lambda relative_y: (relative_y + pitch / 2) % pitch - pitch / 2
            distance = jagged(np.matmul(xy, rotation_mat)[:, 1])  # shortest distance from any wire

        index_row = [np.argmin(np.abs(d - resource.s2_luminescence['x'])) for d in distance]
        index_row = np.repeat(index_row, n_photons).astype(np.int64)
        index_col = np.random.randint(0, resource.s2_luminescence['t'].shape[1], np.sum(n_photons), np.int64)
        
        avgt = np.average(resource.s2_luminescence['t']).astype(int)
        return resource.s2_luminescence['t'][index_row, index_col].astype(np.int64) - avgt

    @staticmethod
    def optical_propagation(channels, config, spline):
        """Function gettting times from s2 timing splines:
        :param channels: The channels of all s2 photon
        :param config: current configuration of wfsim
        :param spline: pointer to s2 optical propagation splines from resources
        """
        prop_time = np.zeros_like(channels)
        u_rand = np.random.rand(len(channels))[:, None]

        is_top = channels < config['n_top_pmts']
        prop_time[is_top] = spline(u_rand[is_top], map_name='top')

        is_bottom = channels >= config['n_top_pmts']
        prop_time[is_bottom] = spline(u_rand[is_bottom], map_name='bottom')

        return prop_time.astype(np.int64)

    @staticmethod
    def photon_timings(positions, n_photons_per_xy, _electron_timings, n_photons_per_ele, _photon_channels, phase, config, resource):
        """Get photon times and add delays based on models

        :param positions: 2d float array, xy positions of s2
        :param n_photons_per_xy: number of photons for each xy position
        :param _electron_timings: electron timings
        :param n_photons_per_ele: number of photons for each electron
        :param _photon_channels: channel of each photon
        :param config: dict of the wfsim config
        :param phase: gas
        :param resource: instance of the resource class
        """

        # Luminescence Timings
        if config['s2_luminescence_model']=='simple':
            _photon_timings = S2.luminescence_timings_simple(positions, n_photons_per_xy,
                                                             config=config,
                                                             resource=resource)
        elif config['s2_luminescence_model']=='garfield':
            # check to see if extraction region in Garfield needs to be confined
            confine_position=None
            if 's2_garfield_confine_position' in config:
                if config['s2_garfield_confine_position'] > 0.0:
                    confine_position=config['s2_garfield_confine_position']
            _photon_timings = S2.luminescence_timings_garfield(positions, n_photons_per_xy,
                                                               config=config,
                                                               resource=resource,
                                                               confine_position=confine_position)
        else:
            raise KeyError(f"{config['s2_luminescence_model']} is not valid! Use 'simple' or 'garfield'")

        # Emission Delay
        _photon_timings += Pulse.singlet_triplet_delays(len(_photon_timings), config['singlet_fraction_gas'], config, phase)

        # Optical Propagation Delay
        if "optical_propagation" in config['s2_time_model']:
            # optical propagation splitting top and bottom
            _photon_timings += S2.optical_propagation(_photon_channels, config, resource.s2_optical_propagation_spline)
        elif "zero_delay" in config['s2_time_model']:
            # no optical propagation delay
            _photon_timings += np.zeros_like(_photon_timings, dtype=np.int64)
        elif "s2_time_spread around zero" in config['s2_time_model']:
            # simple/existing delay
            _photon_timings += np.random.normal(0, config['s2_time_spread'], len(_photon_timings)).astype(np.int64)
        else:
            raise KeyError(f"{config['s2_time_model']} is not in any of the valid s2 time models")

        # repeat for n photons per electron # Should this be before adding delays?
        _photon_timings += np.repeat(_electron_timings, n_photons_per_ele)

        return _photon_timings

    @staticmethod
    def s2_pattern_map_diffuse(n_electron, z, xy, config, resource):
        """Returns an array of pattern of shape [n interaction, n PMTs]
        pattern of each interaction is an average of n_electron patterns evaluated at
        diffused position near xy. The diffused positions sample from 2d symmetric gaussian
        with spread scale with sqrt of drift time.

        :param n_electron: a 1d int array
        :param z: a 1d float array
        :param xy: a 2d float array of shape [n interaction, 2]
        :param config: dict of the wfsim config
        :param resource: instance of the resource class
        """
        assert all(z < 0), 'All S2 in liquid should have z < 0'
        drift_velocity_liquid=S2.get_avg_drift_velocity(z, xy, config, resource)

        if config['enable_field_dependencies']['diffusion_transverse_map']:
            diffusion_constant_radial = resource.field_dependencies_map(z, xy, map_name='diffusion_radial_map')  # cm²/s
            diffusion_constant_azimuthal = resource.field_dependencies_map(z, xy, map_name='diffusion_azimuthal_map') # cm²/s
            diffusion_constant_radial *= 1e-9  # cm²/ns
            diffusion_constant_azimuthal *= 1e-9  # cm²/ns
        else:
            diffusion_constant_transverse = getattr(config, 'diffusion_constant_transverse', 0)
            diffusion_constant_radial = diffusion_constant_transverse
            diffusion_constant_azimuthal = diffusion_constant_transverse

        drift_time_mean = - z / drift_velocity_liquid
        hdiff_stdev_radial = np.sqrt(2 * diffusion_constant_radial * drift_time_mean)
        hdiff_stdev_azimuthal = np.sqrt(2 * diffusion_constant_azimuthal * drift_time_mean)
        
        hdiff_radial = np.random.normal(0, 1, np.sum(n_electron)) * np.repeat(hdiff_stdev_radial, n_electron, axis=0)
        hdiff_azimuthal = np.random.normal(0, 1, np.sum(n_electron)) * np.repeat(hdiff_stdev_azimuthal, n_electron, axis=0)
        hdiff = np.column_stack([hdiff_radial, hdiff_azimuthal])
        theta = np.arctan2(xy[:,1], xy[:,0])
        matrix = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]]).T
        hdiff = np.vstack([(matrix[i] @ np.split(hdiff, np.cumsum(n_electron))[:-1][i].T).T for i in range(len(matrix))])
        # Should we also output this xy position in truth?
        xy_multi = np.repeat(xy, n_electron, axis=0) + hdiff  # One entry xy per electron
        # Remove points outside tpc, and the pattern will be the average inside tpc
        # Should be done naturally with the s2 pattern map, however, there's some bug there so we apply this hard cut
        mask = np.sum(xy_multi ** 2, axis=1) <= config['tpc_radius'] ** 2

        if isinstance(resource.s2_pattern_map, DummyMap):
            output_dim = resource.s2_pattern_map.shape[-1]
        else:
            output_dim = resource.s2_pattern_map.data['map'].shape[-1]
        pattern = np.zeros((len(n_electron), output_dim))
        n0 = 0
        # Average over electrons for each s2
        for ix, ne in enumerate(n_electron):
            s = slice(n0, n0+ne)
            pattern[ix, :] = np.average(resource.s2_pattern_map(xy_multi[s][mask[s]]), axis=0)
            n0 += ne

        return pattern

    @staticmethod
    def photon_channels(n_electron, z_obs, positions, _instruction, config, resource):
        """Set the _photon_channels property list of length same as _photon_timings
        :param n_electron: a 1d int array
        :param z_obs: a 1d float array
        :param positions: a 2d float array of shape [n interaction, 2] for the xy coordinate
        :param _instruction: array of instructions with dtype wfsim.instructions_dtype
        :param config: dict wfsim config
        :param resource: instance of resource class
        """
        if len(_instruction) == 0:
            _photon_channels = np.zeros(0, dtype=np.int64)
            return _photon_channels

        aft = config['s2_mean_area_fraction_top']
        aft_sigma = config.get('s2_aft_sigma', 0.0118)
        aft_skewness = config.get('s2_aft_skewness', -1.433)

        channels = np.arange(config['n_tpc_pmts']).astype(np.int64)
        top_index = np.arange(config['n_top_pmts'])
        bottom_index = np.array(config['channels_bottom'])

        if config.get('diffusion_constant_transverse', 0) > 0:
            pattern = S2.s2_pattern_map_diffuse(n_electron, z_obs, positions, config, resource)  # [position, pmt]
        else:
            pattern = resource.s2_pattern_map(positions)  # [position, pmt]

        if pattern.shape[1] - 1 not in bottom_index:
            pattern = np.pad(pattern, [[0, 0], [0, len(bottom_index)]], 
                             'constant', constant_values=1)

        # Remove turned off pmts
        pattern[:, np.in1d(channels, config['turned_off_pmts'])] = 0

        sum_pat = np.sum(pattern, axis=1).reshape(-1, 1)
        pattern = np.divide(pattern, sum_pat, out=np.zeros_like(pattern), where=sum_pat != 0)

        assert pattern.shape[0] == len(positions)
        assert pattern.shape[1] == len(channels)

        _buffer_photon_channels = []
        # Randomly assign to channel given probability of each channel
        for unique_i, count in zip(*np.unique(_instruction, return_counts=True)):
            pat = pattern[unique_i]  # [pmt]

            if aft > 0:  # Redistribute pattern with user specified aft
                _aft = aft * (1 + skewnorm.rvs(loc=0,
                                               scale=aft_sigma,
                                               a=aft_skewness))
                _aft = np.clip(_aft, 0, 1)
                pat[top_index] = pat[top_index] / pat[top_index].sum() * _aft
                pat[bottom_index] = pat[bottom_index] / pat[bottom_index].sum() * (1 - _aft)

            if np.isnan(pat).sum() > 0:  # Pattern map return zeros
                _photon_channels = np.array([-1] * count)
            else:
                _photon_channels = np.random.choice(
                    channels,
                    size=count,
                    p=pat,
                    replace=True)

            _buffer_photon_channels.append(_photon_channels)

        _photon_channels = np.concatenate(_buffer_photon_channels)
        return _photon_channels
