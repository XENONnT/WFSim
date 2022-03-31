from copy import deepcopy
import logging
import numpy as np
import os.path as osp
import strax
from strax import exporter
import straxen


NT_AUX_INSTALLED = False
STRAX_AUX_INSTALLED = False

try:
    import ntauxfiles
    NT_AUX_INSTALLED = True
except (ModuleNotFoundError, ImportError):
    pass
try:
    import straxauxfiles
    STRAX_AUX_INSTALLED = True
except (ModuleNotFoundError, ImportError):
    pass


export, __all__ = exporter()
logging.basicConfig(handlers=[logging.StreamHandler()])
log = logging.getLogger('wfsim.resource')
log.setLevel('WARNING')


_cached_configs = dict()


@export
def load_config(config):
    """Create a Resource instance from the configuration

    Uses a cache to avoid re-creating instances from the same config
    """
    h = strax.deterministic_hash(Resource.config_to_file(config))
    if h in _cached_configs:
        return _cached_configs[h]
    result = Resource(config)
    _cached_configs[h] = result
    log.debug(f'Caching config file set {h}')
    return result


@export
class Resource:
    """
    Get the configs needed for running WFSim. Configs can be obtained in
        two ways:
        1. Get it directly from the mongo database. This only needs the
            name of the file.
        2. Load it with straxen get_resource, this can either:
            Download from a public repository
            Read from local cache
            Download from a private repository if credentials 
            are properly setup
    """
    @staticmethod
    def config_to_file(config):
        """
        Find and complete all file paths

        returns dictionary mapping item name to path
        """
        if config is None:
            config = dict()

        files = {}
        if config['detector'] == 'XENON1T':
            files.update({
                'photon_area_distribution': 'XENON1T_spe_distributions.csv',
                's1_lce_correction_map': 'XENON1T_s1_xyz_ly_kr83m_SR1_pax-680_fdc-3d_v0.json',
                's1_pattern_map': 'XENON1T_s1_xyz_patterns_interp_corrected_MCv2.1.0.json.gz',
                's2_correction_map': 'XENON1T_s2_xy_ly_SR1_v2.2.json',
                's2_pattern_map': 'XENON1T_s2_xy_patterns_top_corrected_MCv2.1.0.json.gz',
                'photon_ap_cdfs': 'x1t_pmt_afterpulse_config.pkl.gz',
                'fdc_3d': 'XENON1T_FDC_SR1_data_driven_time_dependent_3d_correction_tf_nn_part1_v1.json.gz',
                'ele_ap_pdfs': 'x1t_se_afterpulse_delaytime.pkl.gz',
                'noise_file': 'x1t_noise_170203_0850_00_small.npz'
            })

        elif config['detector'] == 'XENONnT':
            files.update({
                'photon_area_distribution': 'XENONnT_spe_distributions_20210305.csv',
                's1_pattern_map': 'XENONnT_s1_xyz_patterns_LCE_corrected_qes_MCva43fa9b_wires.pkl',
                's1_lce_correction_map': 'XENONnT_s1_xyz_LCE_corrected_qes_MCva43fa9b_wires.json.gz',
                's2_pattern_map': 'XENONnT_s2_xy_patterns_LCE_corrected_qes_MCva43fa9b_wires.pkl',
                's2_correction_map': 'XENONnT_s2_xy_correction_corrected_qes_MCva43fa9b_wires.json.gz',
                'photon_ap_cdfs': 'XENONnT_pmt_afterpulse_config_012605.json.gz',
                's2_luminescence': 'XENONnT_GARFIELD_B1d5n_C30n_G1n_A6d5p_T1d5n_PMTs1d5n_FSR0d95n.npz',
                'gas_gap_map': 'gas_gap_warping_map_January_2021.pkl',
                'ele_ap_pdfs': 'x1t_se_afterpulse_delaytime.pkl.gz',
                'noise_file': 'x1t_noise_170203_0850_00_small.npz',
                'fdc_3d': 'XnT_3D_FDC_xyt_dummy_all_zeros_v0.1.json.gz',
                'field_dependencies_map': '',
                's1_time_spline': 'XENONnT_s1_proponly_va43fa9b_wires_20200625.json.gz',
                's2_time_spline': '',
             })

        elif config['detector'] == 'XENONnT_neutron_veto':
            files.update({
                'photon_area_distribution': 'XENONnT_spe_distributions_nveto_013071.csv',
                'nv_pmt_qe': 'nveto_pmt_qe.json',
                'noise_file': 'xnt_noise_nveto_014901.npz'
            })

        else:
            raise ValueError(f"Unsupported detector {config['detector']}")

        # Allowing user to replace default with specified files
        for k in set(config).intersection(files):
            files[k] = config[k]

        commit = 'master'  # Replace this by a commit hash if you feel solid and responsible
        if config.get('url_base', False):
            files['url_base'] = config['url_base']
        elif config['detector'] == "XENON1T":
            files['url_base'] = f'https://raw.githubusercontent.com/XENONnT/strax_auxiliary_files/{commit}/sim_files'
        else:
            files['url_base'] = f'https://raw.githubusercontent.com/XENONnT/private_nt_aux_files/{commit}/sim_files'

        return files

    @staticmethod
    def get_file_path(base, fname):
        """Find the full path to the resource file
        Try 4 methods in the following order
        1. The base is not url, return base + name
        2. If ntauxfiles (straxauxfiles) is installed, return will be package dir + name
           pip install won't work, try python setup.py in the packages
        3. Download the latest version using straxen mongo downloader from database,
           return the cached file path + md5
        4. Download using straxen get_resource from the url (github raw)
           simply return base + name
            Be careful with the forth options, straxen creates
            cache files that might not be updated with the latest github commit.

        """
        if not fname:
            log.warning(f"A file has value False, assuming this is intentional.")
            return

        if fname.startswith('/'):
            log.warning(f"Using local file {fname} for a resource. "
                        f"Do not set this as a default or TravisCI tests will break")
            return fname
        
        if base.startswith('/'):
            log.warning(f"Using local folder {base} for all resources. "
                        f"Do not set this as a default or TravisCI tests will break")
            return osp.join(base, fname)

        if NT_AUX_INSTALLED:
            # You might want to use this, for example if you are a developer
            if fname in ntauxfiles.list_private_files():
                log.warning(f"Using the private repo to load {fname} locally")
                fpath = ntauxfiles.get_abspath(fname)
                log.info(f"Loading {fname} is successfully from {fpath}")
                return fpath

        if STRAX_AUX_INSTALLED:
            if fname in straxauxfiles.list_aux_files():
                log.warning(f"Using the public repo to load {fname} locally")
                fpath = straxauxfiles.get_abspath(fname)
                log.info(f"Loading {fname} is successfully from {fpath}")
                return fpath

        try:
            # https://straxen.readthedocs.io/en/latest/config_storage.html
            # downloading-xenonnt-files-from-the-database  # noqa

            # we need to add the straxen.MongoDownloader() in this
            # try: except NameError: logic because the NameError
            # gets raised if we don't have access to utilix.
            downloader = straxen.MongoDownloader()
            # FileNotFoundError, ValueErrors can be raised if we
            # cannot load the requested config
            fpath = downloader.download_single(fname)
            log.warning(f"Loading {fname} from mongo downloader")
            return fname  # Keep the name and let get_resource do its thing

        except (FileNotFoundError, ValueError, NameError, AttributeError):
            log.info(f"Mongo downloader not possible or does not have {fname}")

        # We cannot download the file from the database. We need to
        # try to get a placeholder file from a URL.
        furl = osp.join(base, fname)
        log.warning(f'{fname} did not download, trying download from {base}')
        return furl

    def __init__(self, config=None):
        files = self.config_to_file(config)
        log.debug('Getting\n' + '\n'.join([f'{k}: {v}' for k, v in files.items()]))

        for k, v in files.items():
            if isinstance(v, list):
                # It's a dummy map call, do nothing
                continue
            if k == 'url_base':
                continue
            if v == '':
                log.warning(f'{k} has no path so this config file is set to None')
                files[k] = None
                continue

            log.debug(f'Obtaining {k} from {v}')
            files[k] = self.get_file_path(files['url_base'], v)

        if config.get('detector', 'XENONnT') == 'XENON1T':
            self.s1_pattern_map = make_map(files['s1_pattern_map'], fmt='json.gz')
            self.s1_lce_correction_map = make_map(files['s1_lce_correction_map'], fmt='json')
            self.s2_correction_map = make_map(files['s2_correction_map'], fmt='json')
            self.s2_pattern_map = make_map(files['s2_pattern_map'], fmt='json.gz')
            self.fdc_3d = make_map(files['fdc_3d'], fmt='json.gz')

            # Gas gap warping map
            if config.get('enable_gas_gap_warping', False):
                self.gas_gap_length = make_map(["constant dummy", 0.25, [254, ]])

            # Photon After Pulses
            if config.get('enable_pmt_afterpulses', False):
                self.uniform_to_pmt_ap = straxen.get_resource(files['photon_ap_cdfs'], fmt='pkl.gz')

            # Electron After Pulses
            if config.get('enable_electron_afterpulses', False):
                self.uniform_to_ele_ap = straxen.get_resource(files['ele_ap_pdfs'], fmt='pkl.gz')

        elif config.get('detector', 'XENONnT') == 'XENONnT':
            pmt_mask = np.array(config['gains']) > 0  # Converted from to pe (from cmt by default)

            self.s1_pattern_map = make_map(files['s1_pattern_map'], fmt='pkl')
            self.s2_pattern_map = make_map(files['s2_pattern_map'], fmt='pkl')

            # if there is a (data driven!) map, load it. If not make it  from the pattern map
            if files['s1_lce_correction_map']:
                self.s1_lce_correction_map = make_map(files['s1_lce_correction_map'])
            else:
                lymap = deepcopy(self.s1_pattern_map)
                lymap.data['map'] = np.sum(lymap.data['map'][:][:][:], axis=3, keepdims=True, where=pmt_mask)
                lymap.__init__(lymap.data)
                self.s1_lce_correction_map = lymap

            # if there is a (data driven!) map, load it. If not make it  from the pattern map
            if files['s2_correction_map']:
                self.s2_correction_map = make_map(files['s2_correction_map'])
            else:
                s2cmap = deepcopy(self.s2_pattern_map)
                # Lower the LCE by removing contribution from dead PMTs
                s2cmap.data['map'] = np.sum(s2cmap.data['map'][:][:], axis=2, keepdims=True, where=pmt_mask)
                # Scale by median value
                s2cmap.data['map'] = s2cmap.data['map'] / np.median(s2cmap.data['map'][s2cmap.data['map'] > 0])
                s2cmap.__init__(s2cmap.data)
                self.s2_correction_map = s2cmap

            # Garfield luminescence timing samples
            # if config.get('s2_luminescence_model', False) == 'garfield':
            if 'garfield' in config.get('s2_luminescence_model', ''):
                s2_luminescence_map = straxen.get_resource(files['s2_luminescence'], fmt='npy_pickle')['arr_0']
                # Get directly the map for the simulated level
                liquid_level_available = np.unique(s2_luminescence_map['ll'])  # available levels (cm)
                liquid_level = config['gate_to_anode_distance'] - config['elr_gas_gap_length']  # cm
                liquid_level = min(liquid_level_available, key=lambda x: abs(x - liquid_level))
                self.s2_luminescence = s2_luminescence_map[s2_luminescence_map['ll'] == liquid_level]

            if config.get('field_distortion_model', "none") == "inverse_fdc":
                self.fdc_3d = make_map(files['fdc_3d'], fmt='json.gz')
                self.fdc_3d.scale_coordinates([1., 1., - config['drift_velocity_liquid']])

            if config.get('field_distortion_model', "none") == "comsol":
                self.fd_comsol = make_map(config['field_distortion_comsol_map'], fmt='json.gz', method='RectBivariateSpline')

            # Gas gap warping map
            if config.get('enable_gas_gap_warping', False):
                gas_gap_map = straxen.get_resource(files['gas_gap_map'], fmt='pkl')
                self.gas_gap_length = lambda positions: gas_gap_map.lookup(*positions.T)

            # Field dependencies 
            # This config entry a dictionary of 5 items
            if any(config['enable_field_dependencies'].values()):
                field_dependencies_map = make_map(files['field_dependencies_map'], fmt='json.gz', method='RectBivariateSpline')
                self.drift_velocity_scaling=1.0
                # calculating drift velocity scaling to match total drift time for R=0 between cathode and gate
                if "norm_drift_velocity" in config['enable_field_dependencies'].keys():
                    if config['enable_field_dependencies']['norm_drift_velocity']:
                        norm_dvel = field_dependencies_map(np.array([ [0], [- config['tpc_length']]]).T,
                                                           map_name='drift_speed_map')[0]
                        norm_dvel*=1e-4
                        self.drift_velocity_scaling=config['drift_velocity_liquid']/norm_dvel
                def rz_map(z, xy, **kwargs):
                    r = np.sqrt(xy[:, 0]**2 + xy[:, 1]**2)
                    return field_dependencies_map(np.array([r, z]).T, **kwargs)
                self.field_dependencies_map = rz_map

            # Photon After Pulses
            if config.get('enable_pmt_afterpulses', False):
                self.uniform_to_pmt_ap = straxen.get_resource(files['photon_ap_cdfs'], fmt='json.gz')

            # S1 photon timing splines
            if config.get('s1_time_spline', False):
                self.s1_optical_propagation_spline = make_map(files['s1_time_spline'], fmt='json.gz',
                                                              method='RegularGridInterpolator')

            # Electron After Pulses
            if config.get('enable_electron_afterpulses', False):
                self.uniform_to_ele_ap = straxen.get_resource(files['ele_ap_pdfs'], fmt='pkl.gz')

            # S2 photons timing optical propagation delays
            if config.get('s2_time_spline', False):
                self.s2_optical_propagation_spline = make_map(files['s2_time_spline'])

        elif config.get('detector', 'XENONnT') == 'XENONnT_neutron_veto':
            # Neutron veto PMT QE as function of wavelength
                self.nv_pmt_qe = straxen.get_resource(files['nv_pmt_qe'], fmt='json')

        # SPE area distributions
        self.photon_area_distribution = straxen.get_resource(files['photon_area_distribution'], fmt='csv')

        # Noise sample
        if config.get('enable_noise', False):
            self.noise_data = straxen.get_resource(files['noise_file'], fmt='npy')['arr_0']
            n_channels = len(self.noise_data[0])
            log.warning(f'Using noise data {files["noise_file"]} with {n_channels} channels for {config["detector"]}')

        log.debug(f'{self.__class__.__name__} fully initialized')


def make_map(map_file, fmt=None, method='WeightedNearestNeighbors'):
    """Fetch and make an instance of InterpolatingMap based on map_file
    Alternatively map_file can be a list of ["constant dummy", constant: int, shape: list]
    return an instance of  DummyMap"""

    if isinstance(map_file, list):
        assert map_file[0] == 'constant dummy', ('Alternative file input can only be '
                                                 '("constant dummy", constant: int, shape: list')
        return DummyMap(map_file[1], map_file[2])

    elif isinstance(map_file, str):
        if fmt is None:
            fmt = parse_extension(map_file)

        log.debug(f'Initialize map interpolator for file {map_file}')
        map_data = straxen.get_resource(map_file, fmt=fmt)
        return straxen.InterpolatingMap(map_data, method=method)

    else:
        raise TypeError("Can't handle map_file except a string or a list")


@export
class DummyMap:
    """Return constant results
        the length match the length of input
        but from the second dimensions the shape is user defined input
    """
    def __init__(self, const, shape=()):
        self.const = const
        self.shape = shape

    def __call__(self, x, **kwargs):
        shape = [len(x)] + list(self.shape)
        return np.ones(shape) * self.const

    def reduce_last_dim(self):
        assert len(self.shape) >= 1, 'Need at least 1 dim to reduce further'
        const = self.const * self.shape[-1]
        shape = list(self.shape)
        shape[-1] = 1

        return DummyMap(const, shape)


##
# Copy from https://github.com/XENONnT/private_nt_aux_files
##
def parse_extension(name):
    """Get the extention from a file name. If zipped or tarred, can contain a dot"""
    split_name = name.split('.')
    if len(split_name) == 2:
        fmt = split_name[-1]
    elif len(split_name) > 2 and 'gz' in name:
        fmt = '.'.join(split_name[-2:])
    else:
        fmt = split_name[-1]
    log.warning(f'Using {fmt} for unspecified {name}')
    return fmt
