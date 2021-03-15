from copy import deepcopy
import os.path as osp

import numpy as np
import strax
import straxen
import logging
log = logging.getLogger('load_resource')

_cached_configs = dict()

def load_config(config):
    """Create a Resource instance from the configuration

    Uses a cache to avoid re-creating instances from the same config
    """
    h = strax.deterministic_hash(config)
    if h in _cached_configs:
        return _cached_configs[h]
    result = Resource(config)
    _cached_configs[h] = result
    return result


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
    def __init__(self, config=None):
        log.debug(f'Getting {config}')
        if config is None:
            config = dict()

        files = {
            'ele_ap_pdfs': 'x1t_se_afterpulse_delaytime.pkl.gz',
            'noise_file': 'x1t_noise_170203_0850_00_small.npz',
        }
        if config['detector'] == 'XENON1T':
            files.update({
                'photon_area_distribution': 'XENON1T_spe_distributions.csv',
                's1_light_yield_map': 'XENON1T_s1_xyz_ly_kr83m_SR1_pax-680_fdc-3d_v0.json',
                's1_pattern_map': 'XENON1T_s1_xyz_patterns_interp_corrected_MCv2.1.0.json.gz',
                's2_light_yield_map': 'XENON1T_s2_xy_ly_SR1_v2.2.json',
                's2_pattern_map': 'XENON1T_s2_xy_patterns_top_corrected_MCv2.1.0.json.gz',
                'photon_ap_cdfs': 'x1t_pmt_afterpulse_config.pkl.gz',
                'fdc_3d': 'XENON1T_FDC_SR1_data_driven_time_dependent_3d_correction_tf_nn_part1_v1.json.gz',
            })
        elif config['detector'] == 'XENONnT':
            files.update({
                'photon_area_distribution': 'XENONnT_spe_distributions.csv',
                's1_pattern_map': 'XENONnT_s1_xyz_patterns_LCE_corrected_qes_MCva43fa9b_wires.pkl',
                's2_pattern_map': 'XENONnT_s2_xy_patterns_LCE_corrected_qes_MCva43fa9b_wires.pkl',
                'photon_ap_cdfs': 'xnt_pmt_afterpulse_config.pkl.gz',
                's2_luminescence': 'XENONnT_s2_garfield_luminescence_distribution_v0.pkl.gz',
                'gas_gap_map': 'gas_gap_warping_map_January_2021.pkl',
                'nv_pmt_qe_file': 'nveto_pmt_qe.json'
            })
        else:
            raise ValueError(f"Unsupported detector {config['detector']}")

        # Allowing user to replace default with specified files
        for k in set(config).intersection(files):
            files[k] = config[k]

        commit = 'master'  # Replace this by a commit hash if you feel solid and responsible
        if config.get('url_base', False):
            url_base = config['url_base']
        elif config['detector'] == "XENON1T":
            url_base = f'https://raw.githubusercontent.com/XENONnT/strax_auxiliary_files/{commit}/sim_files'
        elif config['detector'] == "XENONnT":
            url_base = f'https://raw.githubusercontent.com/XENONnT/WFSim/{commit}/files'

        for k, v in files.items():
            if isinstance(v, list):
                continue
            log.debug(f'Obtaining {k} from {v}')
            if v.startswith('/'):
                log.warning(f"WARNING: Using local file {v} for a resource. "
                            f"Do not set this as a default or TravisCI tests will break")
                continue
            try:
                # First try downloading it via
                # https://straxen.readthedocs.io/en/latest/config_storage.html
                # downloading-xenonnt-files-from-the-database  # noqa

                # we need to add the straxen.MongoDownloader() in this
                # try: except NameError: logic because the NameError
                # gets raised if we don't have access to utilix.
                downloader = straxen.MongoDownloader()
                # FileNotFoundError, ValueErrors can be raised if we
                # cannot load the requested config
                downloaded_file = downloader.download_single(v)
                files[k] = downloaded_file
            except (FileNotFoundError, ValueError, NameError, AttributeError):
                try:
                    log.warning(f"Trying to use the private repo to load {v} locally")
                    # You might want to use this, for example if you are a developer
                    import ntauxfiles
                    files[k] = ntauxfiles.get_abspath(v)
                    log.info(f"Loading {v} is successfully from {files[k]}")
                except (ModuleNotFoundError, ImportError, FileNotFoundError):
                    log.info(f"ntauxfiles is not installed or does not have {v}")
                    # We cannot download the file from the database. We need to
                    # try to get a placeholder file from a URL.
                    raw_url = osp.join(url_base, v)
                    log.warning(f'{k} did not download, trying {raw_url}')
                    files[k] = raw_url
            log.debug(f'Downloaded {k} successfully')
                             
        self.photon_area_distribution = straxen.get_resource(files['photon_area_distribution'], fmt='csv')

        if config['detector'] == 'XENON1T':
            self.s1_pattern_map = make_map(files['s1_pattern_map'], fmt='json.gz')
            self.s1_light_yield_map = make_map(files['s1_light_yield_map'], fmt='json')
            self.s2_light_yield_map = make_map(files['s2_light_yield_map'], fmt='json')
            self.s2_pattern_map = make_map(files['s2_pattern_map'], fmt='json.gz')
            self.fdc_3d = make_map(files['fdc_3d'], fmt='json.gz')

            # Gas gap warping map
            if config['enable_gas_gap_warping']:
                self.gas_gap_length = make_map(["constant dummy", 0.25, [254,]])

        if config['detector'] == 'XENONnT':
            self.s1_pattern_map = make_map(files['s1_pattern_map'], fmt='pkl')
            if isinstance(self.s1_pattern_map, DummyMap):
                self.s1_light_yield_map = self.s1_pattern_map.reduce_last_dim()
            else:
                lymap = deepcopy(self.s1_pattern_map)
                lymap.data['map'] = np.sum(lymap.data['map'][:][:][:], axis=3, keepdims=True)
                lymap.__init__(lymap.data)
                self.s1_light_yield_map = lymap

            self.s2_pattern_map = make_map(files['s2_pattern_map'], fmt='pkl')
            if isinstance(self.s2_pattern_map, DummyMap):
                self.s2_light_yield_map = self.s2_pattern_map.reduce_last_dim()
            else:
                lymap = deepcopy(self.s2_pattern_map)
                lymap.data['map'] = np.sum(lymap.data['map'][:][:], axis=2, keepdims=True)
                lymap.__init__(lymap.data)
                self.s2_light_yield_map = lymap

            if config['s2_luminescence_model'] == 'garfield':
                self.s2_luminescence = straxen.get_resource(files['s2_luminescence'], fmt='pkl.gz')

            if config['field_distortion_on']:
                self.fdc_3d = make_map(files['fdc_3d'], fmt='json.gz')

            # Gas gap warping map
            if config['enable_gas_gap_warping']:
                gas_gap_map = straxen.get_resource(files['gas_gap_map'], fmt='pkl')
                self.gas_gap_length = lambda positions: gas_gap_map.lookup(*positions.T)

        # Spe area distributions
        self.photon_area_distribution = straxen.get_resource(files['photon_area_distribution'], fmt='csv')

        #Spe area distributions
        self.photon_area_distribution = straxen.get_resource(files['photon_area_distribution'], fmt='csv')
        # Electron After Pulses compressed, haven't figure out how pkl.gz works
        if config['enable_electron_afterpulses']:
            self.uniform_to_ele_ap = straxen.get_resource(files['ele_ap_pdfs'], fmt='pkl.gz')

        # Photon After Pulses
        if config['enable_pmt_afterpulses']:
            self.uniform_to_pmt_ap = straxen.get_resource(files['photon_ap_cdfs'], fmt='pkl.gz')

        # Noise sample
        if config['enable_noise']:
            self.noise_data = straxen.get_resource(files['noise_file'], fmt='npy')['arr_0'].flatten()

        # nVeto PMT Q.E.
        if config['neutron_veto']:
            self.nv_pmt_qe_data = straxen.get_resource(files['nv_pmt_qe_file'], fmt='json')

        log.debug(f'{self.__class__.__name__} fully initialized')

def make_map(map_file, fmt='text'):
    '''Fetch and make an instance of InterpolatingMap based on map_file
    Alternativly map_file can be a list of ["constant dummy", constant: int, shape: list]
    return an instance of  DummyMap'''
        
    if isinstance(map_file, list):
        assert map_file[0] == 'constant dummy', ('Alternative file input can only be '
            '("constant dummy", constant: int, shape: list')
        return DummyMap(map_file[1], map_file[2])

    elif isinstance(map_file, str):
        map_data = straxen.get_resource(map_file, fmt=fmt)
        return straxen.InterpolatingMap(map_data)
    
    else:
        raise TypeError("Can't handle map_file except a string or a list")

class DummyMap():
    """Return constant results
        the length match the length of input
        but from the second dimensions the shape is user defined input
    """
    def __init__(self, const, shape=()):
        self.const = const
        self.shape = shape
    def __call__(self, x):
        shape = [len(x)] + list(self.shape)
        return np.ones(shape) * self.const
    def reduce_last_dim(self):
        assert len(self.shape) >= 1, 'Need at least 1 dim to reduce further'
        const = self.const * self.shape[-1]
        shape = list(self.shape)
        shape[-1] = 1

        return DummyMap(const, shape)
