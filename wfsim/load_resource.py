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
    def __init__(self, config=None):
        log.debug(f'Getting {config}')
        if config is None:
            config = dict()
        config = deepcopy(config)

        files = {
            'ele_ap_pdfs': 'x1t_se_afterpulse_delaytime.pkl.gz',
            'ele_ap_cdfs': 'ele_after_pulse.npy',
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
                's1_pattern_map': 'XENONnT_s1_xyz_patterns_corrected_qes_MCva43fa9b_wires.pkl',
                's2_pattern_map': 'XENONnT_s2_xy_patterns_topbottom_corrected_qes_MCva43fa9b_wires.pkl',
                'photon_ap_cdfs': 'xnt_pmt_afterpulse_config.pkl.gz',
                's2_luminescence': 'XENONnT_s2_garfield_luminescence_distribution_v0.pkl.gz',
                'gas_gap_map': 'gas_gap_warping_map_January_2021.pkl',
            })
        else:
            raise ValueError(f"Unsupported detector {config['detector']}")

        for k in set(config).intersection(files):
            files[k] = config[k]  # Allowing user to replace default with specified files
        commit = 'master'   # Replace this by a commit hash if you feel solid and responsible
        url_base = f'https://raw.githubusercontent.com/XENONnT/strax_auxiliary_files/{commit}/sim_files'
        for k, v in files.items():
            log.debug(f'Obtaining {k} from {v}')
            if v.startswith('/'):
                log.warning(f"WARNING: Using local file {v} for a resource. "
                            f"Do not set this as a default or TravisCI tests will break")
            try:
                # First try downloading it via
                # https://straxen.readthedocs.io/en/latest/config_storage.html#downloading-xenonnt-files-from-the-database  # noqa

                # we need to add the straxen.MongoDownloader() in this
                # try: except NameError: logic because the NameError
                # gets raised if we don't have access to utilix.
                downloader = straxen.MongoDownloader()
                # FileNotFoundError, ValueErrors can be raised if we
                # cannot load the requested config
                downloaded_file = downloader.download_single(v)
                files[k] = downloaded_file
            except (FileNotFoundError, ValueError, NameError, AttributeError):
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
            # TODO
            #  config not set
            self.gas_gap_length = lambda positions: np.ones(253)

        if config['detector'] == 'XENONnT':
            self.s1_pattern_map = make_map(files['s1_pattern_map'], fmt='pkl')
            lymap = deepcopy(self.s1_pattern_map)
            lymap.data['map'] = np.sum(lymap.data['map'][:][:][:], axis=3, keepdims=True)
            lymap.__init__(lymap.data)
            self.s1_light_yield_map = lymap

            self.s2_pattern_map = make_map(files['s2_pattern_map'], fmt='pkl')
            lymap = deepcopy(self.s2_pattern_map)
            lymap.data['map'] = np.sum(lymap.data['map'][:][:], axis=2, keepdims=True)
            lymap.__init__(lymap.data)
            self.s2_light_yield_map = lymap
            self.s2_luminescence = straxen.get_resource(files['s2_luminescence'], fmt='pkl.gz')
            self.fdc_3d = dummy_map(result=0)
            gas_gap_map = straxen.get_resource(files['gas_gap_map'], fmt='pkl')
            self.gas_gap_length = lambda positions: gas_gap_map.lookup(*positions.T)

        # Electron After Pulses compressed, haven't figure out how pkl.gz works
        self.uniform_to_ele_ap = straxen.get_resource(files['ele_ap_pdfs'], fmt='pkl.gz')

        # Photon After Pulses
        self.uniform_to_pmt_ap = straxen.get_resource(files['photon_ap_cdfs'], fmt='pkl.gz')

        # Noise sample
        self.noise_data = straxen.get_resource(files['noise_file'], fmt='npy')['arr_0'].flatten()

        log.debug(f'{self.__class__.__name__} fully initialized')

def make_map(map_file: str, fmt='text'):
    map_data = straxen.get_resource(map_file, fmt)
    return straxen.InterpolatingMap(map_data)


class dummy_map():
    def __init__(self, result):
        self.result = result
    def __call__(self, positions):
        return np.ones(positions.shape[0]) * self.result
