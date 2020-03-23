from copy import deepcopy
import os.path as osp

import numpy as np
import strax
import straxen


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
                's2_per_pmt_params': 'Kr83m_Ddriven_per_pmt_params_dataframe.csv',
                'photon_ap_cdfs': 'x1t_pmt_afterpulse_config.pkl.gz',
            })
        elif config['detector'] == 'XENONnT':
            files.update({
                'photon_area_distribution': 'XENONnT_spe_distributions.csv',
                's1_pattern_map': 'XENONnT_s1_xyz_patterns_corrected_MCv3.1.0_disks.pkl',
                's2_pattern_map': 'XENONnT_s2_xy_patterns_topbottom_corrected_MCv3.1.0_disks.pkl',
                'photon_ap_cdfs': 'xnt_pmt_afterpulse_config.pkl.gz',
            })
        else:
            raise ValueError(f"Unsupported detector {config['detector']}")

        for k in set(config).intersection(files):
            files[k] = config[k] # Allowing user to replace default with specified files
        commit = 'master'   # Replace this by a commit hash if you feel solid and responsible
        url_base = f'https://raw.githubusercontent.com/XENONnT/strax_auxiliary_files/{commit}/fax_files'
        for k, v in files.items():
            if v.startswith('/'):
                print(f"WARNING: Using local file {v} for a resource. "
                      f"Do not set this as a default or TravisCI tests will break")
            files[k] = osp.join(url_base, v)

        self.photon_area_distribution = straxen.get_resource(files['photon_area_distribution'], fmt='csv')

        if config['detector']== 'XENON1T':
            self.s1_pattern_map = make_map(files['s1_pattern_map'], fmt='json.gz')
            self.s1_light_yield_map = make_map(files['s1_light_yield_map'], fmt='json')
            self.s2_light_yield_map = make_map(files['s2_light_yield_map'], fmt='json')
            self.s2_per_pmt_params = straxen.get_resource(files['s2_per_pmt_params'], fmt='csv')

        if config['detector'] == 'XENONnT':
            self.s1_pattern_map = make_map(files['s1_pattern_map'], fmt='pkl')
            lymap = deepcopy(self.s1_pattern_map)
            lymap.data['map'] = np.sum(lymap.data['map'][:][:][:], axis=3)
            self.s1_light_yield_map = lymap

            self.s2_pattern_map = make_map(files['s2_pattern_map'], fmt='pkl')
            lymap = deepcopy(self.s2_pattern_map)
            lymap.data['map'] = np.sum(lymap.data['map'][:][:], axis=2)
            self.s2_light_yield_map = lymap

        # Electron After Pulses compressed, haven't figure out how pkl.gz works
        self.uniform_to_ele_ap = straxen.get_resource(files['ele_ap_pdfs'], fmt='pkl.gz')

        # Photon After Pulses
        self.uniform_to_pmt_ap = straxen.get_resource(files['photon_ap_cdfs'], fmt='pkl.gz')

        # Noise sample
        self.noise_data = straxen.get_resource(files['noise_file'], fmt='npy')['arr_0'].flatten()


def make_map(map_file: str, fmt='text'):
    map_data = straxen.get_resource(map_file, fmt)
    return straxen.InterpolatingMap(map_data)
