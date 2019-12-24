from copy import deepcopy
import os.path as osp

import numpy as np
import straxen


def get_resource_config(config):
    resource_config = {
        'ele_ap_pdfs':      'x1t_se_afterpulse_delaytime.pkl.gz',
        'ele_ap_cdfs':      'ele_after_pulse.npy',
        'noise_file':       'x1t_noise_170203_0850_00_small.npz',
    }
    if config['detector'] == 'XENON1T':
        resource_config.update({
            'photon_area_distribution': 'XENON1T_spe_distributions.csv',
            's1_light_yield_map':       'XENON1T_s1_xyz_ly_kr83m_SR1_pax-680_fdc-3d_v0.json',
            's1_pattern_map':           'XENON1T_s1_xyz_patterns_interp_corrected_MCv2.1.0.json.gz',
            's2_light_yield_map':       'XENON1T_s2_xy_ly_SR1_v2.2.json',
            's2_pattern_map':           'XENON1T_s2_xy_patterns_top_corrected_MCv2.1.0.json.gz',
            's2_per_pmt_params':        'Kr83m_Ddriven_per_pmt_params_dataframe.csv',
            'photon_ap_cdfs':           'x1t_pmt_afterpulse_config.pkl.gz',
        })
    elif config['detector'] == 'XENONnT':
        resource_config.update({
            'photon_area_distribution': 'XENONnT_spe_distributions.csv',
            's1_pattern_map':           'XENONnT_s1_xyz_patterns_corrected_MCv3.0.0.json.gz',
            's2_pattern_map':           'XENONnT_s2_xy_patterns_topbottom_corrected_MCv3.0.0.json.gz',
            'photon_ap_cdfs':           'xnt_pmt_afterpulse_config.pkl.gz',
        })
    else:
        raise ValueError("Unsupported detector {config['detector']}")

    url_base = 'https://raw.githubusercontent.com/XENONnT/strax_auxiliary_files/master/fax_files'
    for k, v in resource_config.items():
        if v.startswith('/'):
            print(f"WARNING: Using local file {v} for a resource. "
                  f"Do not set this as a default or TravisCI tests will break")
        resource_config[k] = osp.join(url_base, v)
    return resource_config


def make_map(map_file: str, fmt='text'):
    if map_file.startswith('/dali/lgrandi/pgaemers'):
        print(f"Kun je {map_file} uploaden naar Github Peter? Dankjewel! -- Jelle")
    map_data = straxen.get_resource(map_file, fmt)
    return straxen.InterpolatingMap(map_data)


class Resource():
    # The private nested inner class __Resource would only be instantiate once 

    class __Resource():

        def __init__(self, config=None):
            if config is None:
                config = dict()
            self.config = get_resource_config(config)

            # Pulse
            self.photon_area_distribution = straxen.get_resource(self.config['photon_area_distribution'], fmt='csv')

            self.s1_pattern_map = make_map(self.config['s1_pattern_map'], fmt='json.gz')

            if config['detector']== 'XENON1T':
                # S1
                self.s1_light_yield_map = make_map(self.config['s1_light_yield_map'], fmt='json')
                # S2
                self.s2_light_yield_map = make_map(self.config['s2_light_yield_map'], fmt='json')
                self.s2_per_pmt_params = straxen.get_resource(self.config['s2_per_pmt_params'], fmt='csv')

            if config['detector'] == 'XENONnT':
                lymap = deepcopy(self.s1_pattern_map)
                lymap['map'] = np.sum(lymap['map'][:][:][:], axis=3)
                self.s1_light_yield_map = lymap

                self.s2_pattern_map = straxen.get_resource(self.config['s2_pattern_map'], fmt='json.gz')
                lymap = deepcopy(self.s2_pattern_map)
                lymap['map'] = np.sum(lymap['map'][:][:], axis=2)
                self.s2_light_yield_map = lymap

            # Electron After Pulses compressed, haven't figure out how pkl.gz works
            self.uniform_to_ele_ap = straxen.get_resource(self.config['ele_ap_pdfs'], fmt='pkl.gz')

            # Photon After Pulses
            self.uniform_to_pmt_ap = straxen.get_resource(self.config['photon_ap_cdfs'], fmt='pkl.gz')

            # Noise sample
            self.noise_data = straxen.get_resource(self.config['noise_file'], fmt='npy')['arr_0'].flatten()

            self.config.update(config)

    instance = None
    
    def __init__(self, config=None):
        if config is None:
            config = dict()
        self.config = config
        if not Resource.instance:
            Resource.instance = Resource.__Resource(config)
    
    def __getattr__(self, name):
        return getattr(self.instance, name)
