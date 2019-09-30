import pandas as pd
import os.path as osp
from .utils import InterpolatingMap as itp_map
from .utils import get_resource

def get_resource_config(config):
    if config['experiment'] == 'XENON1T':
        resource_config = {
    'photon_area_distribution': 'XENON1T_spe_distributions.csv',
    's1_light_yield_map':       'XENON1T_s1_xyz_ly_kr83m_SR1_pax-680_fdc-3d_v0.json',
    's1_pattern_map':           'XENON1T_s1_xyz_patterns_interp_corrected_MCv2.1.0.json.gz',
    's2_light_yield_map':       'XENON1T_s2_xy_ly_SR1_v2.2.json',
    's2_pattern_map':           'XENON1T_s2_xy_patterns_top_corrected_MCv2.1.0.json.gz',
    's2_per_pmt_params':        'Kr83m_Ddriven_per_pmt_params_dataframe.csv',
    'ele_ap_cdfs':              'ele_after_pulse.npy',
    'ele_ap_pdfs':              'x1t_se_afterpulse_delaytime.pkl.gz',
    'photon_ap_cdfs':           'x1t_pmt_afterpulse_config.pkl.gz',
    'noise_file':               'x1t_noise_170203_0850_00_small.npz',
        }
    if config['experiment'] == 'XENONnT':
        resource_config = {
            'photon_area_distribution': 'XENON1T_spe_distributions.csv',
            's1_light_yield_map': 'XENONnT_s1_xyz_ly_kr83m_SR1_pax-680_fdc-3d_v0.json',
            's1_pattern_map': 'XENONnT_s1_xyz_patterns_interp_corrected_MCv2.1.0.json.gz',
            's2_light_yield_map': 'XENONnT_s2_xy_ly_SR1_v2.2.json',
            's2_pattern_map': 'XENONnT_s2_xy_patterns_top_corrected_MCv2.1.0.json.gz',
            's2_per_pmt_params': 'Kr83m_Ddriven_per_pmt_params_dataframe.csv',
            'ele_ap_cdfs': 'ele_after_pulse.npy',
            'ele_ap_pdfs': 'xnt_se_afterpulse_delaytime.pkl.gz',
            'photon_ap_cdfs': 'xnt_pmt_afterpulse_config.pkl.gz',
            'noise_file': 'x1t_noise_170203_0850_00_small.npz',
        }

    for k in resource_config:
        resource_config[k] = osp.join('https://raw.githubusercontent.com/XENONnT/strax_auxiliary_files/'
                         'master/fax_files', resource_config[k])
    return resource_config

class Resource(object):
    # The private nested inner class __Resource would only be instantiate once 

    class __Resource(object):

        def __init__(self, config={}):
            self.config = get_resource_config(config)

            # Pulse
            self.photon_area_distribution = get_resource(self.config['photon_area_distribution'], fmt='csv')

            # S1
            self.s1_light_yield_map = itp_map(self.config['s1_light_yield_map'], fmt='json')
            self.s1_pattern_map = itp_map(self.config['s1_pattern_map'], fmt='json.gz')

            # S2
            self.s2_light_yield_map = itp_map(self.config['s2_light_yield_map'], fmt='json')
            if config['experiment']=='XENON1T':
                self.s2_per_pmt_params = get_resource(self.config['s2_per_pmt_params'], fmt='csv')
            if config['experiment']=='XENONnT':
                self.s2_per_pmt_params = itp_map(self.config['s2_pattern_map'], fmt='json.gz')
            # Electron After Pulses
            # self.uniform_to_ele_ap = get_resource(self.config['ele_ap_cdfs'], fmt='npy')

            # Electron After Pulses compressed, haven't figure out how pkl.gz works
            self.uniform_to_ele_ap = get_resource(self.config['ele_ap_pdfs'], fmt='pkl.gz')

            # Photon After Pulses
            self.uniform_to_pmt_ap = get_resource(self.config['photon_ap_cdfs'], fmt='pkl.gz')

            # Noise sample
            self.noise_data = get_resource(self.config['noise_file'], fmt='npy')['arr_0'].flatten()

            self.config.update(config)
    instance = None
    
    def __init__(self, config={}):
        self.config = config
        if not Resource.instance:
            Resource.instance = Resource.__Resource(config)
    
    def __getattr__(self, name):
        return getattr(self.instance, name)
