import pandas as pd
from utils import InterpolatingMap as itp_map
from utils import get_resource

pax_folder = '/project/lgrandi/anaconda3/envs/pax_head/lib/python3.4/site-packages/pax-6.10.1-py3.4.egg/pax/data/'
ty_folder = '/project2/lgrandi/zhut/LCE_JSONs/'

resource_config = {
    'photon_area_distribution': pax_folder+'XENON1T_spe_distributions.csv',
    's1_light_yield_map': ty_folder+'XENON1T_s1_xyz_ly_kr83m_SR2_pax-6101_fdc-3d_v1.3.json',
    's1_pattern_map': pax_folder+'XENON1T_s1_xyz_patterns_interp_corrected_MCv2.4.1.json.gz',
    's2_light_yield_map': ty_folder+'XENON1T_s2_xy_ly_SR1_v3.2.json',
    's2_pattern_map': pax_folder+'XENON1T_s2_xy_patterns_top_corrected_MCv2.4.1.json.gz',
    's2_per_pmt_params': '/project2/lgrandi/zhut/sim/Kr83m_Ddriven_per_pmt_params_dataframe.csv',
    'ele_ap_cdfs': '/project2/lgrandi/zhut/sim/WFSimDev/ele_after_pulse.npy',
    'ele_ap_pdfs': '/project2/lgrandi/zhut/sim/WFSimDev/x1t_se_afterpulse_delaytime.pkl.gz',
    'photon_ap_cdfs': '/project2/lgrandi/zhut/sim/WFSimDev/pmt_after_pulse_v2.npy',
    'noise_file': '/project2/lgrandi/zhut/sim/WFSimDev/x1t_noise_170203_0850_00_small.npz',
}

class Resource(object):
    def __init__(self, config={}):
        self.config = resource_config
        self.config.update(config)

        # Pulse
        self.photon_area_distribution = get_resource(self.config['photon_area_distribution'], fmt='csv')

        # S1
        self.s1_light_yield_map = itp_map(self.config['s1_light_yield_map'])
        self.s1_pattern_map = itp_map(self.config['s1_pattern_map'])

        # S2
        self.s2_light_yield_map = itp_map(self.config['s2_light_yield_map'])
        self.s2_per_pmt_params = get_resource(self.config['s2_per_pmt_params'], fmt='csv')

        # Electron After Pulses
        # self.uniform_to_ele_ap = get_resource(self.config['ele_ap_cdfs'], fmt='npy')

        # Electron After Pulses compressed, haven't figure out how pkl.gz works
        self.uniform_to_ele_ap = get_resource(self.config['ele_ap_pdfs'], fmt='pkl.gz')

        # Photon After Pulses
        self.uniform_to_pmt_ap = get_resource(self.config['photon_ap_cdfs'], fmt='npy_pickle').item()
        
        # Noise sample
        self.noise_data = get_resource(self.config['noise_file'], fmt='npy')['arr_0'].flatten()
        