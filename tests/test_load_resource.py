from wfsim.load_resource import load_config


def test_load_1t():
    config = {
        "detector": "XENON1T",
        "s1_model_type": "simple",
        "s2_luminescence_model": "simple",
        "enable_gas_gap_warping": False,
        "enable_pmt_afterpulses": True,
        "enable_electron_afterpulses": True,
        "enable_noise": False,
        "field_distortion_on": True,
    }
    result = load_config(config)
    return result, config

def test_load_nt():
    config = {
        "detector": "XENONnT",
        "s2_luminescence_model": "simple",
        "enable_gas_gap_warping": False,
        "enable_pmt_afterpulses": False,
        "enable_electron_afterpulses": False,
        "enable_noise": False,
        "field_distortion_on": False,
        "enable_field_dependencies": {
            "survival_probability_map": True,
            "drift_speed_map": False,
            "diffusion_longitudinal_map": False,
            "diffusion_transverse_map": False},
        "url_base": "https://raw.githubusercontent.com/XENONnT/WFSim/9e6ecfab13a314a83eec9844ba40811bc4a2dc36/files",
        "photon_area_distribution": "XENONnT_spe_distributions_single_channel.csv",
        "s1_pattern_map": ["constant dummy", 14e-5, [494,]],
        "s1_lce_correction_map": ["constant dummy", 1, []],
        "s2_pattern_map": ["constant dummy", 30e-5, [494,]],
        "s2_correction_map": ["constant dummy", 1, []],
        "field_dependencies_map": ["constant dummy", 1, []],
    }
    result = load_config(config)
    return result, config
