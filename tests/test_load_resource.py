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
        "neutron_veto": False,
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
            "diffusion_transverse_radial_map": False,
            "diffusion_transverse_perp_radius_map": False},
        "neutron_veto": False,
        "url_base": "https://raw.githubusercontent.com/XENONnT/WFSim/e890b18f3caab9889a166273580b06f56da5ac58/files",
        "photon_area_distribution": "XENONnT_spe_distributions_single_channel.csv",
        "s1_pattern_map": ["constant dummy", 14e-5, [494,]],
        "s2_pattern_map": ["constant dummy", 30e-5, [494,]],
        "field_dependencies_map": ["constant dummy", 1, []],
    }
    result = load_config(config)
    return result, config
