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


def test_load_nt():
    config = {
        "detector": "XENONnT",
        "s1_model_type": "simple",
        "s2_luminescence_model": "simple",
        "enable_gas_gap_warping": False,
        "enable_pmt_afterpulses": False,
        "enable_electron_afterpulses": False,
        "enable_noise": False,
        "field_distortion_on": False,
        "neutron_veto": False,
        "url_base": "https://raw.githubusercontent.com/XENONnT/WFSim/ce59b93335d233570c184e13b42319210279e0fa/files",
        "photon_area_distribution": "XENONnT_spe_distributions_1T_copy.csv",
        "s1_pattern_map": ["constant dummy", 14e-5, [494,]],
        "s2_pattern_map": ["constant dummy", 30e-5, [494,]],
    }
    result = load_config(config)
