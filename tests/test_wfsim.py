import logging
import os.path as osp
import tempfile
import copy
import numpy as np
import strax
import straxen

import wfsim
from .test_load_resource import test_load_nt, test_load_1t
from unittest import skipIf

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(name)s %(levelname)-8s %(message)s',
    datefmt='%m-%d %H:%M')

log = logging.getLogger()
strax.mailbox.Mailbox.DEFAULT_TIMEOUT = 60

run_id = '010000'


def _sanity_check(raw_records, peaks):
    assert len(raw_records) > 0
    assert raw_records['data'].sum() > 0
    assert peaks['data'].sum() > 0


def test_sim_1T():
    """Test the 1T simulator (should always work with the publicly available files)"""
    with tempfile.TemporaryDirectory() as tempdir:
        log.debug(f'Working in {tempdir}')
        _, conf_1t = test_load_1t()
        testing_config_1t = dict(
            hev_gain_model='legacy-to-pe://1T_to_pe_placeholder',
            gain_model='legacy-to-pe://1T_to_pe_placeholder',
            gain_model_mc='legacy-to-pe://1T_to_pe_placeholder',
        )

        st = strax.Context(
            storage=tempdir,
            config=dict(
                nchunk=2, event_rate=1, chunk_size=1,
                detector='XENON1T',
                fax_config=('https://raw.githubusercontent.com/XENONnT/strax_auxiliary_files'
                            '/a5b92102505d6d0bfcdb563b6117bd4040a93435/sim_files/fax_config_1t.json'),  # noqa
                fax_config_override=dict(
                    url_base=("https://raw.githubusercontent.com/XENONnT/strax_auxiliary_files"
                              "/a5b92102505d6d0bfcdb563b6117bd4040a93435/sim_files/"),
                    **conf_1t,
                ),
                **straxen.legacy.x1t_common_config),
            **straxen.legacy.contexts_1t.get_x1t_context_config(),
        )
        st.register(wfsim.RawRecordsFromFax1T)
        log.debug(f'Setting testing config {testing_config_1t}')
        st.set_config(testing_config_1t)

        log.debug(f'Getting raw-records')
        rr = st.get_array(run_id, 'raw_records')
        p = st.get_array(run_id, 'peaks')
        _sanity_check(rr, p)
        log.info(f'All done')


@skipIf(not straxen.utilix_is_configured(), 'utilix is not configured')
def test_sim_nt_basics():
    """Test the nT simulator. Uses basic config so complicated steps are skipped. So this will test
       the simple s1 model and the simple s2 model"""

    with tempfile.TemporaryDirectory() as tempdir:
        log.debug(f'Working in {tempdir}')
        conf = copy.deepcopy(straxen.contexts.xnt_common_config)
        resource, conf_override = test_load_nt()
        conf['gain_model'] = "cmt://to_pe_model?version=ONLINE&run_id=plugin.run_id"
        conf['gain_model_mc'] = "cmt://to_pe_model?version=ONLINE&run_id=plugin.run_id"
        conf['hev_gain_model'] = 'legacy-to-pe://1T_to_pe_placeholder'
        conf['hit_min_amplitude'] = 'pmt_commissioning_initial'

        # The SPE table in this package is for a single channel
        # We generate the full SPE file for testing here
        for i in range(1, 494):
            resource.photon_area_distribution[str(i)] = \
                resource.photon_area_distribution['0']
        spe_file = osp.join(tempdir, 'XENONnT_spe_distributions.csv')
        resource.photon_area_distribution.to_csv(spe_file, index=False)
        conf_override['photon_area_distribution'] = spe_file

        st = strax.Context(
            storage=tempdir,
            config=dict(
                nchunk=1, event_rate=1, chunk_size=2,
                detector='XENONnT',
                fax_config=('https://raw.githubusercontent.com/XENONnT/WFSim'
                            '/b33656ac59d0366ccf3b20ab0686500cfd403cb6/files/XENONnT_wfsim_config.json'),
                **conf,
                fax_config_override=conf_override),
            **straxen.contexts.common_opts)
        st.register(wfsim.RawRecordsFromFaxNT)

        log.debug(f'Getting raw-records')
        rr = st.get_array(run_id, 'raw_records')
        log.debug(f'Getting peaks')
        p = st.get_array(run_id, 'peaks')
        _sanity_check(rr, p)
        log.info(f'All done')


@skipIf(not straxen.utilix_is_configured(), 'utilix is not configured')
def test_sim_nt_advanced(
        config = None
):
    """
    Test the nT simulator. Works only if one has access to the XENONnT databases.
    Clone the repo to dali and type 'pytest' to run.
    """
    with tempfile.TemporaryDirectory() as tempdir:
        log.debug(f'Working in {tempdir}')

        st = straxen.contexts.xenonnt_simulation(cmt_run_id_sim='010000',
                                                 cmt_version='global_ONLINE',
                                                 fax_config='fax_config_nt_sr0_v0.json',
                                                 _config_overlap={},)
        st.set_config(dict(hit_min_amplitude='legacy-thresholds://pmt_commissioning_initial',
                           per_pmt_truth=True,
                          ))
        st.set_config(dict(nchunk=1, event_rate=1, chunk_size=2,))

        if config is not None:
            log.warning(f'Update config with {config}')
            st.set_config(config)
        log.debug(f'Getting raw-records')
        rr = st.get_array(run_id, 'raw_records')
        log.debug(f'Getting peaks')
        p = st.get_array(run_id, 'peaks')
        _sanity_check(rr, p)
        log.info(f'All done')

        truth = st.get_array(run_id, 'truth')
        for field in 'n_pe n_photon raw_area'.split():
            assert np.all(np.isclose(truth[field], np.sum(truth[field + '_per_pmt'], axis=1)))


@skipIf(not straxen.utilix_is_configured(), 'utilix is not configured')
def test_nt_advanced_alt_s2_model():
    config = dict(
        fax_config_override=dict(
            s2_luminescence_model='simple',
            s2_time_model="s2_time_spread around zero",
            s1_lce_correction_map='XENONnT_s1_xyz_LCE_corrected_qes_MCva43fa9b_wires.json.gz',
            s1_time_spline='XENONnT_s1_proponly_va43fa9b_wires_20200625.json.gz',
            s1_model_type='optical_propagation+simple',
            override_s1_photon_time_field=False, # Set it tot the proper local field, just to test the alternative
        )
    )
    test_sim_nt_advanced(config)


@skipIf(not straxen.utilix_is_configured(), 'utilix is not configured')
def test_nt_advanced_garfield():
    config = dict(
        fax_config_override=dict(
            s2_luminescence_model='garfield',
            s2_correction_map=False,
            s2_time_model="s2_time_spread around zero",
            s1_lce_correction_map='XENONnT_s1_xyz_LCE_corrected_qes_MCva43fa9b_wires.json.gz',
            s1_time_spline='XENONnT_s1_proponly_va43fa9b_wires_20200625.json.gz',
            s1_model_type='optical_propagation+simple',
            override_s1_photon_time_field=200, # V/cm, see github.com/XENONnT/WFSim/pull/381
        )
    )
    test_sim_nt_advanced(config)

@skipIf(not straxen.utilix_is_configured(), 'utilix is not configured')
def test_nt_advanced_gas_gap_garfield():
    config = dict(
        fax_config_override=dict(
            s2_luminescence_model='garfield_gas_gap',
            s2_correction_map="XENONnT_s2_xy_map_v4_210503_mlp_3_in_1_iterated.json",
            s2_luminescence_gg= "garfield_timing_map_gas_gap_sr0.npy",
            garfield_gas_gap_map= "garfield_gas_gap_map_sr0.json",
            se_gain_map="XENONnT_se_xy_map_v1_mlp.json",
    )
    )
    test_sim_nt_advanced(config)


@skipIf(not straxen.utilix_is_configured(), 'utilix is not configured')
def test_sim_mc_chain():
    """Test the nT simulator by Geant4 output file"""

    with tempfile.TemporaryDirectory() as tempdir:
        log.debug(f'Working in {tempdir}')

        # Download test file on the test directory
        import requests
        test_g4 = 'https://raw.githubusercontent.com/XENONnT/WFSim/master/tests/geant_test_data_small.root'
        url_data = requests.get(test_g4).content
        with open('test.root', mode='wb') as f:
            f.write(url_data)
        st = straxen.contexts.xenonnt_simulation(cmt_run_id_sim='010000',
                                                 cmt_version='global_ONLINE',
                                                 _config_overlap={},)
        st.set_config(dict(gain_model_nv=("adc_nv", True),
                          ))

        epix_config = {'cut_by_eventid': True, 'debug': True, 'source_rate': 0, 'micro_separation_time': 10.,
                       'max_delay': 1e7, 'detector_config_override': None, 'micro_separation': 0.05,
                       'tag_cluster_by': 'time', 'nr_only': False}

        st.register(wfsim.RawRecordsFromMcChain)
        st.set_config(dict(
            detector='XENONnT',
            fax_file='./test.root',
            event_rate=100.,
            chunk_size=5.,
            entry_start=0,
            fax_config='fax_config_nt_design.json',
            fax_config_override=dict(
                s1_model_type='nest',
                s2_time_model="s2_time_spread around zero",
                enable_electron_afterpulses=False),
            epix_config=epix_config,
            fax_config_nveto='fax_config_nt_nveto.json',
            fax_config_override_nveto=dict(enable_noise=False,
                                           enable_pmt_afterpulses=False,
                                           enable_electron_afterpulses=False,),
            targets=('tpc', 'nveto'),
            baseline_samples_nv=("nv_baseline_constant", 26, True),
        ))

        log.debug(f'Getting raw-records')
        rr = st.get_array(run_id, 'raw_records')
        assert len(rr) > 0
        rr_nv = st.get_array(run_id, 'raw_records_nv')
        assert len(rr_nv) > 0

        log.debug(f'Getting truths')
        truth = st.get_array(run_id, 'truth', progress_bar=False)
        truth_nv = st.get_array(run_id, 'truth_nv', progress_bar=False)
        assert len(truth) > 0
        assert len(truth_nv) > 0

        log.info(f'All done')
