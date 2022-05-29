import logging
import os.path as osp
import tempfile
import copy

import strax
import straxen

import wfsim
from .test_load_resource import test_load_nt

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
        testing_config_1T = dict(
            hev_gain_model=("1T_to_pe_placeholder", True),
            gain_model=("1T_to_pe_placeholder", True),
            gain_model_mc=("1T_to_pe_placeholder", True),)

        st = strax.Context(
            storage=tempdir,
            config=dict(
                nchunk=2, event_rate=1, chunk_size=1,
                detector='XENON1T',
                fax_config=('https://raw.githubusercontent.com/XENONnT/strax_auxiliary_files'
                            '/36d352580b328ff057b1588b8af8c9a6ed8ae704/sim_files/fax_config_1t.json'),  # noqa
                fax_config_override=dict(
                    url_base=("https://raw.githubusercontent.com/XENONnT/strax_auxiliary_files"
                              "/36d352580b328ff057b1588b8af8c9a6ed8ae704/sim_files/"), ),
                **straxen.contexts.x1t_common_config),
            **straxen.contexts.x1t_context_config,
        )
        st.register(wfsim.RawRecordsFromFax1T)
        log.debug(f'Setting testing config {testing_config_1T}')
        st.set_config(testing_config_1T)

        log.debug(f'Getting raw-records')
        rr = st.get_array(run_id, 'raw_records')
        p = st.get_array(run_id, 'peaks')
        _sanity_check(rr, p)
        log.info(f'All done')


@skipIf(not straxen.utilix_is_configured(), 'utilix is not configured')
def test_sim_nT_basics():
    """Test the nT simulator. Uses basic config so complicated steps are skipped. So this will test
       the simple s1 model and the simple s2 model"""

    with tempfile.TemporaryDirectory() as tempdir:
        log.debug(f'Working in {tempdir}')
        conf = copy.deepcopy(straxen.contexts.xnt_common_config)
        resource, conf_override = test_load_nt()

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
def test_sim_nT_advanced(
        config = None
):
    """Test the nT simulator. Works only if one has access to the XENONnT databases.
        Clone the repo to dali and type 'pytest' to run. The first run will test simple s1,
        garfield s2 and noise/afterpulses. The second run will test the s1 spline model"""
    with tempfile.TemporaryDirectory() as tempdir:
        log.debug(f'Working in {tempdir}')
        st = straxen.contexts.xenonnt_simulation(cmt_run_id_sim='010000',
                                                 cmt_version='global_ONLINE',
                                                 _config_overlap={},)
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

def test_nt_advanced_alt_s2_model():
    config = dict(
        fax_config_override=dict(
            s2_luminescence_model='simple',
            s2_time_model="s2_time_spread around zero",
            s1_lce_correction_map='XENONnT_s1_xyz_LCE_corrected_qes_MCva43fa9b_wires.json.gz',
            s1_time_spline='XENONnT_s1_proponly_va43fa9b_wires_20200625.json.gz',
            s1_model_type='optical_propagation+simple',
        )
    )
    test_sim_nT_advanced(config)

def test_nt_advanced_alt_s2_model_garfield():
    config = dict(
        fax_config_override=dict(
            s2_luminescence_model='garfield',
            s2_correction_map=False,
            s2_time_model="s2_time_spread around zero",
            s1_lce_correction_map='XENONnT_s1_xyz_LCE_corrected_qes_MCva43fa9b_wires.json.gz',
            s1_time_spline='XENONnT_s1_proponly_va43fa9b_wires_20200625.json.gz',
            s1_model_type='optical_propagation+simple',
        )
    )
    test_sim_nT_advanced(config)


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
                # url_base='https://raw.githubusercontent.com/XENONnT/private_nt_aux_files/master/sim_files',
                # s1_lce_correction_map=["constant dummy", 1, []],
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
