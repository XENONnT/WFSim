import tempfile
import strax
import straxen
import wfsim
import logging
import os.path as osp

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
        )
        st = strax.Context(
            storage=tempdir,
            config=dict(
                nchunk=2, event_rate=1, chunk_size=1,
                detector='XENON1T',
                fax_config=('https://raw.githubusercontent.com/XENONnT/strax_auxiliary_files'
                    '/c76f30ad20516efbcc832c97842abcba743f0017/sim_files/fax_config_1t.json'),  # noqa
                fax_config_override=dict(
                    url_base=("https://raw.githubusercontent.com/XENONnT/strax_auxiliary_files"
                              "/c76f30ad20516efbcc832c97842abcba743f0017/sim_files/"),),
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

def test_sim_nT_basics():
    """Test the nT simulator. Uses basic config so complicated steps are skipped. So this will test
       the simple s1 model and the simple s2 model"""

    with tempfile.TemporaryDirectory() as tempdir:
        log.debug(f'Working in {tempdir}')
        conf = straxen.contexts.xnt_common_config
        conf['gain_model'] = ("to_pe_placeholder", True)
        conf['hev_gain_model'] = ("to_pe_placeholder", True)
        resource, conf_override = test_load_nt()

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
                '/9e6ecfab13a314a83eec9844ba40811bc4a2dc36/files/XENONnT_wfsim_config.json'),
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

def test_sim_nT_advanced():
    """Test the nT simulator. Works only if one has access to the XENONnT databases.
        Clone the repo to dali and type 'pytest' to run. The first run will test simple s1,
        garfield s2 and noise/afterpulses. The second run will test the s1 spline model"""
    
    if not straxen.utilix_is_configured():
        log.warning(f"Utilix is not configured, skipping database-requiring tests!")
        return
    
    with tempfile.TemporaryDirectory() as tempdir:
        log.debug(f'Working in {tempdir}')

        st = strax.Context(
            storage=tempdir,
            config=dict(
                nchunk=1, event_rate=1, chunk_size=2,
                detector='XENONnT',
                fax_config=('fax_config_nt_design.json'),
                **straxen.contexts.xnt_simulation_config,),
            **straxen.contexts.common_opts)
        st.register(wfsim.RawRecordsFromFaxNT)

        log.debug(f'Getting raw-records')
        rr = st.get_array(run_id, 'raw_records')
        log.debug(f'Getting peaks')
        p = st.get_array(run_id, 'peaks')
        _sanity_check(rr, p)
        log.info(f'All done')
        
    with tempfile.TemporaryDirectory() as tempdir:
        log.debug(f'Working in {tempdir}')

        st = strax.Context(
            storage=tempdir,
            config=dict(
                nchunk=1, event_rate=1, chunk_size=2,
                detector='XENONnT',
                fax_config=('fax_config_nt_design.json'),
                fax_config_override=dict(s2_luminescence_model='simple',
                                         s1_model_type='splinesimple',), 
                        **straxen.contexts.xnt_simulation_config,),
            **straxen.contexts.common_opts)
        st.register(wfsim.RawRecordsFromFaxNT)

        log.debug(f'Getting raw-records')
        rr = st.get_array(run_id, 'raw_records')
        log.debug(f'Getting peaks')
        p = st.get_array(run_id, 'peaks')
        _sanity_check(rr, p)
        log.info(f'All done')
