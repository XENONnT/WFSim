import tempfile
import strax
import straxen
import wfsim
import logging
from .test_load_resource import test_load_nt

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(name)s %(levelname)-8s %(message)s',
    datefmt='%m-%d %H:%M')

log = logging.getLogger()
strax.mailbox.Mailbox.DEFAULT_TIMEOUT = 60

run_id = '010000'


def test_sim_1T():
    """Test the 1T simulator (should always work with the publicly available files)"""
    with tempfile.TemporaryDirectory() as tempdir:
        log.debug(f'Working in {tempdir}')
        testing_config_1T = dict(
            hev_gain_model=('to_pe_constant', 0.0085),
            gain_model=('to_pe_constant', 0.0085)
        )
        st = strax.Context(
            storage=tempdir,
            config=dict(
                nchunk=1, event_rate=1, chunk_size=2,
                detector='XENON1T',
                fax_config=('https://raw.githubusercontent.com/XENONnT/strax_auxiliary_files'
                    '/1d3178a375b379d5e7be05dda7831b481cd29a24/sim_files/fax_config_1t.json'),  # noqa
                **straxen.contexts.x1t_common_config),
            **straxen.contexts.common_opts)
        st.register(wfsim.RawRecordsFromFax1T)
        log.debug(f'Setting testing config {testing_config_1T}')
        st.set_config(testing_config_1T)

        log.debug(f'Getting raw-records')
        rr = st.get_array(run_id, 'raw_records')
        p = st.get_array(run_id, 'peaks')
        _sanity_check(rr, p)
        log.info(f'All done')


def test_sim_nT():
    """Test the nT simulator. Works only if one has access to the XENONnT databases"""

    with tempfile.TemporaryDirectory() as tempdir:
        log.debug(f'Working in {tempdir}')
        conf = straxen.contexts.xnt_common_config
        conf['gain_model'] = ('to_pe_constant', 0.01)
        conf_override = test_load_nt()
        st = strax.Context(
            storage=tempdir,
            config=dict(
                nchunk=1, event_rate=1, chunk_size=2,
                detector='XENONnT',
                fax_config=('https://raw.githubusercontent.com/XENONnT/WFSim'
                '/22004e28421044452f42aaf5797be7186e07bbba/files/XENONnT_wfsim_config.json'),
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


def _sanity_check(raw_records, peaks):
    assert len(raw_records) > 0
    assert raw_records['data'].sum() > 0
    assert peaks['data'].sum() > 0
