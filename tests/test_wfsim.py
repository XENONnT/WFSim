import tempfile
import strax
import straxen
import wfsim
import logging

log = logging.getLogger('Tests')
logging.basicConfig(level=logging.DEBUG)
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
                fax_config='https://raw.githubusercontent.com/XENONnT/strax_auxiliary_files/0b5a11195554d106c99784d8ad84805b0f42d51d/sim_files/fax_config_1t.json',  # noqa
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
    if not straxen.utilix_is_configured():
        # This means we cannot load the nT files. Most likely will work
        # locally but not a travis job.
        return
    with tempfile.TemporaryDirectory() as tempdir:
        log.debug(f'Working in {tempdir}')
        st = strax.Context(
            storage=tempdir,
            config=dict(
                nchunk=1, event_rate=1, chunk_size=2,
                detector='XENONnT',
                fax_config='fax_config_nt_design.json',
                **straxen.contexts.xnt_common_config),
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
