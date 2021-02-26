import tempfile

import strax
import straxen
import wfsim
import utilix

strax.mailbox.Mailbox.DEFAULT_TIMEOUT = 60

run_id = '010000'


def test_sim_1T():
    """Test the 1T simulator (should always work with the publicly available files)"""
    with tempfile.TemporaryDirectory() as tempdir:
        testing_config_1T = dict(
            hev_gain_model=('to_pe_constant', 0.0085),
            gain_model=('to_pe_constant', 0.0085)
        )
        st = strax.Context(
            storage=tempdir,
            config=dict(
                nchunk=1, event_rate=1, chunk_size=10,
                detector='XENON1T',
                fax_config='https://raw.githubusercontent.com/XENONnT/strax_auxiliary_files/e82a63a4d2d38f291f819bb1322433d2186545d0/sim_files/fax_config_1t.json',  # noqa
                **straxen.contexts.x1t_common_config),
            **straxen.contexts.common_opts)
        st.register(wfsim.RawRecordsFromFax1T)
        st.set_config(testing_config_1T)

        rr = st.get_array(run_id, 'raw_records')
        p = st.get_array(run_id, 'peaks')
        _sanity_check(rr, p)


def test_sim_nT():
    """Test the nT simulator. Works only if one has access to the XENONnT databases"""
    if not straxen.utilix_is_configured():
        # This means we cannot load the nT files. Most likely will work
        # locally but not a travis job.
        return
    with tempfile.TemporaryDirectory() as tempdir:
        st = strax.Context(
            storage=tempdir,
            config=dict(
                nchunk=1, event_rate=1, chunk_size=10,
                detector='XENONnT',
                fax_config='fax_config_nt.json',
                **straxen.contexts.xnt_common_config),
            **straxen.contexts.common_opts)
        st.register(wfsim.RawRecordsFromFaxNT)

        rr = st.get_array(run_id, 'raw_records')
        p = st.get_array(run_id, 'peaks')
        _sanity_check(rr, p)


def _sanity_check(raw_records, peaks):
    assert len(raw_records) > 0
    assert raw_records['data'].sum() > 0
    assert peaks['data'].sum() > 0
