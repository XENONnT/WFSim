import tempfile

import strax
import straxen
import wfsim

strax.mailbox.Mailbox.DEFAULT_TIMEOUT = 60

# Just some id from post-SR1, so the corrections work
run_id = '180519_1902'


def test_sim_nt():
    with tempfile.TemporaryDirectory() as tempdir:
        st = strax.Context(
            storage=tempdir,
            register=wfsim.RawRecordsFromFax,
            config=dict(nchunk=1, event_rate=1, chunk_size=10,
                        detector='XENONnT',
                        to_pe_file= 'https://raw.githubusercontent.com/XENONnT/'
                                       'strax_auxiliary_files/master/fax_files/to_pe_nt.npy',
                        fax_config='https://raw.githubusercontent.com/XENONnT/'
                                   'strax_auxiliary_files/master/fax_files/fax_config_nt.json',
                        **straxen.contexts.xnt_common_config),
            **straxen.contexts.common_opts)
        rr = st.get_array(run_id, 'raw_records')
        p = st.get_array(run_id, 'peaks')
        _sanity_check(rr, p)


def test_sim():
    with tempfile.TemporaryDirectory() as tempdir:
        st = strax.Context(
            storage=tempdir,
            register=wfsim.RawRecordsFromFax,
            config=dict(nchunk=1, event_rate=1, chunk_size=10,
                        detector='XENON1T',
                        **straxen.contexts.x1t_common_config),
            **straxen.contexts.common_opts)

        rr = st.get_array(run_id, 'raw_records')
        p = st.get_array(run_id, 'peaks')
        _sanity_check(rr, p)

        # Test simulation config override
        # We'll set the extraction yield to 0, then check that the
        # total simulated area is much less than before
        # TODO: it would be nicer to do this without random instructions...
        st.set_config(dict(fax_config_override=dict(
            electron_extraction_yield=0)))
        rr2 = st.get_array(run_id, 'raw_records')
        p2 = st.get_array(run_id, 'peaks')
        _sanity_check(rr2, p2)

        assert p2['area'].sum() < 0.1 * p['area'].sum()


def _sanity_check(raw_records, peaks):
    assert len(raw_records) > 0
    assert raw_records['data'].sum() > 0
    assert peaks['data'].sum() > 0
