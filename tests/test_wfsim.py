import tempfile

import strax
import straxen
import wfsim

strax.mailbox.Mailbox.DEFAULT_TIMEOUT = 60


def test_sim():
    # Just some id from post-SR1, so the corrections work
    run_id = '180519_1902'
    with tempfile.TemporaryDirectory() as tempdir:
        st = strax.Context(
            storage=tempdir,
            register=wfsim.RawRecordsFromFax,
            config=dict(dict(nchunk=1, event_rate=1, chunk_size=10)),
            **straxen.contexts.common_opts)

        rr = st.get_array(run_id, 'raw_records')
        p = st.get_array(run_id, 'peaks')

    assert len(rr) > 0
    assert rr['data'].sum() > 0
    assert p['data'].sum() > 0
