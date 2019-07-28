import strax
import straxen
import wfsim

strax.mailbox.Mailbox.DEFAULT_TIMEOUT = 20


def test_sim():
    # Just some id from post-SR1, so the corrections work
    run_id = '180519_1902'

    st = strax.Context(
        storage=[],
        register=wfsim.RawRecordsFromFax,
        config=dict(nevents=4),
        **straxen.contexts.common_opts)

    rr = st.get_array(run_id, 'raw_records')

    assert len(rr) > 0
    assert rr['data'].sum() > 0
