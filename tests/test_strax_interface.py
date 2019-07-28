import strax
import straxen
import wfsim

strax.mailbox.Mailbox.DEFAULT_TIMEOUT = 20


def test_sim():
    # Just some id from post-SR1, so the corrections work
    run_id = '180519_1902'

    st = strax.Context(
        register=wfsim.RawRecordsFromFax,
        config=dict(nevents=4),
        **straxen.contexts.common_opts)

    # Call for event_info so it immediately get processed as well
    peaks = st.get_array(run_id, 'peaks')

    assert len(peaks) > 0
    assert peaks['area'].sum() > 0
