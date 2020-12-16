import tempfile

import strax
import straxen
import wfsim

strax.mailbox.Mailbox.DEFAULT_TIMEOUT = 60

run_id = '010000'

def test_sim():
    with tempfile.TemporaryDirectory() as tempdir:
        st = strax.Context(
            storage=tempdir,
            config=dict(
                nchunk=1, event_rate=1, chunk_size=10,
                detector='XENON1T',
                fax_config='https://raw.githubusercontent.com/XENONnT/strax_auxiliary_files/90a532347fe46fe84fc3ad4f89241c9b0928bd43/fax_files/fax_config_1t.json',
                **straxen.contexts.x1t_common_config),
            **straxen.contexts.common_opts)
        st.register(wfsim.RawRecordsFromFax1T)
        st.config['gain_model'] = ('to_pe_constant', 0.0085)

        rr = st.get_array(run_id, 'raw_records')
        p = st.get_array(run_id, 'peaks')
        _sanity_check(rr, p)


def _sanity_check(raw_records, peaks):
    assert len(raw_records) > 0
    assert raw_records['data'].sum() > 0
    assert peaks['data'].sum() > 0
