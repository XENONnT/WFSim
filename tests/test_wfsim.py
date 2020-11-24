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
                detector='XENONnT',
                fax_config='https://raw.githubusercontent.com/XENONnT/'
                           'strax_auxiliary_files/master/fax_files/fax_config_nt.json',
                **straxen.contexts.xnt_common_config),
            **straxen.contexts.common_opts)
        st.register(wfsim.RawRecordsFromFaxNT)
        st.config['gain_model'] = ('to_pe_per_run', 'https://raw.githubusercontent.com/XENONnT/'
            'strax_auxiliary_files/master/fax_files/to_pe_nt.npy')

        rr = st.get_array(run_id, 'raw_records')
        p = st.get_array(run_id, 'peaks')
        _sanity_check(rr, p)


def _sanity_check(raw_records, peaks):
    assert len(raw_records) > 0
    assert raw_records['data'].sum() > 0
    assert peaks['data'].sum() > 0
