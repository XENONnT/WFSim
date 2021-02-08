import tempfile

import strax
import straxen
import wfsim

run_id = '010000'

def test_sim_1T():
    """Test the 1T simulator (should always work with the publicly available files)"""
    with tempfile.TemporaryDirectory() as tempdir:
        testing_config_1T = dict(
            hev_gain_model=('to_pe_constant', 0.0085),
            gain_model=('to_pe_constant', 0.0085)
        )
        st = straxen.contexts.xenon1t_simulation(output_folder=tempdir)
        st.set_config(testing_config_1T)

        rr = st.get_array(run_id, 'raw_records')
        p = st.get_array(run_id, 'peaks')
        _sanity_check(rr, p)


def test_sim_nT():
    """Test the nT simulator. Works only if one has access to the XENONnT databases"""
    if straxen.uconfig is None:
        # This means we cannot load the nT files. Most likely will work
        # locally but not a travis job.
        return
    with tempfile.TemporaryDirectory() as tempdir:
        st = straxen.contexts.xenonnt_simulation(output_folder=tempdir)
        rr = st.get_array(run_id, 'raw_records')
        p = st.get_array(run_id, 'peaks')
        _sanity_check(rr, p)


def _sanity_check(raw_records, peaks):
    assert len(raw_records) > 0
    assert raw_records['data'].sum() > 0
    assert peaks['data'].sum() > 0
