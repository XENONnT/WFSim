import strax
import straxen
import wfsim


def test_nt_context(register=None):
    """
    Test a context if it is properly setup. To this end, we perform a
    simple scan of the field "time" since all plugins should have this
    field, if there is some option specified badly, we will quickly find
    out since for this scan, we activate all the plugins.
    :param register: Register a plugin (optional)
    """
    st = straxen.contexts.xenonnt_simulation()
    if register is not None:
        assert issubclass(register, strax.Plugin), f'{register} is not a plugin'
        st.register(register)
    st.search_field('time')


def test_mc_chain():
    test_nt_context(wfsim.RawRecordsFromMcChain)


def test_fax_nveto():
    test_nt_context(wfsim.RawRecordsFromFaxnVeto)


def test_1t_context(register=None):
    st = straxen.contexts.xenon1t_simulation()
    if register is not None:
        st.register(register)
    st.search_field('time')
