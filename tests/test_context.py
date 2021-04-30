import straxen
import wfsim


def test_nt_context(register=None):
    st = straxen.contexts.xenonnt_simulation()
    if register is not None:
        st.register(register)
    st.search_field('time')


def test_mc_chain():
    test_nt_context(wfsim.RawRecordsFromMcChain)


def test_fax_nveto():
    test_nt_context(wfsim.RawRecordsFromFaxnVeto)
