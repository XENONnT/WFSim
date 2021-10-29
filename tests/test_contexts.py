import strax
import straxen
import wfsim


def test_nt_context(register=None, context=None):
    """
    Test a context if it is properly setup. To this end, we perform a
    simple scan of the field "time" since all plugins should have this
    field, if there is some option specified badly, we will quickly find
    out since for this scan, we activate all the plugins.
    :param register: Register a plugin (optional)
    :param context: Test with some other context than the nT simulation
    context
    """
    if not straxen.utilix_is_configured():
        return

    if context is None:
        context = straxen.contexts.xenonnt_simulation(cmt_run_id_sim='010000', cmt_version='global_ONLINE')
    assert isinstance(context, strax.Context), f'{context} is not a context'

    if register is not None:
        assert issubclass(register, strax.Plugin), f'{register} is not a plugin'

    # Search all plugins for the time field (each should have one)
    context.search_field('time')


# def test_mc_chain():
#     test_nt_context(wfsim.RawRecordsFromMcChain)


# def test_fax_nveto():
#     test_nt_context(wfsim.RawRecordsFromFaxnVeto)


# def test_1t_context():
#     test_nt_context(context=straxen.contexts.xenon1t_simulation())
