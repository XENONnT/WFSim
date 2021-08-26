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
        context = straxen.contexts.xenonnt_simulation(cmt_run_id_sim='010000')
    assert isinstance(context, strax.Context), f'{context} is not a context'

    if register is not None:
        assert issubclass(register, strax.Plugin), f'{register} is not a plugin'
        context.register(register)

    # Make sure that the non-simulated raw-record types are not requested
    context = deregister_if_not_simulated(context)

    # Search all plugins for the time field (each should have one)
    context.search_field('time')


def test_mc_chain():
    test_nt_context(wfsim.RawRecordsFromMcChain)


def test_fax_nveto():
    test_nt_context(wfsim.RawRecordsFromFaxnVeto)


def test_1t_context():
    test_nt_context(context=straxen.contexts.xenon1t_simulation())


def deregister_if_not_simulated(context, check_for_endswith=('_nv', '_mv')):
    """
    Given a context, remove nv/mv plugins if their raw records are not simulated
    :param context: A fully initialized context where the simulation
        plugin must provide "truth"
    :param check_for_endswith: Check for these patterns. If no
        raw-records of this kind are created, remove any other plugin
        that ends with these strings from the context
    :return: The cleaned-up context
    """
    simulated = context._plugin_class_registry['truth'].provides
    for endswith in check_for_endswith:
        if f'raw_records{endswith}' not in simulated:
            remove_from_registry(context, endswith)
    return context


def remove_from_registry(context, endswith):
    """Remove plugins if their name endswith a given string"""
    for p in list(context._plugin_class_registry.keys()):
        if p.endswith(endswith):
            del context._plugin_class_registry[p]
        # These plugins are known to mix nv/mv/tpc
        elif 'events_tagged' in p.provides or 'peak_veto_tags' in p.provides:
            del context._plugin_class_registry[p]
    
    return context
