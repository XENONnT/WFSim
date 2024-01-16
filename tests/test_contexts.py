import os
import strax
import straxen
from unittest import TestCase, skipIf
from straxen.contexts import xenonnt
import wfsim


@skipIf(not straxen.utilix_is_configured(), 'utilix is not configured')
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
    if context is None:
        context = wfsim.contexts.xenonnt_simulation(cmt_run_id_sim='010000', cmt_version='global_ONLINE')
    assert isinstance(context, strax.Context), f'{context} is not a context'

    if register is not None:
        assert issubclass(register, strax.Plugin), f'{register} is not a plugin'

    # Search all plugins for the time field (each should have one)
    context.search_field('time')


# Simulation contexts are only tested when special flags are set


class TestSimContextNT(TestCase):
    @staticmethod
    def context(*args, **kwargs):
        kwargs.setdefault("cmt_version", "global_ONLINE")
        return wfsim.contexts.xenonnt_simulation(*args, **kwargs)

    @skipIf(not straxen.utilix_is_configured(), "No db access, cannot test!")
    def test_nt_sim_context_main(self):
        st = self.context(cmt_run_id_sim="008000")
        st.search_field("time")

    @skipIf(not straxen.utilix_is_configured(), "No db access, cannot test!")
    def test_nt_sim_context_alt(self):
        """Some examples of how to run with a custom WFSim context."""
        self.context(cmt_run_id_sim="008000", cmt_run_id_proc="008001")
        self.context(cmt_run_id_sim="008000", cmt_option_overwrite_sim={"elife": 1e6})

        self.context(cmt_run_id_sim="008000", overwrite_fax_file_sim={"elife": 1e6})

    @skipIf(not straxen.utilix_is_configured(), "No db access, cannot test!")
    def test_nt_diverging_context_options(self):
        """Test diverging options. Idea is that you can use different settings for processing and
        generating data, should have been handled by RawRecordsFromWFsim but is now hacked into the
        xenonnt_simulation context.

        Just to show how convoluted this syntax for the
        xenonnt_simulation context / CMT is...

        """
        self.context(
            cmt_run_id_sim="008000",
            cmt_option_overwrite_sim={"elife": ("elife_constant", 1e6, True)},
            cmt_option_overwrite_proc={"elife": ("elife_constant", 1e5, True)},
            overwrite_from_fax_file_proc=True,
            overwrite_from_fax_file_sim=True,
            _config_overlap={"electron_lifetime_liquid": "elife"},
        )

    def test_nt_sim_context_bad_inits(self):
        with self.assertRaises(RuntimeError):
            self.context(
                cmt_run_id_sim=None,
                cmt_run_id_proc=None,
            )


def test_sim_context():
    straxen.contexts.xenon1t_simulation()


@skipIf(not straxen.utilix_is_configured(), "No db access, cannot test!")
def test_sim_offline_context():
    wfsim.contexts.xenonnt_simulation_offline(
        run_id="026000",
        global_version="global_v11",
        fax_config="fax_config_nt_sr0_v4.json",
    )


@skipIf(not straxen.utilix_is_configured(), "No db access, cannot test!")
def test_offline():
    st = xenonnt("latest")
    st.provided_dtypes()

