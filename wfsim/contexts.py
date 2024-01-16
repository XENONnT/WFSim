from typing import Optional
from immutabledict import immutabledict
import strax
import straxen
import wfsim
from .strax_interface import RawRecordsFromFax1T


def xenonnt_simulation_offline(
    output_folder: str = "./strax_data",
    wfsim_registry: str = "RawRecordsFromFaxNT",
    run_id: Optional[str] = None,
    global_version: Optional[str] = None,
    fax_config: Optional[str] = None,
    **kwargs,
):
    """
    :param output_folder: strax_data folder
    :param wfsim_registry: Raw_records generation mechanism,
        'RawRecordsFromFaxNT', 'RawRecordsFromMcChain', etc,
        https://github.com/XENONnT/WFSim/blob/master/wfsim/strax_interface.py
    :param run_id: Real run_id to use to fetch the corrections
    :param global_version: Global versions
        https://github.com/XENONnT/corrections/tree/master/XENONnT/global_versions
    :param fax_config: WFSim configuration files
        https://github.com/XENONnT/private_nt_aux_files/blob/master/sim_files/fax_config_nt_sr0_v4.json
    :return: strax context for simulation
    """
    if run_id is None:
        raise ValueError("Specify a run_id to load the corrections")
    if global_version is None:
        raise ValueError("Specify a correction global version")
    if fax_config is None:
        raise ValueError("Specify a simulation configuration file")

    # General strax context, register common plugins
    st = strax.Context(
        storage=strax.DataDirectory(output_folder),
        **straxen.contexts.xnt_common_opts,
        **kwargs,
    )
    # Register simulation configs required by WFSim plugins
    st.config.update(
        dict(
            detector="XENONnT",
            fax_config=fax_config,
            check_raw_record_overlaps=True,
            **straxen.contexts.xnt_common_config,
        )
    )
    # Register WFSim raw_records plugin to overwrite real data raw_records
    wfsim_plugin = getattr(wfsim, wfsim_registry)
    st.register(wfsim_plugin)
    for plugin_name in wfsim_plugin.provides:
        assert "wfsim" in str(st._plugin_class_registry[plugin_name])
    # Register offline global corrections same as real data
    st.apply_xedocs_configs(version=global_version)
    # Real data correction is run_id dependent,
    # but in sim we often use run_id not in the runDB
    # So we switch the run_id dependence to a specific run -> run_id
    local_versions = st.config
    for config_name, url_config in local_versions.items():
        if isinstance(url_config, str):
            if "run_id" in url_config:
                local_versions[config_name] = straxen.URLConfig.format_url_kwargs(
                    url_config, run_id=run_id
                )
    st.config = local_versions
    # In simulation, the raw_records generation depends on gain measurement
    st.config["gain_model_mc"] = st.config["gain_model"]
    # No blinding in simulations
    st.config["event_info_function"] = "disabled"
    return st


def xenonnt_simulation(
    output_folder="./strax_data",
    wfsim_registry="RawRecordsFromFaxNT",
    cmt_run_id_sim=None,
    cmt_run_id_proc=None,
    cmt_version="global_ONLINE",
    fax_config="fax_config_nt_design.json",
    overwrite_from_fax_file_sim=False,
    overwrite_from_fax_file_proc=False,
    cmt_option_overwrite_sim=immutabledict(),
    cmt_option_overwrite_proc=immutabledict(),
    _forbid_creation_of=None,
    _config_overlap=immutabledict(
        drift_time_gate="electron_drift_time_gate",
        drift_velocity_liquid="electron_drift_velocity",
        electron_lifetime_liquid="elife",
    ),
    **kwargs,
):
    """The most generic context that allows for setting full divergent settings for simulation
    purposes.

    It makes full divergent setup, allowing to set detector simulation
    part (i.e. for wfsim up to truth and raw_records). Parameters _sim
    refer to detector simulation parameters.

    Arguments having _proc in their name refer to detector parameters that
    are used for processing of simulations as done to the real detector
    data. This means starting from already existing raw_records and finishing
    with higher level data, such as peaks, events etc.

    If only one cmt_run_id is given, the second one will be set automatically,
    resulting in CMT match between simulation and processing. However, detector
    parameters can be still overwritten from fax file or manually using cmt
    config overwrite options.

    CMT options can also be overwritten via fax config file.
    :param output_folder: Output folder for strax data.
    :param wfsim_registry: Name of WFSim plugin used to generate data.
    :param cmt_run_id_sim: Run id for detector parameters from CMT to be used
        for creation of raw_records.
    :param cmt_run_id_proc: Run id for detector parameters from CMT to be used
        for processing from raw_records to higher level data.
    :param cmt_version: Global version for corrections to be loaded.
    :param fax_config: Fax config file to use.
    :param overwrite_from_fax_file_sim: If true sets detector simulation
        parameters for truth/raw_records from from fax_config file istead of CMT
    :param overwrite_from_fax_file_proc:  If true sets detector processing
        parameters after raw_records(peaklets/events/etc) from from fax_config
        file instead of CMT
    :param cmt_option_overwrite_sim: Dictionary to overwrite CMT settings for
        the detector simulation part.
    :param cmt_option_overwrite_proc: Dictionary to overwrite CMT settings for
        the data processing part.
    :param _forbid_creation_of: str/tuple, of datatypes to prevent form
        being written (e.g. 'raw_records' for read only simulation context).
    :param _config_overlap: Dictionary of options to overwrite. Keys
        must be simulation config keys, values must be valid CMT option keys.
    :param kwargs: Additional kwargs taken by strax.Context.
    :return: strax.Context instance

    """

    st = strax.Context(
        storage=strax.DataDirectory(output_folder),
        config=dict(
            detector="XENONnT",
            fax_config=fax_config,
            check_raw_record_overlaps=True,
            **straxen.contexts.xnt_common_config,
        ),
        **straxen.contexts.xnt_common_opts,
        **kwargs,
    )
    st.register(getattr(wfsim, wfsim_registry))

    # Make sure that the non-simulated raw-record types are not requested
    st.deregister_plugins_with_missing_dependencies()

    if straxen.utilix_is_configured(
        warning_message="Bad context as we cannot set CMT since we have no database access"
    ):
        st.apply_cmt_version(cmt_version)

    if _forbid_creation_of is not None:
        st.context_config["forbid_creation_of"] += strax.to_str_tuple(_forbid_creation_of)

    # doing sanity checks for cmt run ids for simulation and processing
    if (not cmt_run_id_sim) and (not cmt_run_id_proc):
        raise RuntimeError(
            "cmt_run_id_sim and cmt_run_id_proc are None. "
            "You have to specify at least one CMT run id. "
        )
    if (cmt_run_id_sim and cmt_run_id_proc) and (cmt_run_id_sim != cmt_run_id_proc):
        print("INFO : divergent CMT runs for simulation and processing")
        print("    cmt_run_id_sim".ljust(25), cmt_run_id_sim)
        print("    cmt_run_id_proc".ljust(25), cmt_run_id_proc)
    else:
        cmt_id = cmt_run_id_sim or cmt_run_id_proc
        cmt_run_id_sim = cmt_id
        cmt_run_id_proc = cmt_id

    # Replace default cmt options with cmt_run_id tag + cmt run id
    cmt_options_full = straxen.get_corrections.get_cmt_options(st)

    # prune to just get the strax options
    cmt_options = {key: val["strax_option"] for key, val in cmt_options_full.items()}

    # First, fix gain model for simulation
    st.set_config({"gain_model_mc": ("cmt_run_id", cmt_run_id_sim, *cmt_options["gain_model"])})
    fax_config_override_from_cmt = dict()
    for fax_field, cmt_field in _config_overlap.items():
        value = cmt_options[cmt_field]

        # URL configs need to be converted to the expected format
        if isinstance(value, str):
            opt_cfg = cmt_options_full[cmt_field]
            version = straxen.URLConfig.kwarg_from_url(value, "version")
            # We now allow the cmt name to be different from the config name
            # WFSim expects the cmt name
            value = (opt_cfg["correction"], version, True)

        fax_config_override_from_cmt[fax_field] = ("cmt_run_id", cmt_run_id_sim, *value)
    st.set_config({"fax_config_override_from_cmt": fax_config_override_from_cmt})

    # and all other parameters for processing
    for option in cmt_options:
        value = cmt_options[option]
        if isinstance(value, str):
            # for URL configs we can just replace the run_id keyword argument
            # This will become the proper way to override the run_id for cmt configs
            st.config[option] = straxen.URLConfig.format_url_kwargs(value, run_id=cmt_run_id_proc)
        else:
            # FIXME: Remove once all cmt configs are URLConfigs
            st.config[option] = ("cmt_run_id", cmt_run_id_proc, *value)

    # Done with "default" usage, now to overwrites from file
    #
    # Take fax config and put into context option
    if overwrite_from_fax_file_proc or overwrite_from_fax_file_sim:
        fax_config = straxen.get_resource(fax_config, fmt="json")
        for fax_field, cmt_field in _config_overlap.items():
            if overwrite_from_fax_file_proc:
                if isinstance(cmt_options[cmt_field], str):
                    # URLConfigs can just be set to a constant
                    st.config[cmt_field] = fax_config[fax_field]
                else:
                    # FIXME: Remove once all cmt configs are URLConfigs
                    st.config[cmt_field] = (
                        cmt_options[cmt_field][0] + "_constant",
                        fax_config[fax_field],
                    )
            if overwrite_from_fax_file_sim:
                # CMT name allowed to be different from the config name
                # WFSim needs the cmt name
                cmt_name = cmt_options_full[cmt_field]["correction"]

                st.config["fax_config_override_from_cmt"][fax_field] = (
                    cmt_name + "_constant",
                    fax_config[fax_field],
                )

    # And as the last step - manual overrrides, since they have the highest priority
    # User customized for simulation
    for option in cmt_option_overwrite_sim:
        if option not in cmt_options:
            raise ValueError(
                f"Overwrite option {option} is not using CMT by default "
                "you should just use set config"
            )
        if option not in _config_overlap.values():
            raise ValueError(
                f"Overwrite option {option} does not have mapping from CMT to fax config!"
            )
        for fax_key, cmt_key in _config_overlap.items():
            if cmt_key == option:
                cmt_name = cmt_options_full[option]["correction"]
                st.config["fax_config_override_from_cmt"][fax_key] = (
                    cmt_name + "_constant",
                    cmt_option_overwrite_sim[option],
                )
            del (fax_key, cmt_key)
    # User customized for simulation
    for option in cmt_option_overwrite_proc:
        if option not in cmt_options:
            raise ValueError(
                f"Overwrite option {option} is not using CMT by default "
                "you should just use set config"
            )

        if isinstance(cmt_options[option], str):
            # URLConfig options can just be set to constants, no hacks needed
            # But for now lets keep things consistent for people
            st.config[option] = cmt_option_overwrite_proc[option]
        else:
            # CMT name allowed to be different from the config name
            # WFSim needs the cmt name
            cmt_name = cmt_options_full[option]["correction"]
            st.config[option] = (cmt_name + "_constant", cmt_option_overwrite_proc[option])
    # Only for simulations
    st.set_config({"event_info_function": "disabled"})

    return st


def xenon1t_simulation(output_folder="./strax_data"):
    st = strax.Context(
        storage=strax.DataDirectory(output_folder),
        config=dict(
            fax_config="fax_config_1t.json", detector="XENON1T",
            **straxen.legacy.x1t_common_config,
        ),
        **straxen.legacy.get_x1t_context_config(),
    )
    st.register(RawRecordsFromFax1T)
    st.deregister_plugins_with_missing_dependencies()
    return st
