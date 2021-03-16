import tempfile
import strax
import straxen
import wfsim
import logging
import os.path as osp
from .test_load_resource import test_load_nt

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(name)s %(levelname)-8s %(message)s',
    datefmt='%m-%d %H:%M')

log = logging.getLogger()
strax.mailbox.Mailbox.DEFAULT_TIMEOUT = 60

run_id = '010000'


def test_sim_1T():
    """Test the 1T simulator (should always work with the publicly available files)"""
    with tempfile.TemporaryDirectory() as tempdir:
        log.debug(f'Working in {tempdir}')
        testing_config_1T = dict(
            hev_gain_model=('to_pe_constant', 0.0085),
            gain_model=('to_pe_constant', 0.0085)
        )
        st = strax.Context(
            storage=tempdir,
            config=dict(
                nchunk=2, event_rate=1, chunk_size=1,
                detector='XENON1T',
                fax_config=('https://raw.githubusercontent.com/XENONnT/strax_auxiliary_files'
                    '/0813736b133fccf658170207282668177898b47a/sim_files/fax_config_1t.json'),  # noqa
                **straxen.contexts.x1t_common_config),
            **straxen.contexts.common_opts)
        st.register(wfsim.RawRecordsFromFax1T)
        log.debug(f'Setting testing config {testing_config_1T}')
        st.set_config(testing_config_1T)

        log.debug(f'Getting raw-records')
        rr = st.get_array(run_id, 'raw_records')
        p = st.get_array(run_id, 'peaks')
        _sanity_check(rr, p)
        log.info(f'All done')


def test_sim_nT():
    """Test the nT simulator. Works only if one has access to the XENONnT databases"""

    with tempfile.TemporaryDirectory() as tempdir:
        log.debug(f'Working in {tempdir}')
        conf = straxen.contexts.xnt_common_config
        conf['gain_model'] = ('to_pe_constant', 0.01)
        resource, conf_override = test_load_nt()

        # The SPE table in this package is for a single channel
        # We generate the full SPE file for testing here
        for i in range(1, 494):
            resource.photon_area_distribution[str(i)] = \
                resource.photon_area_distribution['0']
        spe_file = osp.join(tempdir, 'XENONnT_spe_distributions.csv')
        resource.photon_area_distribution.to_csv(spe_file, index=False)
        conf_override['photon_area_distribution'] = spe_file

        st = strax.Context(
            storage=tempdir,
            config=dict(
                nchunk=1, event_rate=1, chunk_size=2,
                detector='XENONnT',
                fax_config=('https://raw.githubusercontent.com/XENONnT/WFSim'
                '/a412375ad1fc85f30596b2b73cd8ffe5401de42e/files/XENONnT_wfsim_config.json'),
                **conf,
                fax_config_override=conf_override),
            **straxen.contexts.common_opts)
        st.register(wfsim.RawRecordsFromFaxNT)

        log.debug(f'Getting raw-records')
        rr = st.get_array(run_id, 'raw_records')
        log.debug(f'Getting peaks')
        p = st.get_array(run_id, 'peaks')
        _sanity_check(rr, p)
        log.info(f'All done')


def _sanity_check(raw_records, peaks):
    assert len(raw_records) > 0
    assert raw_records['data'].sum() > 0
    assert peaks['data'].sum() > 0
