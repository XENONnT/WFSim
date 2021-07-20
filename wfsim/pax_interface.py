import logging
import time
import pickle
import zipfile
import zlib
from collections import namedtuple
import os

import numpy as np
import pandas as pd

import wfsim
from .pax_datastructure import datastructure
from .strax_interface import *
from strax import exporter
from straxen.common import get_resource

export, __all__ = exporter()


@export
class PaxEvents(object):
    def __init__(self, config):
        self.config = config
        self.rawdata = wfsim.RawData(self.config)
        
        self.truth_buffer = np.zeros(100000,
                                     dtype=instruction_dtype + truth_extra_dtype + [('fill', bool)])  # 500 s1 + 500 s2

    def __call__(self, instructions):
        event_i = 0  # Indices of event
        new_event = True

        first_left = np.inf
        last_right = -np.inf
        
        for channel, left, right, data in self.rawdata(instructions, self.truth_buffer):
            if self.rawdata.instruction_event_number > event_i:
                event.start_time = (first_left - 100000) * self.config['sample_duration']
                event.stop_time = (last_right + 100000) * self.config['sample_duration']
                yield event
                event_i = self.rawdata.instruction_event_number
                new_event = True

            if new_event:
                event = datastructure.Event(event_number=event_i,
                                            start_time=0,
                                            stop_time=int(3e6),
                                            n_channels=self.config['n_channels'],
                                            sample_duration=self.config['sample_duration'],
                                            pulses=[],)
                new_event = False
                first_left = left

            if right > last_right:
                last_right = right
            event.pulses.append(datastructure.Pulse(
                channel=channel,
                left=left - (first_left - 100000),
                raw_data=data))


EventProxy = namedtuple('EventProxy', ['data', 'event_number', 'block_id'])


default_config = {
    'fax_file': None,
    'detector': 'XENON1T',
    'field_distortion_on': True,
    'event_rate': 1,  # Must set to one so chunk can be interpret as an event
    'chunk_size': 1,  # Must set to one so chunk can be interpret as an event
    'nchunk': 200,  # Number of events
    'fax_config': ('https://raw.githubusercontent.com/XENONnT/'
                   'strax_auxiliary_files/master/fax_files/fax_config_1t.json'),
    'samples_to_store_before': 2,
    'samples_to_store_after': 20,
    'right_raw_extension': 50000,
    'trigger_window': 50,
    'zle_threshold': 0,
    'run_number': 10000,  # Change run_number to prevent overwritting
    'events_per_file': 1000,
    'output_name': './pax_data'  # Each run will be saved to a subfolder under output_name
}


@export
class PaxEventSimulator(object):
    """
    Simulate wf from instruction and stored in wfsim.pax_datastructure.datastructure.Event
    mimicing pax.datastructure.Event
    Then pickled, compressed and saved mimicing pax raw data zips.

    Call compute to start the simulation process.
    """

    def __init__(self, config={}):
        self.config = default_config
        self.config.update(get_resource(self.config['fax_config'], fmt='json'))
        self.config.update(config)

        if self.config['fax_file']:
            if self.config['fax_file'][-5:] == '.root':
                self.instructions = read_g4(self.config['fax_file'])
                self.config['nevents'] = np.max(self.instructions['event_number'])
            else:
                self.instructions = instruction_from_csv(self.config['fax_file'])
                self.config['nevents'] = np.max(self.instructions['event_number'])

        else:
            self.instructions = rand_instructions(self.config)
            
        self.pax_event = PaxEvents(self.config)
        self.transfer_plugin = self.WriteZippedEncoder(self.config)        
        self.output_plugin = self.WriteZipped(self.config)

    class WriteZippedEncoder(object):
        # Pax WriteZippedEncoder plugin with all parent class method extraced
        def __init__(self, config):
            self.config = config

        @staticmethod
        def make_event_proxy(event, data, block_id=None):
            if block_id is None:
                block_id = event.block_id
            return EventProxy(data=data, event_number=event.event_number, block_id=block_id)

        def transfer_event(self, event):
            data = pickle.dumps(event)
            data = zlib.compress(data, 4)
            # We also add start and stop time to the data, for use in MongoDBClearUntriggered
            return self.make_event_proxy(event, data=dict(blob=data,
                                                          start_time=event.start_time,
                                                          stop_time=event.stop_time))

    class WriteZipped(object):
        # Pax WriteZipped plugin with all parent class method extraced
        file_extension = 'zip'

        def __init__(self, config):
            self.config = config
            self.events_per_file = self.config.get('events_per_file', 50)
            self.first_event_in_current_file = None
            self.last_event_written = None
            
            self.output_dir = os.path.join(self.config['output_name'],
                                           '%s_MC_%d' % (self.config['detector'], self.config['run_number']))
            os.makedirs(self.output_dir, exist_ok=True)
            
            # Start the temporary file. Events will first be written here, until events_per_file is reached
            self.tempfile = os.path.join(self.output_dir, 'temp.' + self.file_extension)
        
        def open_new_file(self, first_event_number):
            """Opens a new file, closing any old open ones"""
            if self.last_event_written is not None:
                self.close_current_file()
            self.first_event_in_current_file = first_event_number
            self.events_written_to_current_file = 0
            self.current_file = zipfile.ZipFile(self.tempfile, mode='w')
        
        def write_event(self, event_proxy):
            """Write one more event to the folder, opening/closing files as needed"""
            if self.last_event_written is None \
                    or self.events_written_to_current_file >= self.events_per_file:
                self.open_new_file(first_event_number=event_proxy.event_number)

            self.current_file.writestr(str(event_proxy.event_number), event_proxy.data['blob'])

            self.events_written_to_current_file += 1
            self.last_event_written = event_proxy.event_number
        
        def close_current_file(self):
            """Closes the currently open file, if there is one. Also handles temporary file renaming. """
            if self.last_event_written is None:
                print("You didn't write any events... Did you crash pax?")
                return

            self.current_file.close()

            # Rename the temporary file to reflect the events we've written to it
            os.rename(self.tempfile,
                      os.path.join(self.output_dir,
                                   '%s-%d-%09d-%09d-%09d.%s' % (self.config['detector'],
                                                                self.config['run_number'],
                                                                self.first_event_in_current_file,
                                                                self.last_event_written,
                                                                self.events_written_to_current_file,
                                                                self.file_extension)))

    def compute(self):
        for event in self.pax_event(self.instructions):            
            event = self.transfer_plugin.transfer_event(event)    
            self.output_plugin.write_event(event)
        self.output_plugin.close_current_file()

        # Save truth file as well
        truth_file_path = os.path.join(self.output_plugin.output_dir,
                                       '%s-%d-truth.csv' % (self.config['detector'], self.config['run_number']))
        truth = pd.DataFrame(self.pax_event.truth_buffer[self.pax_event.truth_buffer['fill']])
        truth.drop(columns='fill', inplace=True)
        truth.to_csv(truth_file_path, index=False)
