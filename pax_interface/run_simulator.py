import numba


@numba.jit(numba.int32(numba.float64[:], numba.float64, numba.int64[:, :]),
           nopython=True)
def find_intervals_above_threshold(w, threshold, result_buffer):
    """Fills result_buffer with l, r bounds of intervals in w > threshold.
    Unlike find_intervals_above_threshold(), does not smooth and split hits,
    which allows speed increase in ZLE simulation.
    :param w: Waveform to do hitfinding in
    :param threshold: Threshold for including an interval
    :param result_buffer: numpy N*2 array of ints, will be filled by function.
                          if more than N intervals are found, none past the first N will be processed.
    :returns : number of intervals processed
    Boundary indices are inclusive, i.e. the right boundary is the last index which was > threshold
    """
    result_buffer_size = len(result_buffer)
    last_index_in_w = len(w) - 1

    in_interval = False
    current_interval = 0
    current_interval_start = -1

    for i, x in enumerate(w):
        if not in_interval and x > threshold:
            # Start of an interval
            in_interval = True
            current_interval_start = i

        if in_interval and (x <= threshold or i == last_index_in_w):
            # End of the current interval
            in_interval = False

            # The interval ended just before this index
            # Unless we ended ONLY because this is the last index, then the interval ends right here
            itv_end = i - 1 if x <= threshold else i

            # Add bounds to result buffer
            result_buffer[current_interval, 0] = current_interval_start
            result_buffer[current_interval, 1] = itv_end
            current_interval += 1

            if current_interval == result_buffer_size:
                break

    # No +1, as current_interval was incremented also when the last interval closed
    n_intervals = current_interval
    return n_intervals


import numpy as np
import pandas as pd


def rand_instructions(n=1000):
    nelectrons = 10**(np.random.uniform(1, 4.8, n))

    instructions = pd.DataFrame()
    instructions['event_number'] = np.repeat(np.arange(n), 2)
    instructions['type'] = np.tile(['s1', 's2'], n)
    instructions['t'] = np.ones(n*2) * 1e6
    r = np.sqrt(np.random.uniform(0, 2500, n))
    t = np.random.uniform(-np.pi, np.pi, n)

    instructions['x'] = np.repeat(r * np.cos(t), 2)
    instructions['y'] = np.repeat(r * np.sin(t), 2)
    instructions['z'] = np.repeat(np.random.uniform(-100, 0, n), 2)
    instructions['amp'] = np.vstack(
        [np.random.uniform(3000, 3001, n), nelectrons]).T.flatten().astype(int)
    instructions['recoil'] = ['er' for i in range(n*2)]

    return instructions


from pax import core
import datetime


def init_cores(name=''):
    global core_raw, core_processor
    if name == '':
        name = datetime.datetime.now().strftime("%y%m%d_%H%M")

    core_raw = core.Processor(
        config_names=('_base', 'XENON1T', 'reduce_raw_data', 'Simulation', ),
        config_dict={
            'pax': {
                'input': 'WaveformSimulator.WaveformSimulatorFromCSV',
                'output_name': '/project2/lgrandi/zhut/sim_raw/%s' % name}}
    )

    core_processor = core.Processor(
        config_names=('_base', 'XENON1T'),
        config_dict={
            'pax': {
                'input_name': '/project2/lgrandi/zhut/sim/Fax_SE/raw/Fax_SE_000000',  # Just a place holder
                'output_name': '/project2/lgrandi/zhut/sim_processed/%s' % name,
                'look_for_config_in_runs_db': False},
            'MC': {
                'mc_generated_data': True,
                'mc_sr_parameters': "sr1",
                'mc_run_number': 10000}}
    )

    print('Save processed_data to %s Chicago time' %
          ('/project2/lgrandi/zhut/sim_processed/%s' % name))


from pax import configuration
config_names = ('_base', 'XENON1T', 'Simulation', 'SR1_parameters')
c = config = configuration.load_configuration(config_names)
c.update(c['WaveformSimulator'])
c.update(c['DEFAULT'])
c['noise_file_index'] = '/project2/lgrandi/zhut/sim/WFSimDev/real_noise_sample/noise_file_index.txt'
c['noise_file_folder'] = '/project2/lgrandi/zhut/sim/WFSimDev/real_noise_sample'


from simulator import Simulator
sim = Simulator(c)


from tqdm import tqdm
import sys
import pax

if __name__ == '__main__':
    arg = sys.argv

    n = number_of_event = 1000

    if len(arg) > 1:
        init_cores(arg[1])
    else:
        init_cores()

    instructions = rand_instructions(n)
    core_processor.number_of_events = n

    md = sys.modules['ZLE']
    md.find_intervals_above_threshold = find_intervals_above_threshold

    with tqdm(total=n) as pBar:
        for ie, e in enumerate(sim(instructions)):
            for ip, p in enumerate(core_raw.action_plugins +
                                   core_processor.action_plugins[1:]):
                if p.name == 'SortPulses' and type(e) == pax.datastructure.EventProxy:
                    block_id = e.block_id
                    data = sys.modules['zlib'].decompress(e.data['blob'])
                    e = sys.modules['pickle'].loads(data)
                    e.block_id = block_id
                e = p.process_event(e)
                if ie == 0:
                    print(p.name, type(e))

            pBar.update(1)

        core_raw.shutdown()
        core_processor.shutdown()
