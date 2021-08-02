import numba
import numpy as np
import pandas as pd
from copy import deepcopy
from scipy.interpolate import interp1d

import strax
export, __all__ = strax.exporter(export_self=False)
PULSE_MAX_DURATION = int(1e3)
N_SPLIT_LOOP = 5


@export
@numba.jit(numba.int32(numba.int64[:], numba.int64, numba.int64, numba.int64[:, :]),
           nopython=True)
def find_intervals_below_threshold(w, threshold, holdoff, result_buffer):
    """Fills result_buffer with l, r bounds of intervals in w < threshold.
    :param w: Waveform to do hitfinding in
    :param threshold: Threshold for including an interval
    :param holdoff: Holdoff number of samples after the pulse return back down to threshold
    :param result_buffer: numpy N*2 array of ints, will be filled by function.
                          if more than N intervals are found, none past the first N will be processed.
    :returns : number of intervals processed
    Boundary indices are inclusive, i.e. the right boundary is the last index which was < threshold
    """
    result_buffer_size = len(result_buffer)
    last_index_in_w = len(w) - 1

    in_interval = False
    current_interval = 0
    current_interval_start = -1
    current_interval_end = -1

    for i, x in enumerate(w):

        if x < threshold:
            if not in_interval:
                # Start of an interval
                in_interval = True
                current_interval_start = i

            current_interval_end = i

        if ((i == last_index_in_w and in_interval) or
                (x >= threshold and i >= current_interval_end + holdoff and in_interval)):
            # End of the current interval
            in_interval = False

            # Add bounds to result buffer
            result_buffer[current_interval, 0] = current_interval_start
            result_buffer[current_interval, 1] = current_interval_end
            current_interval += 1

            if current_interval == result_buffer_size:
                result_buffer[current_interval, 1] = len(w) - 1

    n_intervals = current_interval  # No +1, as current_interval was incremented also when the last interval closed
    return n_intervals


@numba.njit
def find_optical_t_range(firsts, lasts, timings, tmins, tmaxs, start=0):
    """
    Helper function find the min and max of each optical entry
    also substract tmin from the timings within each entry
    """
    
    for ix in range(start, len(firsts)):
        if firsts[ix] == lasts[ix]:
           tmins[ix] = -1
           tmaxs[ix] = -1
           # No photons in this instruction
           continue

        tmin = timings[firsts[ix]]
        tmax = timings[firsts[ix]]
        for iy in range(firsts[ix], lasts[ix]):
            if timings[iy] < tmin:
                tmin = timings[iy]
            if timings[iy] > tmax:
                tmax = timings[iy]

        tmins[ix] = tmin
        tmaxs[ix] = tmax

        timings[firsts[ix]: lasts[ix]] -= tmin


@numba.njit
def split_long_optical_pulse(firsts, lasts, timings, channels):
    """
    Helper function to split photon timings of a single optical entry into
    two entries if the is a gap longer than PULSE_MAX_DURATION ns.
    """
    for ix in range(len(firsts)):

        extra_long_time_index = []
        for iy in range(firsts[ix], lasts[ix]):
            if timings[iy] > PULSE_MAX_DURATION:
                extra_long_time_index.append(iy)

        if len(extra_long_time_index) == 0:
            continue

        for cnt, iy in enumerate(extra_long_time_index):
            cnt += firsts[ix]

            if iy > cnt:
                tmp = timings[cnt]                
                timings[cnt] = timings[iy]
                timings[iy] = tmp

                tmp = channels[cnt]                
                channels[cnt] = channels[iy]
                channels[iy] = tmp

        yield ix, firsts[ix], cnt + 1
        firsts[ix] = cnt + 1


@export
def optical_adjustment(instructions, timings, channels):
    """
    Helper function to process optical instructions so that for each entry
    1) Move the instruction timing to the first photon in the entry and move photon timings
    2) Split photon timing into maximum interval of PULSE_MAX_DURATION default 1us
       The split photon are put into new instruction append at the end of the instructions
    """
    tmins = np.zeros(len(instructions), np.int64)
    tmaxs = np.zeros(len(instructions), np.int64)

    start = 0
    for i in range(N_SPLIT_LOOP):
        find_optical_t_range(instructions['_first'], 
                             instructions['_last'],
                             timings, tmins, tmaxs,
                             start=start)

        instructions['time'][start:] += tmins[start:]
        long_pulse = ((tmaxs - tmins) > PULSE_MAX_DURATION) & (np.arange(len(instructions)) >= start)
        n_long_pulse = long_pulse.sum()
        if n_long_pulse < 1:
            break

        extra_inst = []
        for ix, first, last in split_long_optical_pulse(instructions['_first'][long_pulse],
                                                        instructions['_last'][long_pulse],
                                                        timings,
                                                        channels):

            tmp = deepcopy(instructions[np.where(long_pulse)[0][ix]])
            tmp['_first'] = first
            tmp['_last'] = last
            instructions[np.where(long_pulse)[0][ix]]['_first']=last

            extra_inst.append(tmp)

        instructions = np.append(instructions, extra_inst)
        tmins = np.hstack([tmins, np.zeros(len(extra_inst), np.int64)])
        tmaxs = np.hstack([tmaxs, np.zeros(len(extra_inst), np.int64)])
        
        start = len(instructions)

    # Instructions is now a copy so return is needed
    return instructions
