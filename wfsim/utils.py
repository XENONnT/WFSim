import io
import socket
import sys
import tarfile
import os.path as osp
import os
import inspect
import urllib.request

import logging
import re

def exporter(export_self=False):
    """Export utility modified from https://stackoverflow.com/a/41895194
    Returns export decorator, __all__ list
    """
    all_ = []
    if export_self:
        all_.append('exporter')

    def decorator(obj):
        all_.append(obj.__name__)
        return obj

    return decorator, all_

export, __all__ = exporter(export_self=True)

import numpy as np
import json, gzip
import pickle

cache_dict = dict()

@export
def get_resource(x, fmt='text'):
    """Return contents of file or URL x
    :param binary: Resource is binary. Return bytes instead of a string.
    """
    is_binary = fmt != 'text'

    # Try to retrieve from in-memory cache
    if x in cache_dict:
        return cache_dict[x]

    if '://' in x:
        # Web resource; look first in on-disk cache
        # to prevent repeated downloads.
        #cache_fn = strax.utils.deterministic_hash(x)
        cache_fn = x.split('/')[-1]
        cache_folders = ['./resource_cache',
                         '/tmp/straxen_resource_cache',
                         '/dali/lgrandi/strax/resource_cache']
        for cache_folder in cache_folders:
            try:
                os.makedirs(cache_folder, exist_ok=True)
            except (PermissionError, OSError):
                continue
            cf = osp.join(cache_folder, cache_fn)
            if osp.exists(cf):
                return get_resource(cf, fmt=fmt)

        # Not found in any cache; download
        result = urllib.request.urlopen(x).read()
        if not is_binary:
            result = result.decode()

        # Store in as many caches as possible
        m = 'wb' if is_binary else 'w'
        available_cf = None
        for cache_folder in cache_folders:
            if not osp.exists(cache_folder):
                continue
            cf = osp.join(cache_folder, cache_fn)
            try:
                with open(cf, mode=m) as f:
                    f.write(result)
            except Exception:
                pass
            else:
                available_cf = cf
        if available_cf is None:
            raise RuntimeError(
                "Could not load {x},"
                "none of the on-disk caches are writeable??")

        # Retrieve result from file-cache
        # (so we only need one format-parsing logic)
        return get_resource(available_cf, fmt=fmt)

    # File resource
    if fmt == 'npy':
        result = np.load(x)
    if fmt == 'npy_pickle':
        result = np.load(x, allow_pickle=True)
    elif fmt == 'binary':
        with open(x, mode='rb') as f:
            result = f.read()
    elif fmt == 'text':
        with open(x, mode='r') as f:
            result = f.read()
    elif fmt == 'json':
        with open(x, mode='r') as f:
            result = json.load(f)
    elif fmt == 'json.gz':
        with gzip.open(x, 'rb') as f:
            result = json.load(f)
    elif fmt == 'pkl.gz':
        with gzip.open(x, 'rb') as f:
            result = pickle.load(f)
    elif fmt == 'csv':
        result = pd.read_csv(x)
    elif fmt == 'hdf':
        result = pd.read_hdf(x)
    return result

import pandas as pd
from scipy.interpolate import interp1d

def init_spe_scaling_factor_distributions(file):
    # Extract the spe pdf from a csv file into a pandas dataframe
    spe_shapes = pd.read_csv(file)

    # Create a converter array from uniform random numbers to SPE gains (one interpolator per channel)
    # Scale the distributions so that they have an SPE mean of 1 and then calculate the cdf
    uniform_to_pe_arr = []
    for ch in spe_shapes.columns[1:]:  # skip the first element which is the 'charge' header
        if spe_shapes[ch].sum() > 0:
            mean_spe = (spe_shapes['charge'] * spe_shapes[ch]).sum() / spe_shapes[ch].sum()
            scaled_bins = spe_shapes['charge'] / mean_spe
            cdf = np.cumsum(spe_shapes[ch]) / np.sum(spe_shapes[ch])
        else:
            # if sum is 0, just make some dummy axes to pass to interpolator
            cdf = np.linspace(0, 1, 10)
            scaled_bins = np.zeros_like(cdf)

        uniform_to_pe_arr.append(interp1d(cdf, scaled_bins))
    if uniform_to_pe_arr != []:
        return uniform_to_pe_arr
    
from scipy.spatial import cKDTree

class InterpolateAndExtrapolate(object):
    """Linearly interpolate- and extrapolate using inverse-distance
    weighted averaging between nearby points.
    """

    def __init__(self, points, values, neighbours_to_use=None):
        """
        :param points: array (n_points, n_dims) of coordinates
        :param values: array (n_points) of values
        :param neighbours_to_use: Number of neighbouring points to use for
        averaging. Default is 2 * dimensions of points.
        """
        self.kdtree = cKDTree(points)
        self.values = values
        if neighbours_to_use is None:
            neighbours_to_use = points.shape[1] * 2
        self.neighbours_to_use = neighbours_to_use

    def __call__(self, points):
        distances, indices = self.kdtree.query(points, self.neighbours_to_use)
        # If one of the coordinates is NaN, the neighbour-query fails.
        # If we don't filter these out, it would result in an IndexError
        # as the kdtree returns an invalid index if it can't find neighbours.
        result = np.ones(len(points)) * float('nan')
        valid = (distances < float('inf')).max(axis=-1)
        result[valid] = np.average(
            self.values[indices[valid]],
            weights=1/np.clip(distances[valid], 1e-6, float('inf')),
            axis=-1)
        return result

class InterpolateAndExtrapolateArray(InterpolateAndExtrapolate):

    def __call__(self, points):
        distances, indices = self.kdtree.query(points, self.neighbours_to_use)
        result = np.ones((len(points), self.values.shape[-1])) * float('nan')
        valid = (distances < float('inf')).max(axis=-1)

        values = self.values[indices[valid]]
        weights = np.repeat(1/np.clip(distances[valid], 1e-6, float('inf')), values.shape[-1]).reshape(values.shape)

        result[valid] = np.average(values, weights=weights, axis=-2)
        return result


class InterpolatingMap(object):
    """Correction map that computes values using inverse-weighted distance
    interpolation.

    The map must be specified as a json translating to a dictionary like this:
        'coordinate_system' :   [[x1, y1], [x2, y2], [x3, y3], [x4, y4], ...],
        'map' :                 [value1, value2, value3, value4, ...]
        'another_map' :         idem
        'name':                 'Nice file with maps',
        'description':          'Say what the maps are, who you are, etc',
        'timestamp':            unix epoch seconds timestamp

    with the straightforward generalization to 1d and 3d.

    The default map name is 'map', I'd recommend you use that.

    For a 0d placeholder map, use
        'points': [],
        'map': 42,
        etc

    """
    data_field_names = ['timestamp', 'description', 'coordinate_system',
                    'name', 'irregular']
    def __init__(self, data, fmt):
        self.log = logging.getLogger('InterpolatingMap')
        self.data = get_resource(data, fmt)

        self.coordinate_system = cs = self.data['coordinate_system']
        if not len(cs):
            self.dimensions = 0
        elif isinstance(cs[0], list):
            if isinstance(cs[0][0], str):
                grid = [np.linspace(left, right, points) for axis, (left, right, points) in cs]
                cs = np.array(np.meshgrid(*grid))
                cs = np.transpose(cs, np.roll(np.arange(len(grid)+1), -1))
                cs = np.array(cs).reshape((-1, len(grid)))
                self.dimensions =len(grid)
            else:
                self.dimensions = len(cs[0])
        else:
            self.dimensions = 1

        self.interpolators = {}
        self.map_names = sorted([k for k in self.data.keys()
                                 if k not in self.data_field_names])
        self.log.debug('Map name: %s' % self.data['name'])
        self.log.debug('Map description:\n    ' +
                       re.sub(r'\n', r'\n    ', self.data['description']))
        self.log.debug("Map names found: %s" % self.map_names)

        for map_name in self.map_names:
            map_data = np.array(self.data[map_name])
            if self.dimensions == 0:
                # 0 D -- placeholder maps which take no arguments
                # and always return a single value
                def itp_fun(positions):
                    return map_data * np.ones_like(positions)
            elif len(map_data.shape) == self.dimensions + 1:
                map_data = map_data.reshape((-1, map_data.shape[-1]))
                itp_fun = InterpolateAndExtrapolateArray(points=np.array(cs),
                                                    values=np.array(map_data))
            else:
                itp_fun = InterpolateAndExtrapolate(points=np.array(cs),
                                                    values=np.array(map_data))

            self.interpolators[map_name] = itp_fun

    def __call__(self, positions, map_name='map'):
        """Returns the value of the map at the position given by coordinates
        :param positions: array (n_dim) or (n_points, n_dim) of positions
        :param map_name: Name of the map to use. Default is 'map'.
        """
        return self.interpolators[map_name](positions)

import numba
@export
@numba.jit(numba.int32(numba.int64[:], numba.int64, numba.int64, numba.int64[:, :]),
           nopython=True)
def find_intervals_below_threshold(w, threshold, holdoff, result_buffer):
    """Fills result_buffer with l, r bounds of intervals in w > threshold.
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

    n_intervals = current_interval      # No +1, as current_interval was incremented also when the last interval closed
    return n_intervals