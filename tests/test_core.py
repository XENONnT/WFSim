import numpy as np
from hypothesis import strategies, given, example, settings
import wfsim


@settings(max_examples=100, deadline=None)
@given(strategies.integers(min_value=0, max_value=1_000),
       strategies.integers(min_value=0, max_value=4),
       strategies.integers(min_value=0, max_value=1_000))
@example(data_length=101, n_channels=4, noise_data_length=1000)
def test_noise(data_length, n_channels, noise_data_length):
    if data_length <= 0 or noise_data_length <= 0:
        return
    if n_channels > 4 or n_channels < 0:
        return

    data = np.random.randint(-100, 0, size=(n_channels, data_length))
    channel_mask = np.array([(False, 9223372036399775857, -454999850),
                             (False, 9223372036399775857, -454999850),
                             (False, 9223372036399775857, -454999850),
                             (True, n_channels - 1, noise_data_length - n_channels),
                             ],
                            dtype=[('mask', '?'), ('left', '<i8'), ('right', '<i8')])
    channel_mask = channel_mask[:n_channels]
    noise_data = np.random.randint(-10, 10, size=noise_data_length)

    rd = wfsim.RawData
    function = rd.add_noise

    function(data, channel_mask, noise_data, noise_data_length)