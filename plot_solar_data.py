#!/usr/bin/env python2

import numpy as np
import matplotlib.pyplot as plt
import sys

def get_noise_from_file(data_fname, index=8):
    """Loads data file and creates noise from it"""
    solar_data = np.loadtxt(data_fname, comments=['#', ':'])
    good_indices = solar_data[:, 8] > -400.0
    bad_indices = solar_data[:, 8] < -400.0
    noise_data = solar_data[:, index]
    max_noise_data = max(noise_data[good_indices])
    min_noise_data = min(noise_data[good_indices])
    noise = 2*(noise_data - min_noise_data)\
        / (max_noise_data - min_noise_data) - 1
    noise[bad_indices] = 0.0

    plt.plot(noise)
    plt.show()

get_noise_from_file(sys.argv[1])
