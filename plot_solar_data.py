#!/usr/bin/env python2

import argparse
import numpy as np
import matplotlib.pyplot as plt


def get_noise_from_file(data_fname, index=8):
    """Loads data file and creates noise from it"""
    noise_data = parse_data_from_file(fname)
    good_data = noise_data[noise_data > 0]
    # average = np.mean(noise_data[good_indices])
    average = np.median(good_data)
    max_noise = max(np.fabs(good_data - average))
    noise = (noise_data - average)/max_noise
    noise[noise_data < 0] = 0.0
    percentile = np.percentile(noise, 90)
    noise = noise/percentile

    return noise


def parse_data_from_file(fname, index=8):
    data = np.loadtxt(fname, comments=['#', ':'])
    return data[:, index]


parser = argparse.ArgumentParser(description='Plots solar data')
parser.add_argument('--raw', action='store_true',
                    help='Plot raw data')
parser.add_argument('fnames', nargs='+',
                    help='Data files to plot')

args = parser.parse_args()

if args.raw:
    solar_data = np.concatenate([parse_data_from_file(fname) for fname in args.fnames])
    good_data = solar_data[np.logical_and(solar_data > 0, solar_data < 800)]

    plt.plot(good_data)
else:
    total_noise = np.concatenate([get_noise_from_file(fname) for fname in
                                  args.fnames])
    plt.plot(total_noise)

plt.show()
