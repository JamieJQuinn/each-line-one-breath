#!/usr/bin/env python2

import argparse
import numpy as np
import matplotlib.pyplot as plt


def calculate_noise(noise_data, max_angle_noise):
    """Returns normalised and centred noise data"""
    good_data = noise_data[noise_data >= 0]
    average = np.median(good_data)
    noise = np.zeros_like(noise_data)
    if (good_data != 0.0).any():
        noise = (noise_data - average)/max(np.fabs(good_data - average))
    noise[noise_data < 0] = 0.0
    noise *= max_angle_noise

    return noise


def get_noise_from_file(config):
    """Loads noise from given file"""
    noise_data = np.genfromtxt(config["noise_file"])
    noise_data = noise_data[:-(len(noise_data) % config["points_per_line"])]

    if config["use_global_average"]:
        noise = calculate_noise(noise_data, config["max_angle_noise"])
    else:
        noise = np.concatenate([calculate_noise(array, config["max_angle_noise"])
                                for array in
                                np.split(noise_data, config["points_per_line"])])

    return noise


parser = argparse.ArgumentParser(description='Plots solar data')
parser.add_argument('--raw', action='store_true',
                    help='Plot raw data')
parser.add_argument('--use_global_average', action='store_true',
                    help='Calc average over all noise instead of per line')
parser.add_argument('noise_file')
parser.add_argument('--points_per_line', required=True, type=int)
parser.add_argument('--max_angle_noise', type=float, default=1.0,
                    help='Maximum angle in degrees')
args = parser.parse_args()

config = {}
config["points_per_line"] = args.points_per_line
config["noise_file"] = args.noise_file
config["use_global_average"] = args.use_global_average
config["max_angle_noise"] = args.max_angle_noise

if args.raw:
    noise_data = np.genfromtxt(config["noise_file"])

    plt.plot(noise_data)
else:
    noise = get_noise_from_file(config)

    plt.plot(noise)

plt.show()
