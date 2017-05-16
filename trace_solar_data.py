#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""Prints art in the style of each line one breath"""

import argparse
import json

from scipy.spatial import cKDTree
from numpy import zeros, sin, cos, pi
import numpy as np
# import matplotlib.pyplot as plt
import cairo
# import gtk, gobject


def get_near_indices(tree, xy_points, upper_bound, number_of_points, one):
    """Returns lists containing nearest points and distances to those points"""
    dist, data_inds = tree.query(xy_points, k=number_of_points,
                                 distance_upper_bound=upper_bound, eps=one)

    dist = dist.flatten()
    data_inds = data_inds.flatten()

    sort_inds = np.argsort(data_inds)

    dist = dist[sort_inds]
    data_inds = data_inds[sort_inds]

    return dist, data_inds.flatten()


def alignment(angle, dist, use_inverse_distance):
    """Controls how the distance from nearest line works"""
    distance_x = cos(angle)
    distance_y = sin(angle)

    if use_inverse_distance:
        # inverse proporional distance scale
        distance_x = np.sum(distance_x/dist)
        distance_y = np.sum(distance_y/dist)
    else:
        # linear distance scale
        distance_x = np.sum(distance_x*(1.-dist))
        distance_y = np.sum(distance_y*(1.-dist))

    total_distance = (distance_x*distance_x+distance_y*distance_y)**0.5

    return distance_x/total_distance, distance_y/total_distance


class Render(object):
    """Renders lines"""
    def __init__(self, xsize, ysize, background_colour, line_width):
        self.xsize = xsize
        self.ysize = ysize
        self.line_width = line_width
        self.__init_cairo(background_colour)

    def __init_cairo(self, background_colour):
        sur = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.xsize, self.ysize)
        ctx = cairo.Context(sur)
        ctx.scale(self.xsize, self.ysize)
        ctx.set_source_rgb(background_colour,
                           background_colour, background_colour)
        ctx.rectangle(0, 0, 1, 1)
        ctx.fill()

        self.sur = sur
        self.ctx = ctx

    def line(self, xy_points, colour=(0, 0, 0)):
        """Add a line joining the points in xy_points"""
        self.ctx.set_source_rgba(colour[0], colour[1], colour[2], 0.6)
        self.ctx.set_line_width(self.line_width)

        self.ctx.move_to(xy_points[0, 0], xy_points[0, 1])
        for (x_pt, y_pt) in xy_points[1:]:
            self.ctx.line_to(x_pt, y_pt)
        self.ctx.stroke()


def generate_line(config, xy_old, angles_old, noise, is_first):
    """Generates a line following the old line and introducing noise"""

    max_points = len(noise)

    if not is_first:
        tree = cKDTree(xy_old)
    xy_new = zeros((max_points, 2), 'float')
    angles = zeros(max_points, 'float')

    xy_point = np.array([[config["xy_start"][0], config["xy_start"][1]]])

    xy_new[0, :] = xy_point
    angles[0] = 0.5*pi

    for i in xrange(1, max_points):
        if not is_first:
            dist, inds = get_near_indices(tree, xy_point,
                                          np.inf, config["num_near_indices"],
                                          config["one"])

            # dist[dist<ONE] = ONE
            dist = dist[config["num_shift_indices"]:]
            inds = inds[config["num_shift_indices"]:]

            distance_x, distance_y = alignment(angles_old[inds], dist,
                                               config["use_global_average"])

            angles[i] += np.arctan2(distance_y, distance_x)
        else:
            angles[i] = 0.5*pi

        angles[i] += noise[i]

        xy_next_point = xy_point\
            + np.array([[cos(angles[i]), sin(angles[i])]])\
            * config["step_length"]
        xy_new[i, :] = xy_next_point

        xy_point = xy_next_point

        if xy_point[0, 0] > config["xy_max"][0] or\
           xy_point[0, 0] < config["xy_min"][0] or\
           xy_point[0, 1] > config["xy_max"][1] or\
           xy_point[0, 1] < config["xy_min"][1]:
            xy_new = xy_new[:i, :]
            angles = angles[:i]
            break

    return xy_new, angles


def split_uneven_array(array, N):
    """Split uneven array into """
    i = 1
    while i*N < len(array):
        yield array[(i-1)*N:i*N]
        i += 1

    yield array[(i-1)*N:]


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
        noise = np.concatenate(
            [calculate_noise(array, config["max_angle_noise"]) for array in
             np.split(noise_data, config["points_per_line"])])

    return noise


def draw_image(config):
    """Draws each line one breath image"""

    noise = get_noise_from_file(config)

    num_lines = len(noise)/config["points_per_line"]
    xsize = int((num_lines-1)*config["pixels_per_line"]/config["inner_width"])
    ysize = int(config["points_per_line"]/config["inner_width"])
    config["one"] = 1./xsize
    config["step_length"] = 1./ysize
    render = Render(xsize, ysize, 0.9, 3*config["one"])

    config["xy_start"] = [(1-config["inner_width"])/2.0,
                          (1-config["inner_width"])/2.0]
    config["xy_min"] = [config["xy_start"][0]/2.0,
                        config["xy_start"][1]/2.0]
    config["xy_max"] = [1-config["xy_start"][0]/2.0,
                        1-config["xy_start"][1]/2.0]

    print config

    line_sep = config["inner_width"]/num_lines
    xy_points = np.zeros((config["points_per_line"], 2))
    angles = np.zeros(config["points_per_line"])
    for line_num, noise_array in enumerate(
            np.split(noise, num_lines)):
        if line_num % 100 == 0:
            print line_num+1, "/", num_lines
            if not config["print_final"]:
                render.sur.write_to_png('{:s}_{:d}.png'.format(
                    config["output"], line_num))

        xy_points, angles = generate_line(config, xy_points, angles,
                                          noise_array, line_num == 0)

        config["xy_start"][0] += line_sep

        colour = (0, 0, 0)
        # if line_num % 24 == 0:
            # colour = (255, 0, 0)
        # elif line_num == 24*7 + 16:
            # colour = (0, 0, 255)

        if colour:
            render.line(xy_points, colour)
        else:
            render.line(xy_points)

    render.sur.write_to_png('{:s}.png'.format(config["output"]))


def parse_arguments():
    """Parse... arguments"""
    with open("config.json", 'r') as config_file:
        config = json.load(config_file)

    parser = argparse.ArgumentParser(description='Renders solar data as each\
                                     line one breath')

    parser.add_argument('--num_near_indices', type=int, default=30,
                        help='Number of nearest neighbours')
    parser.add_argument('--num_shift_indices', type=int, default=5,
                        help='Number of nearest neighbours to ignore')
    parser.add_argument('--max_angle_noise', type=float, default=1.0,
                        help='Maximum angle in degrees')
    parser.add_argument('--use_inverse_distance', action='store_true',
                        help='Switch on inverse distance as opposed to linear')
    parser.add_argument('--use_global_average', action='store_true',
                        help='Calc average over all noise instead of per line')
    parser.add_argument('--output', default='solar_breath')
    parser.add_argument('--noise_file', required=True)
    parser.add_argument('--points_per_line', required=True, type=int)
    parser.add_argument('--print_final', action='store_true')

    args = parser.parse_args()
    config["num_near_indices"] = args.num_near_indices
    config["num_shift_indices"] = args.num_shift_indices
    config["use_inverse_distance"] = args.use_inverse_distance
    config["use_global_average"] = args.use_global_average
    config["max_angle_noise"] = args.max_angle_noise
    config["output"] = args.output
    config["points_per_line"] = args.points_per_line
    config["noise_file"] = args.noise_file
    config["print_final"] = args.print_final

    return config


def main():
    """Main"""
    config = parse_arguments()
    draw_image(config)


if __name__ == '__main__':
    PROFILING_ENABLED = False
    if PROFILING_ENABLED:
        import pstats
        import cProfile
        OUT = 'profile'
        PROFILE_FILENAME = '{:s}.profile'.format(OUT)
        cProfile.run('main()', PROFILE_FILENAME)
        pstats.Stats(PROFILE_FILENAME).strip_dirs().sort_stats('cumulative')\
            .print_stats()
    else:
        main()
