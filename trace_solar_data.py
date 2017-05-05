#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""Prints art in the style of each line one breath"""

import argparse

from scipy.spatial import cKDTree
from numpy import zeros, sin, cos, pi
import numpy as np
# import matplotlib.pyplot as plt
import cairo
# import gtk, gobject

ONE = 0

BACK = 1.

DIST_NEAR_INDICES = np.inf
NUM_NEAR_INDICES = 30
SHIFT_INDICES = 5

INVERSE_DISTANCE_ENABLED = False
MAX_ANGLE_NOISE = 1.0

W = 0.95
PIX_BETWEEN = 11

START_X = (1.-W)*0.5
START_Y = (1.-W)*0.5

X_MIN = 0+START_X/2.0
Y_MIN = 0+START_Y/2.0
X_MAX = 1-START_X
Y_MAX = 1-START_Y

STEP_LENGTH = 0.0

OUTPUT_FOLDER = "output"
FILENAME = 'solar_lines'
FILES = 'data_filenames'
DATA_COLUMN_INDEX = 8


def get_near_indices(tree, xy_points, upper_bound, number_of_points):
    """Returns lists containing nearest points and distances to those points"""
    dist, data_inds = tree.query(xy_points, k=number_of_points,
                                 distance_upper_bound=upper_bound, eps=ONE)

    dist = dist.flatten()
    data_inds = data_inds.flatten()

    sort_inds = np.argsort(data_inds)

    dist = dist[sort_inds]
    data_inds = data_inds[sort_inds]

    return dist, data_inds.flatten()


def alignment(angle, dist):
    """Controls how the distance from nearest line works"""
    distance_x = cos(angle)
    distance_y = sin(angle)

    if INVERSE_DISTANCE_ENABLED:
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
    def __init__(self, n, ysize):
        self.size = n
        self.ysize = ysize
        self.__init_cairo()

    def __init_cairo(self):

        sur = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.size, self.ysize)
        ctx = cairo.Context(sur)
        ctx.scale(self.size, self.ysize)
        ctx.set_source_rgb(BACK, BACK, BACK)
        ctx.rectangle(0, 0, 1, 1)
        ctx.fill()

        self.sur = sur
        self.ctx = ctx

    def line(self, xy_points):
        """Add a line joining the points in xy_points"""
        self.ctx.set_source_rgba(0, 0, 0, 0.6)
        self.ctx.set_line_width(ONE*3.)

        self.ctx.move_to(xy_points[0, 0], xy_points[0, 1])
        for (x_pt, y_pt) in xy_points[1:]:
            self.ctx.line_to(x_pt, y_pt)
        self.ctx.stroke()


def generate_line(start_x, start_y, start_angle, xy_old, angles_old, noise):
    """Generates a line following the old line and introducing noise"""

    if xy_old != []:
        tree = cKDTree(xy_old)
    max_points = len(noise)
    xy_new = zeros((max_points, 2), 'float')
    angles = zeros(max_points, 'float')

    xy_point = np.array([[start_x, start_y]])

    xy_new[0, :] = xy_point
    angles[0] = 0.5*pi
    angle = 0.5*pi

    for i in xrange(1, max_points):
        angle = 0.5*pi
        if xy_old != []:
            dist, inds = get_near_indices(tree, xy_point,
                                          DIST_NEAR_INDICES, NUM_NEAR_INDICES)

            # dist[dist<ONE] = ONE
            dist = dist[SHIFT_INDICES:]
            inds = inds[SHIFT_INDICES:]

            distance_x, distance_y = alignment(angles_old[inds], dist)

            angle = np.arctan2(distance_y, distance_x)
        angle += noise[i]

        xy_next_point = xy_point \
            + np.array([[cos(angle), sin(angle)]])*STEP_LENGTH
        xy_new[i, :] = xy_next_point
        angles[i] = angle

        xy_point = xy_next_point

        if xy_point[0, 0] > X_MAX or xy_point[0, 0] < X_MIN or\
           xy_point[0, 1] > Y_MAX or xy_point[0, 1] < Y_MIN:
            xy_new = xy_new[:i, :]
            angles = angles[:i]
            break

    return xy_new, angles


def draw_image():
    """Draws each line one breath image"""
    with open(FILES, 'r') as data_filenames_fp:
        data_filenames = data_filenames_fp.read().split()

    num_lines = len(data_filenames)
    line_sep = W/num_lines
    size = int(num_lines*PIX_BETWEEN/W)

    test_noise = get_noise_from_file(data_filenames[0],
                                     index=DATA_COLUMN_INDEX)
    global START_X, START_Y, X_MAX, Y_MAX, X_MIN, Y_MIN
    ysize = len(test_noise) + (START_X + (1 - X_MAX))*size

    START_Y = START_X*float(size)/float(ysize)
    Y_MAX = X_MAX*float(size)/float(ysize)
    Y_MIN = X_MIN*float(size)/float(ysize)

    global ONE, STEP_LENGTH
    ONE = 1./size
    STEP_LENGTH = 1./ysize

    render = Render(size, ysize=int(ysize))

    noise = get_noise_from_file(data_filenames[0], index=DATA_COLUMN_INDEX)
    max_points = len(noise)
    line_points, angles = generate_line(START_X, START_Y,
                                        pi*0.5, [], [], noise)

    render.line(line_points)

    for line_num in range(1, num_lines):
        noise = get_noise_from_file(data_filenames[line_num],
                                    index=DATA_COLUMN_INDEX)
        line_points, angles = generate_line(START_X+line_num*line_sep, START_Y,
                                            pi*0.5, line_points, angles, noise)

        render.line(line_points)

        if line_num % 100 == 0:
            render.sur.write_to_png('{:s}_{:d}.png'.format(FILENAME, line_num))

    render.sur.write_to_png('{:s}_final.png'.format(FILENAME))


def main():
    """Main"""
    draw_image()


def get_noise_from_file(data_fname, index=8):
    """Loads data file and creates noise from it"""
    solar_data = np.loadtxt(data_fname, comments=['#', ':'])
    good_indices = solar_data[:, 8] > -400.0
    bad_indices = solar_data[:, 8] < -400.0
    noise_data = solar_data[:, index]
    average = np.mean(noise_data[good_indices])
    noise = (noise_data - average)/max(noise_data - average)
    noise[bad_indices] = 0.0

    noise *= MAX_ANGLE_NOISE

    return noise

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
