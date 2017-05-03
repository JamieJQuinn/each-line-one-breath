#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""Prints art in the style of each line one breath"""

from scipy.spatial import cKDTree
from numpy.random import random, shuffle, normal
from numpy import zeros, sin, cos, pi
import numpy as np
import cairo
# import gtk, gobject

np.random.seed(1)

SIZE = 2000
ONE = 1./SIZE

BACK = 1.

X_MIN = 0+10*ONE
Y_MIN = 0+10*ONE
X_MAX = 1-10*ONE
Y_MAX = 1-10*ONE

DIST_NEAR_INDICES = np.inf
NUM_NEAR_INDICES = 30
SHIFT_INDICES = 5

W = 0.9
PIX_BETWEEN = 11

START_X = (1.-W)*0.5
START_Y = (1.-W)*0.5

NUMMAX = int(2*SIZE)
NUM_LINES = int(SIZE*W/PIX_BETWEEN)
H = W/NUM_LINES

FILENAME = 'tt_brownianbridge'

TURTLE_ANGLE_NOISE = np.pi*0.1
INIT_TURTLE_ANGLE_NOISE = 0


def myrandom(size):
    """Returns array of size size of random numbers"""
    # res = normal(size=size)

    # res = 1.-2.*random(size=size)

    # almost but not entirely unlike a brownian bridge
    rnd = 1.-2.*random(size=size/2)
    res = np.concatenate((rnd, -rnd))
    shuffle(res)
    return res


def turtle(starting_angle, starting_x, starting_y, steps):
    """Returns turtle shape"""
    xy_line = zeros((steps, 2), 'float')
    angles = zeros(steps, 'float')

    xy_line[0, 0] = starting_x
    xy_line[0, 1] = starting_y
    angles[0] = starting_angle
    angle = starting_angle

    noise = myrandom(size=steps)*INIT_TURTLE_ANGLE_NOISE
    for k in xrange(1, steps):
        x_new = xy_line[k-1, 0] + cos(angle)*ONE
        y_new = xy_line[k-1, 1] + sin(angle)*ONE
        xy_line[k, 0] = x_new
        xy_line[k, 1] = y_new
        angles[k] = angle
        angle += noise[k]

        if x_new > X_MAX or x_new < X_MIN or y_new > Y_MAX or y_new < Y_MIN:
            xy_line = xy_line[:k, :]
            angles = angles[:k]
            break

    return angles, xy_line


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

    # inverse proporional distance scale
    # distance_x = np.sum(distance_x/dist)
    # distance_y = np.sum(distance_y/dist)

    # linear distance scale
    distance_x = np.sum(distance_x*(1.-dist))
    distance_y = np.sum(distance_y*(1.-dist))

    total_distance = (distance_x*distance_x+distance_y*distance_y)**0.5

    return distance_x/total_distance, distance_y/total_distance


class Render(object):
    """Renders lines"""
    def __init__(self, n):
        self.size = n
        self.__init_cairo()

    def __init_cairo(self):

        sur = cairo.ImageSurface(cairo.FORMAT_ARGB32, self.size, self.size)
        ctx = cairo.Context(sur)
        ctx.scale(self.size, self.size)
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


def main():
    """Main function"""
    render = Render(SIZE)

    angle = pi*0.5

    angle, xy_line = turtle(angle, START_X, START_Y, NUMMAX)

    render.line(xy_line)

    xy_old = xy_line
    angles_old = angle

    tree = cKDTree(xy_old)

    for line_num in range(1, NUM_LINES):
        print(line_num, NUM_LINES)

        xy_new = zeros((NUMMAX, 2), 'float')
        angle_new = zeros(NUMMAX, 'float')

        # x = X_MIN + random()*(X_MAX-X_MIN)
        # y = Y_MIN + random()*(Y_MAX-Y_MIN)

        xy_last_point = np.array([[START_X+line_num*H, START_Y]])
        angle_last_point = 0.5*pi

        xy_new[0, :] = xy_last_point
        angle_new[0] = angle_last_point

        noise = myrandom(size=NUMMAX)*TURTLE_ANGLE_NOISE

        for i in xrange(1, NUMMAX):
            dist, inds = get_near_indices(tree, xy_last_point,
                                          DIST_NEAR_INDICES, NUM_NEAR_INDICES)

            # dist[dist<ONE] = ONE
            dist = dist[SHIFT_INDICES:]
            inds = inds[SHIFT_INDICES:]

            distance_x, distance_y = alignment(angles_old[inds], dist)

            angle = np.arctan2(distance_y, distance_x)
            angle += noise[i]

            xy_next_point = xy_last_point \
                + np.array([[cos(angle), sin(angle)]])*ONE
            xy_new[i, :] = xy_next_point
            angle_new[i] = angle

            xy_last_point = xy_next_point

            if xy_last_point[0, 0] > X_MAX or xy_last_point[0, 0] < X_MIN or\
               xy_last_point[0, 1] > Y_MAX or xy_last_point[0, 1] < Y_MIN:
                xy_new = xy_new[:i, :]
                angle_new = angle_new[:i]
                break

        render.line(xy_new)

        # replace all old nodes
        xy_old = xy_new
        angles_old = angle_new

        tree = cKDTree(xy_old)

        if not line_num % 100:

            render.sur.write_to_png('{:s}_{:d}.png'.format(FILENAME, line_num))

    render.sur.write_to_png('{:s}_final.png'.format(FILENAME))


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
