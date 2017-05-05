#!/usr/bin/env bash

for inverse_enabled in '' '--inverse_distance_enabled'; do
  for num_nearest_neighbours in 5 10 20 30; do
    for shift_indices in 1 3 5; do
      for max_angle in 0.1 0.5 1.0 2.0 5.0; do
        echo "./trace_solar_data.py $inverse_enabled\
        --num_near_indices=$num_nearest_neighbours\
        --shift_indices=$shift_indices --max_angle_noise=$max_angle"
      done
    done
  done
done
