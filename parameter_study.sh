#!/usr/bin/env bash

for max_angle in 0.02 0.0175 0.015 0.0125; do
  echo "./trace_solar_data.py --noise_file noise_data --points_per_line=12671\
  --max_angle_noise=$max_angle --output=$max_angle --print_final"
done
