#!/usr/bin/env bash

for year in $(seq 2001 2015)
do
  cd $year
  ../get_bulk_speed.sh "$year"0101 "$year"1231
  cd ..
done
