#!/usr/bin/env bash

if [ "$2" != '' ]
then
  d=$1
  until [[ $d > "$2" ]]; do
    mkdir -p $(date +%Y -d"$d")
    if [ ! -f $(date +%Y -d"$d")/g15_xrs_2s_"$d"_"$d".csv ]; then
      wget "https://satdat.ngdc.noaa.gov/sem/goes/data/new_full/"$(date +%Y -d"$d")"/"$(date +%m -d "$d")"/goes15/csv/g15_xrs_2s_"$d"_"$d".csv" -P$(date +%Y -d"$d")
    fi
    d=$(date +%Y%m%d -d "$d + 1 day")
  done
elif [ "$1" != '' ]
then
  # Get specific date
  d=$1
  mkdir -p $(date +%Y -d"$d")
  wget "https://satdat.ngdc.noaa.gov/sem/goes/data/new_full/"$(date +%Y -d"$d")"/"$(date +%m -d "$d")"/goes15/csv/g15_xrs_2s_"$d"_"$d".csv" -P$(date +%Y -d"$d")
else
  echo "No date specified"
fi
