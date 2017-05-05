#!/usr/bin/env bash

if [ "$2" != '' ]
then
  d=$1
  until [[ $d > "$2" ]]; do
    wget "ftp://sohoftp.nascom.nasa.gov/sdb/goes/ace/daily/"$d"_ace_swepam_1m.txt"
    d=$(date +%Y%m%d -d "$d + 1 day")
  done
elif [ "$1" != '' ]
then
  # Get specific date
  wget "ftp://sohoftp.nascom.nasa.gov/sdb/goes/ace/daily/"$1"_ace_swepam_1m.txt"
else
  # Get daily
  wget "ftp://sohoftp.nascom.nasa.gov/sdb/goes/ace/daily/ace_swepam_1m.txt"
fi
