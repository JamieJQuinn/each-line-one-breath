#!/usr/bin/env bash

if [ "$1" != '' ]
then
  # Get specific date
  url="ftp://sohoftp.nascom.nasa.gov/sdb/goes/ace/daily/"$1"_ace_swepam_1m.txt"
else
  # Get daily
  url="ftp://sohoftp.nascom.nasa.gov/sdb/goes/ace/daily/ace_swepam_1m.txt"
fi

wget $url
