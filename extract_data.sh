#!/usr/bin/env bash

usage () { echo "Usage: $0 -i <index> -t <data_type> <input files>"; }

while getopts "t:i:" flag; do
  case "${flag}" in
    # Specify index
    i) index=${OPTARG} ;;
    # Specify data type
    t) data_type=${OPTARG} ;;
  esac
done

if [ ! "$index"  ] || [ ! "$data_type" ]; then
  usage
  exit 1
fi

if [ "$data_type" == "goes" ]; then
  awk -F ',' '{print $'"$index"'}' "${@:$OPTIND}" | grep '\d\.\d{4}'
elif [ "$data_type" == "swepam" ]; then
  awk '/^[^#:]/ {print $'"$index"'}' "${@:$OPTIND}"
fi
