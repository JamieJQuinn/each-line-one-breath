#!/usr/bin/env bash

DATE_PREV=$(date -I -d '10 minutes ago')
TIME_PREV=$(date +%T -d '10 minutes ago')
DATE_NOW=$(date -I)
TIME_NOW=$(date +%T)

DATE_START="2015-01-01"
TIME_START="00:00:00"
DATE_END="2016-01-01"
TIME_END="00:00:00"

url="http://www.srl.caltech.edu/ACE/ASC/level2/new/ACEL2Server.cgi?datasetID=swepam_level2_data_64sec&PARAM=x_dot_GSE&sd="$DATE_START"&st="$TIME_START"&ed="$DATE_END"&et="$TIME_END"&dataformat=TEXT&nonint=1"

echo $url

wget $url -O solar_wind_x_now.txt
