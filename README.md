# Each Line One Breath (Solar Edition)

### Running parameter studies
the script `./parameter_study.sh` outputs a list of executable lines that run through many different parameters. This can be made parallel using `xargs` in the following way
```
./parameter_study.sh | xargs -I CMD --max-procs=4 bash -c CMD
```

### Creating noise data
By running something like 
```
./extract_data.sh -i 7 -t goes goes/2016/*.csv > noise_data
```
and then passing the noise data file to the tracer, the noise data will be used to generate the piece.
