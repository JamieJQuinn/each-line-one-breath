# Each Line One Breath (Solar Edition)

### Running parameter studies
the script `./parameter_study.sh` outputs a list of executable lines that run through many different parameters. This can be made parallel using `xargs` in the following way
```
./parameter_study.sh | xargs -I CMD --max-procs=4 bash -c CMD
```

### Creating list of input files
By running something like 
```
find swepam/2016/* > input_files
```
the tracer will draw the data from each line in `input_files` sequentially.
