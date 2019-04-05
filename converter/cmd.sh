#!/bin/bash
wget https://csr.lanl.gov/data/unified-host-network-dataset-2017/netflow/netflow_day-02.bz2
bunzip ./netflow_day-02.bz2
split -l 10000000 -d ./netflow_day-02
python3 csv2hdf.py --formats-file=lanl_format.py --format=lanl --outdir=./hdf5 .netflow_day-02-??
