#!/usr/bin/env python

import csv
import os

node_counts = (1, 2, 4, 8, 16, 32, 64, 128, 240, 256, 512)
files = ('argsort.dat', 'gather.dat', 'scatter.dat', 'stream.dat')

for f in files:
  dats = [] 
  locales = [] 
  for nl in node_counts:
    # read individual dat files in
    dat_file = os.path.join('ak-perf-{}'.format(nl), f)
    if os.path.isfile(dat_file):
      with open (dat_file, 'r') as dat:
        dats.append(list(csv.reader(dat, delimiter='\t')))
        locales.append(nl)

  if len(dats) == 0:
    continue
  # merge .dat files data into array
  merged_data = []
  header = dats[0][0]
  header[0] = 'Locales '
  merged_data.append(header)
  for dat, nl in zip(dats, locales):
    data = dat[1] # ignore header
    data[0] = nl # replace date with locale 
    merged_data.append(data)

  # write merged data into new .dat file
  with open (f, 'w') as merged_dat:
    writer = csv.writer(merged_dat, delimiter='\t')
    writer.writerows(merged_data)
