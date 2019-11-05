#!/bin/bash

echo ---- argsort ----
./argsort.py -n 10000000 localhost 5555
echo ---- gather ----
./gather.py localhost 5555
echo ---- reduce ----
./reduce.py -t 10 localhost 5555
echo ---- scan ----
./scan.py localhost 5555
echo ---- scatter ----
./scatter.py localhost 5555
echo ---- stream ----
./stream.py localhost 5555

