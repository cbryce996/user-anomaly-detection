#!/bin/bash

python ../src/parse.py && python ../src/dbscan.py && python ../src/kmeans.py && python ../src/spectral.py