#!/bin/sh

cases=("CNTL")

for case in ${cases[@]}; do
    echo "Processing $case"
    python EOF.py $case
done