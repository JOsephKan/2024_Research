#!/bin/sh

case=("CNTL" "NCRF" "NSC")

for c in ${case[@]}; do
    echo "Running LRF_heating.py for $c"
    python LRF_heating_no_filt.py $c
done