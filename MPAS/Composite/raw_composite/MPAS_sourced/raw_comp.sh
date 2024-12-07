#!/bin/sh

cases=("CNTL" "NCRF" "NSC")

for case in ${cases[@]}; do
    echo "Processing $case"
    python raw_comp.py $case
done