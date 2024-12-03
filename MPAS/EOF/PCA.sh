#!/bin/sh

cases=("CNTL" "NCRF" "NSC")

for case in ${cases[@]}; do
    echo "Processing $case"
    python PCA.py $case
done