#!/bin/sh

exp=("CNTL" "NCRF" "NSC")

for i in "${exp[@]}"; do
    echo experiment $i start
    python raw_comp.py $i
    echo experiment $i end
done