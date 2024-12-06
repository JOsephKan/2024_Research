#!/bin/sh

exp_list=("CNTL" "NCRF" "NSC")

for exp in ${exp_list[@]}; do
    echo "Start LRF construct for $exp"
    python LRF_construct.py $exp
    echo "Finish LRF construct for $exp"
done