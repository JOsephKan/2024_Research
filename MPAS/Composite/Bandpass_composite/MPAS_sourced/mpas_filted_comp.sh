#!/bin/sh

exps=("CNTL" "NCRF" "NSC")

for exp in ${exps[@]}; do
    echo start processing experiment $exp
    python3 mpas_filted_comp.py $exp
    echo finish processing experiment $exp
done