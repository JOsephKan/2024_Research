#!/bin/bash

for y in {2007..2008}; do
  for d in $(seq 1 365); do
    date=$(printf "%03d" "$d") # Ensure day (d) is padded to 3 digits
    python CloudSat_chunk.py "$y" "$date"
  done
done


