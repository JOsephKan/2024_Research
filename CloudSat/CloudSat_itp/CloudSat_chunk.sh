#!/bin/bash

for y in {2006..2017}; do
  for d in {1..365}; do
    date=$(printf "%03d" "$d")
    python CloudSat_chunk.py "$y" "$date"
  done
done