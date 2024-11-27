#!/bin/sh

for y in {2006..2017}; do
  python Itp.py ${y}
done
