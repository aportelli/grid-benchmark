#!/usr/bin/env bash

rocm-smi -a --csv | tail -n +4 | head -n 1 | sed 's/^/epoch,/'
interval=1
next=$(date +%s.%N)
while true; do
    rocm-smi -a --csv | tail -n +5 | head -n -1 | sed "s/^/$(date '+%s'),/"
    next=$(awk -v t="$next" -v i="$interval" 'BEGIN{printf "%.9f\n", t + i}')
    now=$(date +%s.%N)
    sleep_time=$(awk -v n="$next" -v c="$now" 'BEGIN{d=n-c; if(d<0) d=0; printf "%.9f\n", d}')
    sleep "$sleep_time"
done
