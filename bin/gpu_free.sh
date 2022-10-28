#!/bin/bash

# check whether GPU device is used or not
# rct 1 if GPU $1 is used, else 0

GPU=$1
COUNT=$(nvidia-smi |cut -c6- | grep -P "\d +\d+ +[C] +[\S]+ +\d+[KMG]iB " | gawk '{print $1}' | grep -c $GPU)

if [ $COUNT -eq 1 ]; then
    echo "GPU $1 used"
    exit 1
else
    echo "GPU $1 free"
    exit 0
fi

