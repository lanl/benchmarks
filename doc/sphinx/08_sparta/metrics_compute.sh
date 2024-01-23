#!/bin/bash

find . -name "log.sparta" -type f -print0 \
    | xargs -0 -P 8 -I mfl ./sparta_fom.py --file mfl --numRanksPerNode 112

exit 0
