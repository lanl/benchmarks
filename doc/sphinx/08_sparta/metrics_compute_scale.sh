#!/bin/bash

compute_fom()
{
    ./sparta_fom.py --file $1 --numRanksPerNode 112
}
export -f compute_fom

find $1 -name "log.sparta" -type f -print0 \
    | xargs -0 -P 8 -I mfl bash -c "compute_fom mfl"

exit 0
