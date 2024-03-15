#!/bin/bash

compute_fom()
{
    num_ranks=` grep Running $1 | awk '{print $3}' `
    ./sparta_fom.py --file $1 --numRanksPerNode $num_ranks
}
export -f compute_fom

find $1 -name "log.sparta" -type f -print0 \
    | xargs -0 -P 8 -I mfl bash -c "compute_fom mfl"

exit 0
