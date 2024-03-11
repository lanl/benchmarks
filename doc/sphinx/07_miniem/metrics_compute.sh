#!/bin/bash

export DIR_BASE=` pwd -P `
find $1 -name "output-srun*.log" -type f -print0 \
    | xargs -0 -I file ./miniem_fom2.py -f file

exit 0
