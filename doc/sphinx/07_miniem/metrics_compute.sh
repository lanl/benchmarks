#!/bin/bash

export DIR_BASE=` pwd -P `
find . -name "output-srun*.log" -type f -print0 \
    | xargs -0 -I file ./miniem_fom.py -f file

exit 0
