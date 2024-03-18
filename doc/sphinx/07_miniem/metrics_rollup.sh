#!/bin/bash

find $1 -name "output-*run*.csv" -type f -print0 \
    | xargs -0 -I file cat file \
    | sort \
    | uniq \
    | sort -V

exit 0
