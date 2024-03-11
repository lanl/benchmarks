#!/bin/bash
# This runs a single case

export MINIEM_IS_KOKKOS_TOOLS="no"

export MINIEM_SIZE=70
export RANKS_PER_DOMAIN=14
export MINIEM_STEPS=1263
sleep 0.2
./run-crossroads-mapcpu.sh

exit 0
