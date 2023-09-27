#!/bin/bash
# This submits a full sweep

export NUM_DUPLICATES=8
export APP_REPEAT=8
export SBATCH_OPTS="--core-spec=0 --exclude=nid006219,nid005850,nid003658,nid001813,nid001451,nid002233,nid005892,nid005896,nid003804,nid002065,nid001855,nid005912,nid006402,nid005723,nid006615,nid005851,nid005356"

for (( i=0 ; i<${NUM_DUPLICATES} ; i++ )) ; do

    export SPARTA_PPC=140
    export RANKS_PER_DOMAIN=14
    export SPARTA_RUN=661
    export SPARTA_STATS=10
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh
    export RANKS_PER_DOMAIN=11
    export SPARTA_RUN=555
    export SPARTA_STATS=10
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh
    export RANKS_PER_DOMAIN=7
    export SPARTA_RUN=413
    export SPARTA_STATS=9
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh
    export RANKS_PER_DOMAIN=4
    export SPARTA_RUN=307
    export SPARTA_STATS=7
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh
    export RANKS_PER_DOMAIN=1
    export SPARTA_RUN=202
    export SPARTA_STATS=4
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh

    export SPARTA_PPC=130
    export RANKS_PER_DOMAIN=14
    export SPARTA_RUN=873
    export SPARTA_STATS=20
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh
    export RANKS_PER_DOMAIN=11
    export SPARTA_RUN=723
    export SPARTA_STATS=10
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh
    export RANKS_PER_DOMAIN=7
    export SPARTA_RUN=523
    export SPARTA_STATS=10
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh
    export RANKS_PER_DOMAIN=4
    export SPARTA_RUN=373
    export SPARTA_STATS=8
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh
    export RANKS_PER_DOMAIN=1
    export SPARTA_RUN=224
    export SPARTA_STATS=5
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh

    export SPARTA_PPC=120
    export RANKS_PER_DOMAIN=14
    export SPARTA_RUN=1086
    export SPARTA_STATS=20
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh
    export RANKS_PER_DOMAIN=11
    export SPARTA_RUN=892
    export SPARTA_STATS=20
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh
    export RANKS_PER_DOMAIN=7
    export SPARTA_RUN=634
    export SPARTA_STATS=10
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh
    export RANKS_PER_DOMAIN=4
    export SPARTA_RUN=440
    export SPARTA_STATS=10
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh
    export RANKS_PER_DOMAIN=1
    export SPARTA_RUN=247
    export SPARTA_STATS=5
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh

    export SPARTA_PPC=110
    export RANKS_PER_DOMAIN=14
    export SPARTA_RUN=1298
    export SPARTA_STATS=30
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh
    export RANKS_PER_DOMAIN=11
    export SPARTA_RUN=1060
    export SPARTA_STATS=20
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh
    export RANKS_PER_DOMAIN=7
    export SPARTA_RUN=744
    export SPARTA_STATS=10
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh
    export RANKS_PER_DOMAIN=4
    export SPARTA_RUN=507
    export SPARTA_STATS=10
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh
    export RANKS_PER_DOMAIN=1
    export SPARTA_RUN=270
    export SPARTA_STATS=6
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh

    export SPARTA_PPC=100
    export RANKS_PER_DOMAIN=14
    export SPARTA_RUN=1511
    export SPARTA_STATS=30
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh
    export RANKS_PER_DOMAIN=11
    export SPARTA_RUN=1229
    export SPARTA_STATS=20
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh
    export RANKS_PER_DOMAIN=7
    export SPARTA_RUN=855
    export SPARTA_STATS=20
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh
    export RANKS_PER_DOMAIN=4
    export SPARTA_RUN=574
    export SPARTA_STATS=10
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh
    export RANKS_PER_DOMAIN=1
    export SPARTA_RUN=293
    export SPARTA_STATS=6
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh

    export SPARTA_PPC=64
    export RANKS_PER_DOMAIN=14
    export SPARTA_RUN=2176
    export SPARTA_STATS=50
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh
    export RANKS_PER_DOMAIN=11
    export SPARTA_RUN=1770
    export SPARTA_STATS=40
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh
    export RANKS_PER_DOMAIN=7
    export SPARTA_RUN=1228
    export SPARTA_STATS=20
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh
    export RANKS_PER_DOMAIN=4
    export SPARTA_RUN=822
    export SPARTA_STATS=10
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh
    export RANKS_PER_DOMAIN=1
    export SPARTA_RUN=417
    export SPARTA_STATS=9
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh

    export SPARTA_PPC=24
    export RANKS_PER_DOMAIN=14
    export SPARTA_RUN=6380
    export SPARTA_STATS=100
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh
    export RANKS_PER_DOMAIN=11
    export SPARTA_RUN=5150
    export SPARTA_STATS=100
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh
    export RANKS_PER_DOMAIN=7
    export SPARTA_RUN=3510
    export SPARTA_STATS=80
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh
    export RANKS_PER_DOMAIN=4
    export SPARTA_RUN=2280
    export SPARTA_STATS=50
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh
    export RANKS_PER_DOMAIN=1
    export SPARTA_RUN=1051
    export SPARTA_STATS=20
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh

done

exit 0
