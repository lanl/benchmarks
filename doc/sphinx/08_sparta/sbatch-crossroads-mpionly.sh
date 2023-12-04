#!/bin/bash
# This submits a full sweep

export NUM_DUPLICATES=4
export APP_REPEAT=8
# export SBATCH_OPTS="--core-spec=0 --exclude=nid006219,nid005850,nid003658,nid001813,nid001451,nid002233,nid005892,nid005896,nid003804,nid002065,nid001855,nid005912,nid006402,nid005723,nid006615,nid005851,nid005356"
export SBATCH_OPTS="--core-spec=0 --partition=hbm"

for (( i=0 ; i<${NUM_DUPLICATES} ; i++ )) ; do

    export APP_NAME="spa_crossroads_omp_spr"
    export SPARTA_PPC=130
    export RANKS_PER_DOMAIN=14
    export SPARTA_RUN=1356
    export SPARTA_STATS=30
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh
    export RANKS_PER_DOMAIN=11
    export SPARTA_RUN=1377
    export SPARTA_STATS=30
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh
    export RANKS_PER_DOMAIN=7
    export SPARTA_RUN=1149
    export SPARTA_STATS=20
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh
    export RANKS_PER_DOMAIN=4
    export SPARTA_RUN=757
    export SPARTA_STATS=10
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh
    export RANKS_PER_DOMAIN=1
    export SPARTA_RUN=226
    export SPARTA_STATS=5
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh

    export SPARTA_PPC=120
    export RANKS_PER_DOMAIN=14
    export SPARTA_RUN=1413
    export SPARTA_STATS=30
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh
    export RANKS_PER_DOMAIN=11
    export SPARTA_RUN=1419
    export SPARTA_STATS=30
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh
    export RANKS_PER_DOMAIN=7
    export SPARTA_RUN=1212
    export SPARTA_STATS=20
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh
    export RANKS_PER_DOMAIN=4
    export SPARTA_RUN=794
    export SPARTA_STATS=10
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh
    export RANKS_PER_DOMAIN=1
    export SPARTA_RUN=242
    export SPARTA_STATS=5
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh

    export SPARTA_PPC=110
    export RANKS_PER_DOMAIN=14
    export SPARTA_RUN=1496
    export SPARTA_STATS=30
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh
    export RANKS_PER_DOMAIN=11
    export SPARTA_RUN=1512
    export SPARTA_STATS=30
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh
    export RANKS_PER_DOMAIN=7
    export SPARTA_RUN=1297
    export SPARTA_STATS=30
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh
    export RANKS_PER_DOMAIN=4
    export SPARTA_RUN=846
    export SPARTA_STATS=20
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh
    export RANKS_PER_DOMAIN=1
    export SPARTA_RUN=262
    export SPARTA_STATS=6
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh

    export SPARTA_PPC=100
    export RANKS_PER_DOMAIN=14
    export SPARTA_RUN=1628
    export SPARTA_STATS=30
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh
    export RANKS_PER_DOMAIN=11
    export SPARTA_RUN=1642
    export SPARTA_STATS=30
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh
    export RANKS_PER_DOMAIN=7
    export SPARTA_RUN=1416
    export SPARTA_STATS=30
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh
    export RANKS_PER_DOMAIN=4
    export SPARTA_RUN=921
    export SPARTA_STATS=20
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh
    export RANKS_PER_DOMAIN=1
    export SPARTA_RUN=287
    export SPARTA_STATS=6
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh

    export SPARTA_PPC=64
    export RANKS_PER_DOMAIN=14
    export SPARTA_RUN=2698
    export SPARTA_STATS=60
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh
    export RANKS_PER_DOMAIN=11
    export SPARTA_RUN=2719
    export SPARTA_STATS=60
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh
    export RANKS_PER_DOMAIN=7
    export SPARTA_RUN=2283
    export SPARTA_STATS=50
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh
    export RANKS_PER_DOMAIN=4
    export SPARTA_RUN=1458
    export SPARTA_STATS=30
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh
    export RANKS_PER_DOMAIN=1
    export SPARTA_RUN=440
    export SPARTA_STATS=10
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh

    export SPARTA_PPC=24
    export RANKS_PER_DOMAIN=14
    export SPARTA_RUN=6659
    export SPARTA_STATS=100
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh
    export RANKS_PER_DOMAIN=11
    export SPARTA_RUN=6169
    export SPARTA_STATS=100
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh
    export RANKS_PER_DOMAIN=7
    export SPARTA_RUN=5186
    export SPARTA_STATS=100
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh
    export RANKS_PER_DOMAIN=4
    export SPARTA_RUN=3363
    export SPARTA_STATS=80
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh
    export RANKS_PER_DOMAIN=1
    export SPARTA_RUN=1039
    export SPARTA_STATS=20
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh

    export APP_NAME="spa_crossroads_omp_skx"
    export SPARTA_PPC=130
    export RANKS_PER_DOMAIN=14
    export SPARTA_RUN=1356
    export SPARTA_STATS=30
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh
    export RANKS_PER_DOMAIN=11
    export SPARTA_RUN=1377
    export SPARTA_STATS=30
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh
    export RANKS_PER_DOMAIN=7
    export SPARTA_RUN=1149
    export SPARTA_STATS=20
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh
    export RANKS_PER_DOMAIN=4
    export SPARTA_RUN=757
    export SPARTA_STATS=10
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh
    export RANKS_PER_DOMAIN=1
    export SPARTA_RUN=226
    export SPARTA_STATS=5
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh

    export SPARTA_PPC=120
    export RANKS_PER_DOMAIN=14
    export SPARTA_RUN=1413
    export SPARTA_STATS=30
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh
    export RANKS_PER_DOMAIN=11
    export SPARTA_RUN=1419
    export SPARTA_STATS=30
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh
    export RANKS_PER_DOMAIN=7
    export SPARTA_RUN=1212
    export SPARTA_STATS=20
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh
    export RANKS_PER_DOMAIN=4
    export SPARTA_RUN=794
    export SPARTA_STATS=10
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh
    export RANKS_PER_DOMAIN=1
    export SPARTA_RUN=242
    export SPARTA_STATS=5
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh

    export SPARTA_PPC=110
    export RANKS_PER_DOMAIN=14
    export SPARTA_RUN=1496
    export SPARTA_STATS=30
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh
    export RANKS_PER_DOMAIN=11
    export SPARTA_RUN=1512
    export SPARTA_STATS=30
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh
    export RANKS_PER_DOMAIN=7
    export SPARTA_RUN=1297
    export SPARTA_STATS=30
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh
    export RANKS_PER_DOMAIN=4
    export SPARTA_RUN=846
    export SPARTA_STATS=20
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh
    export RANKS_PER_DOMAIN=1
    export SPARTA_RUN=262
    export SPARTA_STATS=6
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh

    export SPARTA_PPC=100
    export RANKS_PER_DOMAIN=14
    export SPARTA_RUN=1628
    export SPARTA_STATS=30
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh
    export RANKS_PER_DOMAIN=11
    export SPARTA_RUN=1642
    export SPARTA_STATS=30
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh
    export RANKS_PER_DOMAIN=7
    export SPARTA_RUN=1416
    export SPARTA_STATS=30
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh
    export RANKS_PER_DOMAIN=4
    export SPARTA_RUN=921
    export SPARTA_STATS=20
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh
    export RANKS_PER_DOMAIN=1
    export SPARTA_RUN=287
    export SPARTA_STATS=6
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh

    export SPARTA_PPC=64
    export RANKS_PER_DOMAIN=14
    export SPARTA_RUN=2698
    export SPARTA_STATS=60
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh
    export RANKS_PER_DOMAIN=11
    export SPARTA_RUN=2719
    export SPARTA_STATS=60
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh
    export RANKS_PER_DOMAIN=7
    export SPARTA_RUN=2283
    export SPARTA_STATS=50
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh
    export RANKS_PER_DOMAIN=4
    export SPARTA_RUN=1458
    export SPARTA_STATS=30
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh
    export RANKS_PER_DOMAIN=1
    export SPARTA_RUN=440
    export SPARTA_STATS=10
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh

    export SPARTA_PPC=24
    export RANKS_PER_DOMAIN=14
    export SPARTA_RUN=6659
    export SPARTA_STATS=100
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh
    export RANKS_PER_DOMAIN=11
    export SPARTA_RUN=6169
    export SPARTA_STATS=100
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh
    export RANKS_PER_DOMAIN=7
    export SPARTA_RUN=5186
    export SPARTA_STATS=100
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh
    export RANKS_PER_DOMAIN=4
    export SPARTA_RUN=3363
    export SPARTA_STATS=80
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh
    export RANKS_PER_DOMAIN=1
    export SPARTA_RUN=1039
    export SPARTA_STATS=20
    sleep 0.2
    sbatch ${SBATCH_OPTS} run-crossroads-mpionly.sh

done

exit 0
