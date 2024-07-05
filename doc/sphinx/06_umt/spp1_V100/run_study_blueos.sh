srun -n1 ./install/bin/test_driver -b 1 -g -B global -d 1,1,1 |& tee run.blueos.1.log
srun -n1 ./install/bin/test_driver -b 1 -g -B global -d 2,2,2 |& tee run.blueos.2.log
srun -n1 ./install/bin/test_driver -b 1 -g -B global -d 3,3,3 |& tee run.blueos.3.log
srun -n1 ./install/bin/test_driver -b 1 -g -B global -d 4,4,4 |& tee run.blueos.4.log
srun -n1 ./install/bin/test_driver -b 1 -g -B global -d 5,5,5 |& tee run.blueos.5.log
srun -n1 ./install/bin/test_driver -b 1 -g -B global -d 6,6,6 |& tee run.blueos.6.log
