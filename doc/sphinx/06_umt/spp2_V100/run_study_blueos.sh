srun -n1 ./install/bin/test_driver -b 2 -g -B global -d 3,3,3 |& tee run.blueos.3.log
srun -n1 ./install/bin/test_driver -b 2 -g -B global -d 4,4,4 |& tee run.blueos.4.log
srun -n1 ./install/bin/test_driver -b 2 -g -B global -d 5,5,5 |& tee run.blueos.5.log
srun -n1 ./install/bin/test_driver -b 2 -g -B global -d 6,6,6 |& tee run.blueos.6.log
srun -n1 ./install/bin/test_driver -b 2 -g -B global -d 8,8,8 |& tee run.blueos.8.log
srun -n1 ./install/bin/test_driver -b 2 -g -B global -d 11,11,11 |& tee run.blueos.11.log
srun -n1 ./install/bin/test_driver -b 2 -g -B global -d 13,13,13 |& tee run.blueos.13.log
srun -n1 ./install/bin/test_driver -b 2 -g -B global -d 15,15,15 |& tee run.blueos.15.log
