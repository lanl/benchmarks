#!/usr/bin/gnuplot
set terminal pngcairo enhanced size 1024, 768 dashed font 'Helvetica,18'
set output "cpu_weak_average_001.png"

set title "Spatter Weak Scaling Performance on CTS-1, Flag Static 2D 001 Patterns" font "serif,22"
set xlabel "No. Processing Elements"
set ylabel "Figure of Merit (Avg. Bandwidth per rank MB/s)"

set xrange [1:64]
set key outside

set logscale x 2

set grid
show grid

set datafile separator comma
set key autotitle columnheader

do for [i=1:10] {
    set style line i linewidth 3 pointsize 1.5
}

plot "cpu_weak_average_001.csv" using 1:2 with linespoints linestyle 1, "" using 1:3 with linespoints linestyle 2, "" using 1:4 with linespoints linestyle 3, "" using 1:5 with linespoints linestyle 4, "" using 1:6 with linespoints linestyle 5, "" using 1:7 with linespoints linestyle 6, "" using 1:8 with linespoints linestyle 7, "" using 1:9 with linespoints linestyle 8


set output "cpu_weak_total_001.png"
set title "Spatter Weak Scaling Performance on CTS-1, Flag Static 2D 001 Patterns" font "serif,22"
set ylabel "Figure of Merit (Total Bandwidth MB/s)"
plot "cpu_weak_total_001.csv" using 1:2 with linespoints linestyle 1, "" using 1:3 with linespoints linestyle 2, "" using 1:4 with linespoints linestyle 3, "" using 1:5 with linespoints linestyle 4, "" using 1:6 with linespoints linestyle 5, "" using 1:7 with linespoints linestyle 6, "" using 1:8 with linespoints linestyle 7, "" using 1:9 with linespoints linestyle 8


set output "cpu_weak_average_asteroid.png"
set title "Spatter Weak Scaling Performance on CTS-1, xRAGE Asteroid Patterns" font "serif,22"
set ylabel "Figure of Merit (Avg. Bandwidth per rank MB/s)"
plot "cpu_weak_average_asteroid.csv" using 1:2 with linespoints linestyle 1, "" using 1:3 with linespoints linestyle 2, "" using 1:4 with linespoints linestyle 3, "" using 1:5 with linespoints linestyle 4, "" using 1:6 with linespoints linestyle 5, "" using 1:7 with linespoints linestyle 6, "" using 1:8 with linespoints linestyle 7, "" using 1:9 with linespoints linestyle 8, "" using 1:10 with linespoints linestyle 9

set output "cpu_weak_total_asteroid.png"
set title "Spatter Weak Scaling Performance on CTS-1, xRAGE Asteroid Patterns" font "serif,22"
set ylabel "Figure of Merit (Total Bandwidth MB/s)"
plot "cpu_weak_total_asteroid.csv" using 1:2 with linespoints linestyle 1, "" using 1:3 with linespoints linestyle 2, "" using 1:4 with linespoints linestyle 3, "" using 1:5 with linespoints linestyle 4, "" using 1:6 with linespoints linestyle 5, "" using 1:7 with linespoints linestyle 6, "" using 1:8 with linespoints linestyle 7, "" using 1:9 with linespoints linestyle 8, "" using 1:10 with linespoints linestyle 9

