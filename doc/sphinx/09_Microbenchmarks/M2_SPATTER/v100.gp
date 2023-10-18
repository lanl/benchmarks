#!/usr/bin/gnuplot
set terminal pngcairo enhanced size 1024, 768 dashed font 'Helvetica,18'
set output "v100_throughput_asteroid.png"

set xlabel "No. Gathers/Scatters"
set ylabel "Figure of Merit (Total Bandwidth MB/s)"

set xrange [512:65535]
set key outside

set logscale x 2

set grid
show grid

set datafile separator comma
set key autotitle columnheader

do for [i=1:10] {
    set style line i linewidth 3 pointsize 1.5
}

do for [i=11:15] {
    set style line i linewidth 3 dashtype 2 pointsize 1.5
}

plot "v100_throughput_asteroid.csv" using 1:2 with linespoints linestyle 1, "" using 1:3 with linespoints linestyle 2, "" using 1:4 with linespoints linestyle 3, "" using 1:5 with linespoints linestyle 4, "" using 1:6 with linespoints linestyle 5, "" using 1:7 with linespoints linestyle 11, "" using 1:8 with linespoints linestyle 12, "" using 1:9 with linespoints linestyle 13, "" using 1:10 with linespoints linestyle 14
