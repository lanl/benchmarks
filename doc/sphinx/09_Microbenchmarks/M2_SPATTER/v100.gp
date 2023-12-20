#!/usr/bin/gnuplot
set terminal pngcairo enhanced size 1024, 768 dashed font 'Helvetica,18'
set output "v100_throughput_asteroid_5.png"

set xlabel "Data Transferred (MB)"
set ylabel "Figure of Merit (Total Bandwidth MB/s)"

set xrange [4:10000]
set nokey

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


plot "v100_throughput_asteroid_5.csv" using 1:2 with linespoints linestyle 1

set output "v100_throughput_asteroid_5.png"
set ylabel "Figure of Merit (Total Bandwidth MB/s)"
plot "v100_throughput_asteroid_5.csv" using 1:2 with linespoints linestyle 1

set output "v100_throughput_asteroid_9.png"
set ylabel "Figure of Merit (Total Bandwidth MB/s)"
plot "v100_throughput_asteroid_5.csv" using 1:2 with linespoints linestyle 1

set output "v100_throughput_asteroid_9.png"
set ylabel "Figure of Merit (Total Bandwidth MB/s)"
plot "v100_throughput_asteroid_9.csv" using 1:2 with linespoints linestyle 1
