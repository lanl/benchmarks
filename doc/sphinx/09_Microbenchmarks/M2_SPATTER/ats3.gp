#!/usr/bin/gnuplot
set terminal pngcairo enhanced size 1024, 768 dashed font 'Helvetica,18'
set output "ats3_weak_average_asteroid_5.png"

set xlabel "No. Processing Elements"
set ylabel "Figure of Merit (Avg. Bandwidth per rank MB/s)"

set xrange [1:128]
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


plot "ats3_weak_average_asteroid_5.csv" using 1:2 with linespoints linestyle 1

set output "ats3_weak_total_asteroid_5.png"
set ylabel "Figure of Merit (Total Bandwidth MB/s)"
plot "ats3_weak_total_asteroid_5.csv" using 1:2 with linespoints linestyle 1

set output "ats3_weak_average_asteroid_9.png"
set ylabel "Figure of Merit (Total Bandwidth MB/s)"
plot "ats3_weak_average_asteroid_5.csv" using 1:2 with linespoints linestyle 1

set output "ats3_weak_total_asteroid_9.png"
set ylabel "Figure of Merit (Total Bandwidth MB/s)"
plot "ats3_weak_total_asteroid_9.csv" using 1:2 with linespoints linestyle 1
