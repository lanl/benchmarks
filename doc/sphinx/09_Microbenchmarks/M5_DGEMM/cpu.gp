#!/usr/bin/gnuplot
set terminal pngcairo enhanced size 1024, 768 dashed font 'Helvetica,18'
# set output "dgemm_cts1.png"

# set title " Single node Dgemm" font "serif,22"
set xlabel "No. Processing Elements"
set ylabel "Figure of Merit (GFlops)"

set xrange [1:64]
set key left top

set logscale x 2
set logscale y 2

set grid
show grid

set datafile separator comma
set key autotitle columnheader

set style line 1 linetype 6 dashtype 1 linecolor rgb "#FF0000" linewidth 2 pointtype 6 pointsize 3
set style line 2 linetype 1 dashtype 2 linecolor rgb "#FF0000" linewidth 2

#plot "dgemm_cts1.csv" using 1:2 with linespoints linestyle 1, "" using 1:3 with line linestyle 2

set output "dgemm_ats3.png"
set xrange [32:128]
plot "dgemm_ats3.csv" using 1:2 with linespoints linestyle 1


