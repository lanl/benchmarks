#!/usr/bin/gnuplot
set terminal pngcairo enhanced size 1024, 768 dashed font 'Helvetica,18'
set output "ats3--scale--main.png"

set title "SPARTA Weak Scaling Throughput w/ ppc=35" font "serif,22"
set xlabel "No. Nodes"
set ylabel "Figure of Merit (Mega particle steps per sec per node)"

set xrange [1:2048]
set key left bottom

set logscale x 2
# set logscale y 2
set xtics 1,2,2048

# set format x "%.0f%%"

set grid
show grid

set datafile separator comma
set key autotitle columnheader

set style line 1 linetype 6 dashtype 1 linecolor rgb "#FF0000" linewidth 2 pointtype 6 pointsize 3
set style line 2 linetype 1 dashtype 2 linecolor rgb "#FF0000" linewidth 2

plot "ats3--scale--main.csv" using 1:2 with linespoints linestyle 1, "" using 1:3 with line linestyle 2
