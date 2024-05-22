#!/usr/bin/gnuplot
set terminal pngcairo enhanced size 1024, 768 dashed font 'Helvetica,18'

#set title "Parthenon-VIBE weak-scaling on Crossroads" font "serif,22"
set xlabel "No. of Nodes (8 ranks x 14 threads per Node)"
set ylabel "Figure of Merit (zone-cycles/sec) per Node"
set format y "%.2e"
set xrange [2:4096]
set key left top

set logscale x 2
set logscale y 2

set grid
show grid

set datafile separator comma
set key autotitle columnheader

set style line 1 linetype 6 dashtype 1 linecolor rgb "#FF0000" linewidth 2 pointtype 6 pointsize 3
set style line 2 linetype 1 dashtype 2 linecolor rgb "#FF0000" linewidth 2


set output "weak.png"
plot "weak.csv" using 1:3 with linespoints linestyle 1, "" using 1:4 with line linestyle 2
