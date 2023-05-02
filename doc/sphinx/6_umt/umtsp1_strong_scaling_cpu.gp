#!/usr/bin/gnuplot
set terminal pngcairo enhanced size 1024, 768 dashed font 'Helvetica,18'
set output "umtsp1-strong_scaling_cpu.png"

set title "UMT Strong Scaling Performance on CTS-2, SP #1" font "serif,22"
set xlabel "No. Cores"
set ylabel "Figure of Merit (unknowns/sec)"

set xrange [1:128]
set key left top

set logscale x 2
set logscale y 2

set grid
show grid

set datafile separator comma
set key autotitle columnheader

set style line 1 linetype 6 dashtype 1 linecolor rgb "#FF0000" linewidth 2 pointtype 6 pointsize 3
set style line 2 linetype 1 dashtype 2 linecolor rgb "#FF0000" linewidth 2

plot "umtsp1_strong_scaling_cpu.csv" using 1:2 with linespoints linestyle 1, "" using 1:3 with line linestyle 2

