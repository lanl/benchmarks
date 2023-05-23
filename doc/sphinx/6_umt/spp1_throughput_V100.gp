#!/usr/bin/gnuplot
set terminal pngcairo enhanced size 1024, 768 dashed font 'Helvetica,18'
set output "spp1_throughput_V100.png"

# no title needed since we will caption the figure
#set title "Power9/V100 single GPU throughput as a function of problem size" font "serif,22"
set xlabel "Num Unknowns"
set ylabel "Figure of Merit (unknowns/sec)"

# allow autoscaling
#set xrange [1:128]
set key left top

# linear axes for throughput plots
#set logscale x 2
#set logscale y 2

set grid
show grid

set datafile separator comma
set key autotitle columnheader

set style line 1 linetype 6 dashtype 1 linecolor rgb "#FF0000" linewidth 2 pointtype 6 pointsize 3
set style line 2 linetype 1 dashtype 2 linecolor rgb "#FF0000" linewidth 2

plot "umtsp2_throughput_gpu.csv" using 1:2 with linespoints linestyle 1

