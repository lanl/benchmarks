#!/usr/bin/gnuplot
set terminal pngcairo enhanced size 1024, 768 dashed font 'Helvetica,18'

# no title needed since we will caption the figure
#set title "Strong scaling of SPP 1 on Crossroads" font "serif,22"
set xlabel "Num Cores"
set ylabel "Figure of Merit (unknowns/sec)"

set xrange [1:112]
set key left top

set logscale x 2
set logscale y 2

set grid
show grid

set datafile separator comma
set key autotitle columnheader

set style line 1 linetype 6 dashtype 1 linecolor rgb "#FF0000" linewidth 2 pointtype 6 pointsize 3
set style line 2 linetype 1 dashtype 2 linecolor rgb "#FF0000" linewidth 2

set output "spr_p1_roci.png"
plot "roci_spr_p1_plot.csv" using 1:2 with linespoints linestyle 1, "" using 1:3 with line linestyle 2

set output "spr_p2_roci.png"
plot "roci_spr_p2_plot.csv" using 1:2 with linespoints linestyle 1, "" using 1:3 with line linestyle 2