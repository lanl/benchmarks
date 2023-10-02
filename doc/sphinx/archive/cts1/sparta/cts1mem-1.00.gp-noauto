#!/usr/bin/gnuplot
set terminal pngcairo enhanced size 1024, 768 dashed font 'Helvetica,18'
set output "cts1mem-1.00.png"

set title "MiniEM Strong Scaling High-water Memory on CTS-1/Manzano (1.00 GiB/PE)" font "serif,22"
set xlabel "No. Processing Elements"
set ylabel "Maximum Resident Set Size (GiB)"

set xrange [1:64]
set key left top

set logscale x 2
# set logscale y 2

set grid
show grid

set datafile separator comma
set key autotitle columnheader

set style line 1 linetype 6 dashtype 1 linecolor rgb "#FF0000" linewidth 2 pointtype 6 pointsize 3
set style line 2 linetype 1 dashtype 2 linecolor rgb "#FF0000" linewidth 2

plot "cts1-1.00.csv" using 1:4 with linespoints linestyle 1
