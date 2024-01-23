#!/usr/bin/gnuplot
set terminal pngcairo enhanced size 1024, 768 dashed font 'Helvetica,18'
set output "roci_mem.png"

set title "AMG2023 FOM at varying memory usage, Problem 1 and 2" font "serif,22"
set xlabel "GB"
set ylabel "FOM"

set xrange [10:40]
set key left top

set yrange [1.0e+6: 2.0e+7]
set grid
show grid

set datafile separator comma
set key autotitle columnheader

set style line 1 linetype 6 dashtype 1 linecolor rgb "#FF0000" linewidth 2 pointtype 6 pointsize 3
set style line 2 linetype 6 dashtype 1 linecolor rgb "#0000FF" linewidth 2 pointtype 6 pointsize 3

plot "roci_mem.csv" using 1:2 with linespoints linestyle 1, "roci_mem.csv" using 1:3 with linespoints linestyle 2
