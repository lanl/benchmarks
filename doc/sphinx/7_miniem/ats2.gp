#!/usr/bin/gnuplot
set terminal pngcairo enhanced size 1024, 768 dashed font 'Helvetica,18'
set output "ats2.png"

set title "MiniEM Throughput Performance on ATS-2/Vortex" font "serif,24"
set xlabel "No. Cells"
set ylabel "(k-cell-steps/sec)"

# set xrange [1:64]
set yrange [0:30]
set key left top

# set logscale x 2
# set logscale y 2

set grid
show grid

set datafile separator comma
set key autotitle columnheader

set style line 1 linetype 6 dashtype 1 linecolor rgb "#FF0000" linewidth 2 pointtype 6 pointsize 3
set style line 2 linetype 1 dashtype 2 linecolor rgb "#FF0000" linewidth 2
set style line 3 linetype 6 dashtype 1 linecolor rgb "#0000FF" linewidth 2 pointtype 6 pointsize 3

plot "ats2.csv" using 1:2 with linespoints linestyle 3
