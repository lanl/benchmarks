#!/usr/bin/gnuplot
#STREAM
set terminal pngcairo enhanced size 1024, 768 dashed font 'Helvetica,18'
set output "cpu_cts1.png"

set title "STREAM Single node bandwidth" font "serif,22"
set ylabel "Per core triad (MB/s)"
set y2label "FOM: Total Triad (MB/s)"

set xrange [1:40]
set yrange [3000:15000]

# set logscale x 2
# set logscale y 2

set grid
show grid

set datafile separator comma
set key autotitle columnheader

set style line 1 linetype 6 dashtype 1 linecolor rgb "#FF0000" linewidth 2 pointtype 6 pointsize 3
set style line 2 linetype 1 dashtype 2 linecolor rgb "#FF0000" linewidth 2

plot "stream_cts1.csv" using 1:2 with linespoints linestyle 1 axis x1y1, "" using 1:3 with line linestyle 2 axis x1y2



