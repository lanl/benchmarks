#!/usr/bin/gnuplot
set terminal pngcairo enhanced size 1024, 768 dashed font 'Helvetica,18'
set output "ats3-4116k-mem.png"

set title "MiniEM Strong Scaling Memory on ATS-3/Crossroads w/ 4116k Cells" font "serif,22"
set xlabel "Processing Elements on Each NUMA Domain"
set ylabel "Maximum Resident Set Size (GiB)"

set xrange [0.0625:1]
set key left top
set xtics ("6.25%%" 0.0625, "12.5%%" 0.125, "25%%" 0.25, "50%%" 0.5, "100%%" 1.0) 

set logscale x 2
# set logscale y 2

set grid
show grid

set datafile separator comma
set key autotitle columnheader

set style line 1 linetype 6 dashtype 1 linecolor rgb "#FF0000" linewidth 2 pointtype 6 pointsize 3
set style line 2 linetype 1 dashtype 2 linecolor rgb "#FF0000" linewidth 2

# plot "ats3.csv" every ::0::4 using 2:5 with linespoints linestyle 1
plot "ats3-4116k.csv" using 2:5 with linespoints linestyle 1
