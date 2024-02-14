#!/usr/bin/gnuplot
set terminal pngcairo enhanced size 1024, 768 dashed font 'Helvetica,18'
set output "cpu_20.png"

#set title "Parthenon-VIBE Strong Scaling Performance on CTS-1, 20% Memory" font "serif,22"
set xlabel "No. Processing Elements"
set ylabel "Figure of Merit (zone-cycles/sec)"
set format y "%.2e"
set xrange [1:64]
set key left top

set logscale x 2
set logscale y 2

set grid
show grid

set datafile separator comma
set key autotitle columnheader

set style line 1 linetype 6 dashtype 1 linecolor rgb "#FF0000" linewidth 2 pointtype 6 pointsize 3
set style line 2 linetype 1 dashtype 2 linecolor rgb "#FF0000" linewidth 2

set xrange [4:120]


set output "ats3_20.png"
plot "cpu_20.csv" using 1:2 with linespoints linestyle 1, "" using 1:3 with line linestyle 2



set output "ats3_40.png"
plot "cpu_40.csv" using 1:2 with linespoints linestyle 1, "" using 1:3 with line linestyle 2


set output "ats3_60.png"
plot "cpu_60.csv" using 1:2 with linespoints linestyle 1, "" using 1:3 with line linestyle 2

# Scaling Output
set output "parthenon_roci_scale_range.png"
set xrange [380:650]
unset logscale xy
set format y "%.1e"
set xlabel "NX (NX=nx=ny=nz)"
set key title "Nodes"
set title "Parthenon Multi Node Scaling" font "serif,22"
plot "parthenon_roci_scale_nxrange.csv" using 1:2 with linespoints linestyle 1, "" using 1:3 with line linestyle 2, "" using 1:4 with line linestyle 3

# SCALING PLOTS, Y IS FOM PER NODE

set xrange [32:96]
set yrange [7e6:1.5e7]
set xlabel "Nodes"
set ylabel "FOM/node"
unset title
unset key
# set title "Branson Multi Node Scaling" font "serif,22"
set output "parthenon_roci_scale_pernode.png"
plot "parthenon_roci_scale_pernode.csv" using 1:5 with linespoints linestyle 1
