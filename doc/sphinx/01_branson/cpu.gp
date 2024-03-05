#!/usr/bin/gnuplot
set terminal pngcairo enhanced size 1024, 768 dashed font 'Helvetica,18'
set output "cpu_10M.png"

#set title "Branson Strong Scaling Performance on Crossroads, 10M particles" font "serif,22"
set xlabel "No. Processing Elements"
set ylabel "Figure of Merit (particles/sec)"

set xrange [8:112]
set key left top

set logscale x 2
set logscale y 2

set grid
show grid

set datafile separator comma
set key autotitle columnheader

set style line 1 linetype 6 dashtype 1 linecolor rgb "#FF0000" linewidth 2 pointtype 6 pointsize 3
set style line 2 linetype 1 dashtype 2 linecolor rgb "#FF0000" linewidth 2
set style line 3 linetype 6 dashtype 1 linecolor rgb "#0000FF" linewidth 2 pointtype 6 pointsize 3

plot "cpu_10M.csv" using 1:2 with linespoints linestyle 1, "" using 1:3 with line linestyle 2

set output "cpu_66M.png"
#set title "Branson Strong Scaling Performance on Crossroads, 66M particles" font "serif,22"
plot "cpu_66M.csv" using 1:2 with linespoints linestyle 1, "" using 1:3 with line linestyle 2

set output "cpu_200M.png"
#set title "Branson Strong Scaling Performance on Crossroads, 200M particles" font "serif,22"
plot "cpu_200M.csv" using 1:2 with linespoints linestyle 1, "" using 1:3 with line linestyle 2

set output "cpu_10M_new.png"
plot "cpu_10M_new.csv" using 1:2 with linespoints linestyle 1, "" using 1:3 with line linestyle 2

set output "cpu_66M_new.png"
plot "cpu_66M_new.csv" using 1:2 with linespoints linestyle 1, "" using 1:3 with line linestyle 2

set output "cpu_200M_new.png"
plot "cpu_200M_new.csv" using 1:2 with linespoints linestyle 1, "" using 1:3 with line linestyle 2

# Scaling Output
set output "branson_roci_scale_range.png"
set xrange [200:8000]
set format y "%.1e"
unset logscale xy
set key title "Nodes"
set title "Branson Multi Node Scaling" font "serif,22"
plot "branson_roci_scale_photonrange.csv" using 1:2 with linespoints linestyle 1, "" using 1:3 with line linestyle 2, "" using 1:4 with line linestyle 3

# SCALING PLOTS, Y IS FOM PER NODE
set xrange [32:96]
set yrange [2.5e6:3.5e6]
set xlabel "Nodes"
set ylabel "FOM/node"
# set title "Branson Multi Node Scaling" font "serif,22"
set output "branson_roci_scale.png"
plot "branson_roci_scale.csv" using 3:5 with linespoints linestyle 1

set yrange [2e5:3e6]
set output "branson_roci_scale_badnodes.png"
set title "Branson Multi Node Scaling" font "serif,22"
plot "branson_roci_badnodes_scale.csv" using 3:5 with linespoints linestyle 1
