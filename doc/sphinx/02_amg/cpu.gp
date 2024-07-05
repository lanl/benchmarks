#!/usr/bin/gnuplot
set terminal pngcairo enhanced size 1024, 768 dashed font 'Helvetica,18'
set output "roci_1_120.png"

set title "AMG2023 Strong Scaling for Problem 1, 120 x 120 x 120" font "serif,22"
set xlabel "n"
set ylabel "FOM"

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
set style line 3 linetype 6 dashtype 1 linecolor rgb "#0000FF" linewidth 2 pointtype 6 pointsize 3

plot "roci_1_120.csv" using 1:2 with linespoints linestyle 1, "" using 1:3 with line linestyle 2

set output "roci_1_160.png"
set title "AMG2023 Strong Scaling for Problem 1, 160 x 160 x 160" font "serif,22"
plot "roci_1_160.csv" using 1:2 with linespoints linestyle 1, "" using 1:3 with line linestyle 2

set output "roci_1_200.png"
set title "AMG2023 Strong Scaling for Problem 1, 200 x 200 x 200" font "serif,22"
plot "roci_1_200.csv" using 1:2 with linespoints linestyle 1, "" using 1:3 with line linestyle 2

set output "roci_2_200.png"
set title "AMG2023 Strong Scaling for Problem 2, 200 x 200 x 200" font "serif,22"
plot "roci_2_200.csv" using 1:2 with linespoints linestyle 1, "" using 1:3 with line linestyle 2

set output "roci_2_256.png"
set title "AMG2023 Strong Scaling for Problem 2, 256 x 256 x 256" font "serif,22"
plot "roci_2_256.csv" using 1:2 with linespoints linestyle 1, "" using 1:3 with line linestyle 2

set output "roci_2_320.png"
set title "AMG2023 Strong Scaling for Problem 2, 320 x 320 x 320" font "serif,22"
plot "roci_2_320.csv" using 1:2 with linespoints linestyle 1, "" using 1:3 with line linestyle 2

