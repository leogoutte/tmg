set terminal pdf enhanced color font ",30" size 4, 5
set palette defined ( 0  "green", 5 "yellow", 10 "red" )
set output 'bulkek.pdf'
set style data linespoints
unset ztics
unset key
set pointsize 0.8
set view 0,0
set xtics font ",24"
set ytics font ",24"
set ylabel font ",24"
set ylabel offset 1.5,0
set xrange [0:    0.58002]
emin=   -1.730765
emax=    1.717325
set ylabel "Energy (eV)"
set yrange [ emin : emax ]
set xtics ("M  "    0.00000,"K- "    0.07502,"G  "    0.22505,"M  "    0.35498,"K+ "    0.42999,"G  "    0.58002)
set arrow from    0.07502, emin to    0.07502, emax nohead
set arrow from    0.22505, emin to    0.22505, emax nohead
set arrow from    0.35498, emin to    0.35498, emax nohead
set arrow from    0.42999, emin to    0.42999, emax nohead
# please comment the following lines to plot the fatband 
plot 'bulkek.dat-valley-K' u 1:2  w lp lw 2 pt 7  ps 0.1 lc rgb 'blue', \
     'bulkek.dat-valley-Kprime' u 1:2  w lp lw 2 pt 7  ps 0.1 lc rgb 'red'
 
# uncomment the following lines to plot the fatband 
#set cbrange [0:1]
#set cbtics 1
#plot 'bulkek.dat-valley-K' u 1:2:3  w lp lw 2 pt 7  ps 0.1 lc palette
