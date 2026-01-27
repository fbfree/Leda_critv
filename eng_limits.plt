set xrange [0:100]
set yrange [0:10]
set ylabel 'Peak surface acceleration (m/s²)'
set xlabel 'Train speed (m/s)'
set key top left
set x2range [0:360]
set x2label 'Train speed (km/h)'
set x2tics
set xtics nomirror
# plot 'beauharnois2_limits.dat' u 1:3 w l ti 'Thick Beauharnois clay', 'beauharnois2a_limits.dat' u 1:3 w l ti 'modified Beauharnois clay'

# 'gloucester_limits.dat' u 1:3 w l ti 'South Gloucester clay', 'limoges4_limits.dat' u 1:3 w l ti '4m sand over clay', 3.5 ti 'Engineering limit'

set yrange [0:0.001]
set ylabel 'Peak shear strain in clay'
plot 'beauharnois_UIC60_limits.dat' u 1:4 w l ti 'UIC60 rail on clay', 'beauharnois_136lb_limits.dat' u 1:4 w l ti '136lb/ft rail on clay','beauharnois_UIC60_slab_limits.dat' u 1:4 w l ti 'Slab track on clay', 1.5e-4 ti 'Limit of linearity'
 #, '5m_limits.dat' u 1:4 w l ti '5 meters', '10m_limits.dat' u 1:4 w l ti '10 meters',
replot 'beauharnois_UIC60_degraded_limits.dat' u 1:4 w l ti 'UIC60 rail on degraded clay'
# replot 'beauharnois_limits.dat' u 1:($4*5) w l ti 'Beauharnois clay'
replot 'limoges4_UIC60_limits.dat' u 1:4 w l ti 'UIC60 rail on 4m sand over clay'
replot 'limoges3_UIC60_limits.dat' u 1:4 w l ti 'UIC60 rail on 3m sand over clay'

# replot 'beauharnois_UIC60_degraded50_limits.dat' u 1:4 w l ti 'UIC60 rail on degraded 50'
# replot 'beauharnois_UIC60_degraded40_limits.dat' u 1:4 w l ti 'UIC60 rail on degraded 40'
# replot 'beauharnois_UIC60_degraded30_limits.dat' u 1:4 w l ti 'UIC60 rail on degraded 30'
# replot 'beauharnois_UIC60_degraded20_limits.dat' u 1:4 w l ti 'UIC60 rail on degraded 20'

# , 'gloucester_limits.dat' u 1:4 w l ti 'South Gloucester clay', 'limoges4_limits.dat' u 1:4 w l ti '4m sand over clay',
