# WFSimDev
Development of fax

This is recoding of Fax Simulation inside pax,
for the original code see
[simulator](https://github.com/XENON1T/pax/blob/master/pax/simulation.py) and 
[WaveformSimulator](https://github.com/XENON1T/pax/blob/master/pax/plugins/io/WaveformSimulator.py).

Studies that results into simulation parameters are summariezed on 
[wiki](https://xe1t-wiki.lngs.infn.it/doku.php?id=xenon:xenon1t:analysis:sciencerun1:waveform_simulator_summary_170629).

### simulator.py
This contain all the code used for waveform simulation.

### run_simulator.py
This is an example of generating fake instructions then use classes in simulator to generate pax event datastructure, and then process them.

### itp_map.py
This is a quick fix to vectorize multi-dimentional interpolation.
