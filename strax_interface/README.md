# Fax in strax

Here are the files needed to run the waveform simulator fax and give output in the same formats as used by strax.

## Installation

Copy the fax.py and fax_interface to your straxen/plugins folder. Then add "import fax_interface, fax" to the __init__.py file
Since github has a limit on the maximum allowed file size not all configuration files can be hosted here. So you still need to get your hands on the files needed for pmt_after_pulse and noise. They are somewhere on dali (please look up).
Alternatively you can just disable both, afterpulses are currently not working and commenting line 152 in fax_interface.py will disable noise data.

##Usage
You can choose the simulate too two different data types. If you want a quick and dirty simulation and are not really interested in low level stuff you can simulate directly to Peaks. This is done by PeaksFromFax.
If you want to go deeper and get raw_records, normally created by the DAQ reader, you can use RawRecordsFromFax. Depending on what you want your strax contex will look a bit different.

If you want to look at peaks starting at raw_records, launch a jupyter notebook and do the following:
```python
import strax
import straxen
st = strax.Context(
        storage=[
            strax.DataDirectory('./',)],
    
        register=[straxen.plugins.fax_interface.RawRecordsFromFax,
                  straxen.plugins.plugins.Records,
                  straxen.plugins.plugins.Peaks,
                 ],
        config=dict(nevents=2,  
                   )
```
Nevents is an option telling fax how many events to simulate. The default value is 10, but maybe you want a different amount

To get some data do:
```python
rr = st.get_array('1','raw_records')
p = st.get_array('1','peaks')
```

And then enjoy your newly made data :)

If you want to get the data straight to peaks change the register value in the Context declaration to:
```python
register = [straxen.plugins.fax_interface.PeaksFromFax]
```


