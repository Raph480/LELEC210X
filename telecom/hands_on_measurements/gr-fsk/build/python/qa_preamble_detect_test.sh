#!/usr/bin/sh
export VOLK_GENERIC=1
export GR_DONT_LOAD_PREFS=1
export srcdir="/mnt/c/Users/marti/OneDrive/Documents/Ecole/EPL/M1/Q1/LELEC2102-Project_Embedded_system/LELEC210X/telecom/hands_on_measurements/gr-fsk/python"
export GR_CONF_CONTROLPORT_ON=False
export PATH="/mnt/c/Users/marti/OneDrive/Documents/Ecole/EPL/M1/Q1/LELEC2102-Project_Embedded_system/LELEC210X/telecom/hands_on_measurements/gr-fsk/build/python":$PATH
export LD_LIBRARY_PATH="":$LD_LIBRARY_PATH
export PYTHONPATH=/mnt/c/Users/marti/OneDrive/Documents/Ecole/EPL/M1/Q1/LELEC2102-Project_Embedded_system/LELEC210X/telecom/hands_on_measurements/gr-fsk/build/swig:$PYTHONPATH
/usr/bin/python3 /mnt/c/Users/marti/OneDrive/Documents/Ecole/EPL/M1/Q1/LELEC2102-Project_Embedded_system/LELEC210X/telecom/hands_on_measurements/gr-fsk/python/qa_preamble_detect.py 
