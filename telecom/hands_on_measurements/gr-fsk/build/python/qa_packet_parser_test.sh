#!/usr/bin/sh
export VOLK_GENERIC=1
export GR_DONT_LOAD_PREFS=1
export srcdir=/home/martin/Documents/EPL/M1/Project-Embedded/LELEC210X/telecom/hands_on_measurements/gr-fsk/python
export GR_CONF_CONTROLPORT_ON=False
export PATH="/home/martin/Documents/EPL/M1/Project-Embedded/LELEC210X/telecom/hands_on_measurements/gr-fsk/build/python":"$PATH"
export LD_LIBRARY_PATH="":$LD_LIBRARY_PATH
export PYTHONPATH=/home/martin/Documents/EPL/M1/Project-Embedded/LELEC210X/telecom/hands_on_measurements/gr-fsk/build/test_modules:/home/martin/Documents/EPL/M1/Project-Embedded/LELEC210X/telecom/hands_on_measurements/gr-fsk/build/swig:$PYTHONPATH
/usr/bin/python3 /home/martin/Documents/EPL/M1/Project-Embedded/LELEC210X/telecom/hands_on_measurements/gr-fsk/python/qa_packet_parser.py 
