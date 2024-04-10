#!/bin/bash

# Activate the virtual environment
source /home/rdpuser/scripts/scriptsv2-main/venv/bin/activate

# Change directory to the script location
cd /home/rdpuser/scripts/scriptsv2-main/

# Run the first Python script and then run the second Python script if the first one succeeds,
# then run the third Python script if the second one succeeds
python /home/rdpuser/scripts/scriptsv2-main/new_fetchv2.py && \
python /home/rdpuser/scripts/scriptsv2-main/new_calculationsv3.py && \
python /home/rdpuser/scripts/scriptsv2-main/app.py
