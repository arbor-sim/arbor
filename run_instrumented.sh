#!/bin/bash

export OMP_NUM_THREADS=4
export EXTRAE_CONFIG_FILE=extrae.xml
export EXTRAE_HOME=/apps/CEPBATOOLS/extrae/3.3.0/openmpi+libgomp4.2/64
source ${EXTRAE_HOME}/etc/extrae.sh

./miniapp.exe
