#!/bin/bash


export EXTRAE_CONFIG_FILE=extrae.xml
source ${EXTRAE_HOME}/etc/extrae.sh
export LD_PRELOAD=$(find $EXTRAE_HOME -name "libseqtrace.so")

./../../build/miniapp/miniapp.exe 

unset LD_PRELOAD