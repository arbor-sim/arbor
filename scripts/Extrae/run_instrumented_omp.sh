#!/bin/bash

display_usage() { 
	echo -e "\nUsage:\n$0 [num_threads] \n" 
} 

# if less than one arguments supplied, display usage 
if [  $# -le 0 ] 
	then 
		display_usage
		exit 1
fi 
 
# check whether user had supplied -h or --help . If yes display usage 
if [[ ( $# == "--help") ||  $# == "-h" ]] 
	then 
		display_usage
		exit 0
fi


export OMP_NUM_THREADS=$1
export EXTRAE_CONFIG_FILE=extrae.xml
source ${EXTRAE_HOME}/etc/extrae.sh
export LD_PRELOAD=$(find $EXTRAE_HOME -name "libomptrace.so")


./../../build/miniapp/miniapp.exe -n 10

unset LD_PRELOAD