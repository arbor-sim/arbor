#!/bin/bash

if [ -z ${EXTRAE_HOME+x} ] 
	then 
		echo -e "\nSpecify EXTRAE_HOME variable before executing this script\n"
		echo "	export EXTRAE_HOME=path/to/directory"
		exit 1
	else 
		echo -e "\nEXTRAE_HOME is set to '$EXTRAE_HOME'"
fi


export EXTRAE_CONFIG_FILE=extrae.xml
source ${EXTRAE_HOME}/etc/extrae.sh
export LD_PRELOAD=$(find $EXTRAE_HOME -name "libseqtrace.so")

./../../build/miniapp/miniapp.exe 

unset LD_PRELOAD