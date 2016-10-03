#!/bin/bash

# simple script for generating source code for mechanisms
# required to build the runtime tests

args=""
#args+=" --verbose"
mechanisms="Ca KdShu2007 Ih Im NaTs2_t expsyn Ca_HVA ProbAMPANMDA_EMS ProbGABAAB_EMS"
for mech in $mechanisms
do
    ../bin/modcc ./modfiles/$mech.mod -o ./runtime/mechanisms/$mech.h $args
done

