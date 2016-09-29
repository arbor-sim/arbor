#!/usr/bin/env bash


for mech in pas hh expsyn exp2syn
do
    echo ../external/modparser/bin/modcc ${flags} -o ../include/mechanisms/$mech.hpp ./mod/$mech.mod
    modcc -t cpu -o $mech.hpp ./mod/$mech.mod
    modcc -t gpu -o gpu/$mech.hpp ./mod/$mech.mod
done
