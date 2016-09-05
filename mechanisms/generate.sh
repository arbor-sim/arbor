#!/usr/bin/env bash

#flags="-t cpu -O"
flags="-t cpu"

for mech in pas hh expsyn
do
    echo ../external/modparser/bin/modcc ${flags} -o ../include/mechanisms/$mech.hpp ./mod/$mech.mod
    ../external/modparser/bin/modcc ${flags} -o ../include/mechanisms/$mech.hpp ./mod/$mech.mod
done
