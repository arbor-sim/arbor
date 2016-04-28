for mech in pas hh
do
    modcc -t cpu -o ../include/mechanisms/$mech.hpp ./mod/$mech.mod
done
