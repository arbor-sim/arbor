# Gap Junctions Example

A miniapp that demonstrates how to describe how to build a network with gap junctions.

##Structure:
Cells are structured into groups that are inter-connected by gap junctions; Groups are
connected by synapses. The first cell of the first group (top left in diagram) has a 
current stimulus.


```
c --gj-- c --gj-- c --gj-- c --gj-- c
                                    |
                                    syn
                                    |
c --gj-- c --gj-- c --gj-- c --gj-- c
|
syn
|
c --gj-- c --gj-- c --gj-- c --gj-- c
```
     

## Tunable parameters
* _n_cables_: number of groups of cells connected by gap junctions.
* _n_cells_per_cable_: number of cells in a group.
* _stim_duration_: duration that the stimulus on the first cell is on. 
* _event_min_delay_: minimum delay of the network.
* _event_weight_: weight of an event.
* _sim_duration_: duration of the simulation. 
* _print_all_: print the voltages of all cells in nerwork.

An example parameter file is:
```
{
    "name": "small test",
    "n-cables": 3,
    "n-cells-per-cable": 5,
    "stim-duration": 30,
    "event-min-delay": 10,
    "event-weight": 0.05,
    "sim-duration": 100, 
    "print-all": false
}
```