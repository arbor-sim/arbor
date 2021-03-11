## Miniapp for simulating the Brunel network of LIF neurons.

The network consists of 2 main populations: excitatory and inhibitory populations.
Each neuron from the network receives a fixed number (proportional to the size
of the population) of incoming connections from both of these groups.
In addition to the input from excitatory and inhibitory populations,
each neuron receives a fixed number of connections from the Poisson neurons
producing the Poisson-like input that is integrated into the LIF cell group so
that the communication of Poisson events is bypassed.

### Parameters

The parameters that can be passed as command-line arguments are the following:

* `-n` (`--n_excitatory`): total number of cells in the excitatory population.
* `-m` (`--n_inhibitory`): total number of cells in the inhibitory population.
* `-e` (`--n_external`): number of incoming Poisson sources each excitatory and inhibitory neuron receives.
* `-p` (`--in_degree_prop`): the proportions of connections from both populations that each neurons receives.
* `-w` (`--weight`): weight of all excitatory connections.
* `-g` (`--rel_inh_w`): relative strength of inhibitory synapses with respect to the excitatory ones.
* `-d` (`--delay`): the delay of all connections.
* `-l` (`--lambda`): rate of Poisson cells (kHz).
* `-t` (`--tfinal`): length of the simulation period (ms).
* `-s` (`--delta_t`): simulation time step (ms). (this parameter is ignored)
* `-G` (`--group-size`): number of cells per cell group
* `-S` (`--seed`): seed of the Poisson sources attached to cells.
* `-f` (`--spike-file-output`): save spikes to file (Bool).
* `-z` (`--profile-only-zero`): only output profile information for rank 0.
* `-v` (`--verbose`): present more verbose information to stdout.

For example, we could run the miniapp as follows:

```
./brunel -n 400 -m 100 -e 20 -p 0.1 -w 1.2 -d 1 -g 0.5 -l 5 -t 100 -s 1 -G 50 -S 123 -f
```
