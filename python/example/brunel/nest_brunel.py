import numpy as np
from parameters import *
import nest


# set up seed for random number
seed = seed
rng_seed = seed


# set up NEST kernel
nest.ResetKernel()
nest.SetKernelStatus({"resolution": dt, "print_time": False})
nest.SetDefaults(neuron_model, neuron_params)


# create generic neuron with Axon and Dendrite
nest.CopyModel(neuron_model, "excitatory")
nest.CopyModel(neuron_model, "inhibitory")


# build network
pop_exc = nest.Create("excitatory", NE)
pop_inh = nest.Create("inhibitory", NI)

nest.CopyModel("static_synapse", "device", {"weight": weight, "delay": delay})

poisson_generator_inh = nest.Create("poisson_generator")
nest.SetStatus(poisson_generator_inh, {"rate": rate})
nest.Connect(poisson_generator_inh, pop_inh, "all_to_all", syn_spec="device")

poisson_generator_ex = nest.Create("poisson_generator")
nest.SetStatus(poisson_generator_ex, {"rate": rate})
nest.Connect(poisson_generator_ex, pop_exc, "all_to_all", syn_spec="device")


spike_detector = nest.Create("spike_recorder")


nest.Connect(pop_exc + pop_inh, spike_detector, "all_to_all", syn_spec="device")

nest.CopyModel(
    "static_synapse", "inhibitory_synapse", {"weight": -g * weight, "delay": delay}
)
nest.Connect(
    pop_inh,
    pop_exc + pop_inh,
    conn_spec={"rule": "fixed_indegree", "indegree": CI},
    syn_spec="inhibitory_synapse",
)

nest.CopyModel("static_synapse", "EI_synapse", {"weight": weight, "delay": delay})
nest.Connect(
    pop_exc,
    pop_inh + pop_exc,
    conn_spec={"rule": "fixed_indegree", "indegree": CE},
    syn_spec="EI_synapse",
)

nest.Simulate(tfinal)


events = nest.GetStatus(spike_detector, "events")[0]
times = events["times"]
sources = events["senders"]

np.save("times.npy", times)
np.save("sources.npy", sources)
