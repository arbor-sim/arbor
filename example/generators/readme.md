# Event Generator Example

A miniapp that demonstrates how to describe event generators in a `arb::recipe`.

This miniapp uses a simple model of a single cell, with one compartment corresponding to the soma.
The soma has one synapse, to which two event generators, one inhibitory and one excitatory, are attached.

The following topics are covered:
* How to describe a simple single cell model with a `recipe`.
* How to connect Poisson event generators to a synapse in a `recipe`.
* How to sample the voltage on the soma of a cell.

## The Recipe

We need to create a recipe that describes the model.
All models derive from the `arb::recipe` base class.

We call the recipe for this example `generator_recipe`, and declare it as follows:

```C++
class generator_recipe: public arb::recipe {
public:
    // There is just the one cell in the model
    cell_size_type num_cells() const override {
        return 1;
    }

    // ...
};
```

### Describing The Network

Every user-defined recipe must provide implementations for the following three methods:
* `recipe::num_cells()`: return the total number of cells in the model.
* `recipe::get_cell_description(gid)`: return a description of the cell with `gid`.
* `recipe::get_cell_kind(gid)`:  return an `arb::cell_kind` enum value for the cell type.

This single cell model has no connections for spike communication, so the
default `recipe::connections_on(gid)` method can use the default implementation,
which returns an empty list of connections for a cell.

Above you can see the definition of `recipe::num_cells()`, which is trivial for this model, which has only once cell.
The `recipe::get_cell_description(gid)` and `recipe::get_cell_kind(gid)` methods are defined to return a single
compartment cell with one synapse, and a `arb::cell_kind::cable` respectively:

```C++
    // Return an arb::cell that describes a single compartment cell with
    // passive dynamics, and a single expsyn synapse.
    arb::util::unique_any get_cell_description(cell_gid_type gid) const override {
        arb::segment_tree tree;
        double r = 18.8/2.0; // convert 18.8 μm diameter to radius
        tree.append(arb::mnpos, {0,0,-r,r}, {0,0,r,r}, 1);

        arb::label_dict labels;
        labels.set("soma", arb::reg::tagged(1));

        arb::decor decor;
        decor.paint("\"soma\"", "pas");

        // Add one synapse labeled "syn" at the soma.
        // This synapse will be the target for all events, from both event_generators.
        decor.place(arb::mlocation{0, 0.5}, "expsyn", "syn");

        return arb::cable_cell(tree, labels, decor);
    }

    cell_kind get_cell_kind(cell_gid_type gid) const override {
        return cell_kind::cable;
    }
```

### Event Generators

For our demo, we want to attach two Poisson event generators to the synapse on our cell.

1. An excitatory synapse with spiking frequency λ\_e and weight w\_e.
2. An inhibitory synapse with spiking frequency λ\_i and weight w\_i.

To add the event generators to the synapse, we implement the `recipe::event_generators(gid)` method.

The implementation of this with hard-coded frequencies and weights is:

```C++
    std::vector<arb::event_generator> event_generators(cell_gid_type gid) const override {
        // The type of random number generator to use.
        using RNG = std::mt19937_64;

        auto hz_to_freq = [](double hz) { return hz*1e-3; };
        time_type t0 = 0;

        // Define frequencies and weights for the excitatory and inhibitory generators.
        double lambda_e =  hz_to_freq(500);
        double lambda_i =  hz_to_freq(20);
        double w_e =  0.001;
        double w_i = -0.005;

        // Make two event generators.
        std::vector<arb::event_generator> gens;

        // Add excitatory generator
         gens.push_back(
            arb::poisson_generator(
                {"syn"},               // Target synapse index on cell `gid`
                w_e,                   // Weight of events to deliver
                t0,                    // Events start being delivered from this time
                lambda_e,              // Expected frequency (kHz)
                RNG(29562872)));       // Random number generator to use

                // Add inhibitory generator
        gens.emplace_back(arb::poisson_generator({"syn"}, w_i, t0, lambda_i,  RNG(86543891)));
        return gens;
    }
```

The `recipe::event_generators(gid)` method returns a vector of `event_generator`s that are attached to the cell with `gid`.

In the implementation, an empty vector is created, and the generators are created and `push_back`ed into the vector one after the other.

Of the arguments used to construct the Poisson event generator `pgen`, the random number generator state require further explanation.
Each Poisson generator has its own private random number generator state.
The initial random number state is provided on construction.
For a real world model, the state should have a seed that is a hash of `gid` and the generator id, to ensure reproducable random streams.
For this simple example, we use hard-coded seeds to initialize the random number state.

### Sampling Voltages

To visualise the result of our experiment, we want to sample the voltage in our cell at regular time points and save the resulting sequence to a JSON file.

There are three parts to this process.

1. Define the a `probe` in the recipe, which describes the location and variable to be sampled, i.e. the soma and voltage respectively for this example.
2. Attach a sampler to this probe once the simulation has been created.
3. Write the sampled values to a JSON file once the simulation has been run.

#### 1. Define probe in recipe

The `recipe::num_probes(gid)` and `recipe::get_probe(id)` have to be defined for sampling.
In our case, the cell has one probe, which refers to the voltage at the soma.

```C++
    // There is one probe (for measuring voltage at the soma) on the cell
    cell_size_type num_probes(cell_gid_type gid)  const override {
        return 1;
    }

    arb::probe_info get_probe(cell_member_type id) const override {
        // The location at which we measure: position 0 on branch 0.
        // The cell has only one branch, branch 0, which is the soma.
        arb::mlocation loc{0, 0.0};

        // The thing we are measuring is the membrane potential.
        arb::cable_probe_membrane_voltage address{loc};

        // Put this together into a `probe_info`; the tag value 0 is not used.
        return arb::probe_info{id, 0, address};
    }
```

#### 2. Attach sampler to simulation

Once the simulation has been created, the simulation has internal state that tracks the value of the voltage at the probe location described by the recipe.
We must attach a sampler to this probe to get sample values.

The sampling interface is rich, and can be extended in many ways.
For our simple use case there are three bits of information that need to be provided when creating a sampler

1. The `probeset_id` that identifies the probe (generated in the recipe).
2. The schedule to use for sampling (in our case 10 samples every ms).
3. The location where we want to save the samples for outputing later.

```C++
    // The id of the only probe (cell 0, probe 0)
    auto probeset_id = cell_member_type{0, 0};
    // The schedule for sampling is 10 samples every 1 ms.
    auto sched = arb::regular_schedule(0.1);
    // Where the voltage samples will be stored as (time, value) pairs
    arb::trace_data<double> voltage;
    // Now attach the sampler:
    sim.add_sampler(arb::one_probe(probeset_id), sched, arb::make_simple_sampler(voltage));
```

When the simulation is run, the `simple_sampler` that we attached to the probe will store the sample values as (time, voltage) pairs in the `voltage` vector.

#### 3. Output to JSON

The function `write_trace_json()` uses the [JSON library](https://github.com/nlohmann/json) that is included in the Arbor library to write the output.
We don't document its implementation here, instead we note that the format of the output was chosen to be compatible with the `tsplot` script provided by Arbor for plotting traces.

## Visualising the Results.

The voltage samples are saved to `voltages.json`, and can be visualised using the `tsplot` Python script distributed with Arbor.
Here is an example set of steps used to build, run and plot the voltage trace:

```bash
# assuming that build is a subdirectory of the main arbor project path
cd build

# build the event generator demo
make -j event_gen.exe

# run the event generator demo
./example/event_gen.exe

# draw a plot of the results.
# uses the matplotlib in Python.
../scripts/tsplot voltages.json
```

