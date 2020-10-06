Python API reference
====================

.. py:module:: arbor

arbor: multicompartment neural network models.


.. py:function:: allen_catalogue() -> arbor.mechanism_catalogue
   :module: arbor


.. py:function:: attach_sampler(*args, **kwargs)
   :module: arbor

   Overloaded function.
   
   1. attach_sampler(sim: arb::simulation, dt: float) -> arbor.sampler
   
   Attach a sample recorder to an arbor simulation.
   The recorder will record all samples from a regular sampling interval [ms] matching all probe ids.
   
   2. attach_sampler(sim: arb::simulation, dt: float, probe_id: arbor.cell_member) -> arbor.sampler
   
   Attach a sample recorder to an arbor simulation.
   The recorder will record all samples from a regular sampling interval [ms] matching one probe id.
   

.. py:function:: attach_spike_recorder(sim: arbor.simulation) -> arbor.spike_recorder
   :module: arbor

   Attach a spike recorder to an arbor simulation.
   The recorder that is returned will record all spikes generated after it has been
   attached (spikes generated before attaching are not recorded).
   

.. py:class:: backend
   :module: arbor

   Enumeration used to indicate which hardware backend to execute a cell group on.
   
   Members:
   
     gpu : Use GPU backend.
   
     multicore : Use multicore backend.
   
   
   .. py:method:: backend.__eq__
      :module: arbor
   
      (self: object, arg0: object) -> bool
      
   
   .. py:method:: backend.__getstate__
      :module: arbor
   
      (self: object) -> int_
      
   
   .. py:method:: backend.__hash__
      :module: arbor
   
      (self: object) -> int_
      
   
   .. py:method:: backend.__init__(self: arbor.backend, arg0: int) -> None
      :module: arbor
   
   
   .. py:method:: backend.__int__(self: arbor.backend) -> int
      :module: arbor
   
   
   .. py:attribute:: backend.__members__
      :module: arbor
      :value: {'gpu': backend.gpu, 'multicore': backend.multicore}
   
   
   .. py:attribute:: backend.__module__
      :module: arbor
      :value: 'arbor'
   
   
   .. py:method:: backend.__ne__
      :module: arbor
   
      (self: object, arg0: object) -> bool
      
   
   .. py:method:: backend.__new__(**kwargs)
      :module: arbor
   
      Create and return a new object.  See help(type) for accurate signature.
      
   
   .. py:method:: backend.__repr__
      :module: arbor
   
      (self: handle) -> str
      
   
   .. py:method:: backend.__setstate__
      :module: arbor
   
      (self: arbor.backend, arg0: int) -> None
      
   
   .. py:attribute:: backend.gpu
      :module: arbor
      :value: backend.gpu
   
   
   .. py:attribute:: backend.multicore
      :module: arbor
      :value: backend.multicore
   
   
   .. py:method:: backend.name
      :module: arbor
      :property:
   
      (self: handle) -> str
      

.. py:class:: benchmark_cell
   :module: arbor

   A benchmarking cell, used by Arbor developers to test communication performance.
   A benchmark cell generates spikes at a user-defined sequence of time points, and
   the time taken to integrate a cell can be tuned by setting the realtime_ratio,
   for example if realtime_ratio=2, a cell will take 2 seconds of CPU time to
   simulate 1 second.
   
   
   .. py:method:: benchmark_cell.__init__(*args, **kwargs)
      :module: arbor
   
      Overloaded function.
      
      1. __init__(self: arbor.benchmark_cell, schedule: pyarb::regular_schedule_shim, realtime_ratio: float = 1.0) -> None
      
      Construct a benchmark cell that generates spikes at regular intervals.
      
      2. __init__(self: arbor.benchmark_cell, schedule: pyarb::explicit_schedule_shim, realtime_ratio: float = 1.0) -> None
      
      Construct a benchmark cell that generates spikes at a sequence of user-defined times.
      
      3. __init__(self: arbor.benchmark_cell, schedule: pyarb::poisson_schedule_shim, realtime_ratio: float = 1.0) -> None
      
      Construct a benchmark cell that generates spikes at times defined by a Poisson sequence.
      
   
   .. py:attribute:: benchmark_cell.__module__
      :module: arbor
      :value: 'arbor'
   
   
   .. py:method:: benchmark_cell.__new__(**kwargs)
      :module: arbor
   
      Create and return a new object.  See help(type) for accurate signature.
      
   
   .. py:method:: benchmark_cell.__repr__(self: arbor.benchmark_cell) -> str
      :module: arbor
   
   
   .. py:method:: benchmark_cell.__str__(self: arbor.benchmark_cell) -> str
      :module: arbor
   

.. py:class:: binning
   :module: arbor

   Enumeration for event time binning policy.
   
   Members:
   
     none : No binning policy.
   
     regular : Round time down to multiple of binning interval.
   
     following : Round times down to previous event if within binning interval.
   
   
   .. py:method:: binning.__eq__
      :module: arbor
   
      (self: object, arg0: object) -> bool
      
   
   .. py:method:: binning.__getstate__
      :module: arbor
   
      (self: object) -> int_
      
   
   .. py:method:: binning.__hash__
      :module: arbor
   
      (self: object) -> int_
      
   
   .. py:method:: binning.__init__(self: arbor.binning, arg0: int) -> None
      :module: arbor
   
   
   .. py:method:: binning.__int__(self: arbor.binning) -> int
      :module: arbor
   
   
   .. py:attribute:: binning.__members__
      :module: arbor
      :value: {'following': binning.following, 'none': binning.none, 'regular': binning.regular}
   
   
   .. py:attribute:: binning.__module__
      :module: arbor
      :value: 'arbor'
   
   
   .. py:method:: binning.__ne__
      :module: arbor
   
      (self: object, arg0: object) -> bool
      
   
   .. py:method:: binning.__new__(**kwargs)
      :module: arbor
   
      Create and return a new object.  See help(type) for accurate signature.
      
   
   .. py:method:: binning.__repr__
      :module: arbor
   
      (self: handle) -> str
      
   
   .. py:method:: binning.__setstate__
      :module: arbor
   
      (self: arbor.binning, arg0: int) -> None
      
   
   .. py:attribute:: binning.following
      :module: arbor
      :value: binning.following
   
   
   .. py:method:: binning.name
      :module: arbor
      :property:
   
      (self: handle) -> str
      
   
   .. py:attribute:: binning.none
      :module: arbor
      :value: binning.none
   
   
   .. py:attribute:: binning.regular
      :module: arbor
      :value: binning.regular
   

.. py:class:: cable
   :module: arbor

   
   .. py:method:: cable.__init__(self: arbor.cable, branch: int, prox: float, dist: float) -> None
      :module: arbor
   
   
   .. py:attribute:: cable.__module__
      :module: arbor
      :value: 'arbor'
   
   
   .. py:method:: cable.__new__(**kwargs)
      :module: arbor
   
      Create and return a new object.  See help(type) for accurate signature.
      
   
   .. py:method:: cable.__repr__(self: arbor.cable) -> str
      :module: arbor
   
   
   .. py:method:: cable.__str__(self: arbor.cable) -> str
      :module: arbor
   
   
   .. py:method:: cable.branch
      :module: arbor
      :property:
   
      The id of the branch on which the cable lies.
      
   
   .. py:method:: cable.dist
      :module: arbor
      :property:
   
      The relative position of the distal end of the cable on its branch ∈ [0,1].
      
   
   .. py:method:: cable.prox
      :module: arbor
      :property:
   
      The relative position of the proximal end of the cable on its branch ∈ [0,1].
      

.. py:class:: cable_cell
   :module: arbor

   Represents morphologically-detailed cell models, with morphology represented as a
   tree of one-dimensional cable segments.
   
   
   .. py:method:: cable_cell.__init__(*args, **kwargs)
      :module: arbor
   
      Overloaded function.
      
      1. __init__(self: arbor.cable_cell, morphology: arb::morphology, labels: arbor.label_dict) -> None
      
      2. __init__(self: arbor.cable_cell, segment_tree: arb::segment_tree, labels: arbor.label_dict) -> None
      
      Construct with a morphology derived from a segment tree.
      
   
   .. py:attribute:: cable_cell.__module__
      :module: arbor
      :value: 'arbor'
   
   
   .. py:method:: cable_cell.__new__(**kwargs)
      :module: arbor
   
      Create and return a new object.  See help(type) for accurate signature.
      
   
   .. py:method:: cable_cell.__repr__(self: arbor.cable_cell) -> str
      :module: arbor
   
   
   .. py:method:: cable_cell.__str__(self: arbor.cable_cell) -> str
      :module: arbor
   
   
   .. py:method:: cable_cell.cables(self: arbor.cable_cell, label: str) -> List[arb::mcable]
      :module: arbor
   
      The cable segments of the cell morphology for a region label.
      
   
   .. py:method:: cable_cell.compartments_length(self: arbor.cable_cell, maxlen: float) -> None
      :module: arbor
   
      Decompose each branch into compartments of equal length, not exceeding maxlen.
      
   
   .. py:method:: cable_cell.compartments_on_segments(self: arbor.cable_cell) -> None
      :module: arbor
   
      Decompose each branch into compartments defined by segments.
      
   
   .. py:method:: cable_cell.compartments_per_branch(self: arbor.cable_cell, n: int) -> None
      :module: arbor
   
      Decompose each branch into n compartments of equal length.
      
   
   .. py:method:: cable_cell.locations(self: arbor.cable_cell, label: str) -> List[arb::mlocation]
      :module: arbor
   
      The locations of the cell morphology for a locset label.
      
   
   .. py:method:: cable_cell.num_branches
      :module: arbor
      :property:
   
      The number of unbranched cable sections in the morphology.
      
   
   .. py:method:: cable_cell.paint(*args, **kwargs)
      :module: arbor
   
      Overloaded function.
      
      1. paint(self: arbor.cable_cell, region: str, mechanism: arb::mechanism_desc) -> None
      
      Associate a mechanism with a region.
      
      2. paint(self: arbor.cable_cell, region: str, mechanism: str) -> None
      
      Associate a mechanism with a region.
      
      3. paint(self: arbor.cable_cell, region: str, Vm: Optional[float] = None, cm: Optional[float] = None, rL: Optional[float] = None, tempK: Optional[float] = None) -> None
      
      Set cable properties on a region.
       Vm:    initial membrane voltage [mV].
       cm:    membrane capacitance [F/m²].
       rL:    axial resistivity [Ω·cm].
       tempK: temperature [Kelvin].
      
      4. paint(self: arbor.cable_cell, region: str, ion_name: str, int_con: Optional[float] = Intial internal concentration [mM], ext_con: Optional[float] = Intial external concentration [mM], rev_pot: Optional[float] = Intial reversal potential [mV]) -> None
      
      Set ion species properties conditions on a region.
      
   
   .. py:method:: cable_cell.place(*args, **kwargs)
      :module: arbor
   
      Overloaded function.
      
      1. place(self: arbor.cable_cell, locations: str, mechanism: arb::mechanism_desc) -> None
      
      Place one instance of synapse described by 'mechanism' to each location in 'locations'.
      
      2. place(self: arbor.cable_cell, locations: str, mechanism: str) -> None
      
      Place one instance of synapse described by 'mechanism' to each location in 'locations'.
      
      3. place(self: arbor.cable_cell, locations: str, gapjunction: arbor.gap_junction) -> None
      
      Place one gap junction site at each location in 'locations'.
      
      4. place(self: arbor.cable_cell, locations: str, iclamp: arbor.iclamp) -> None
      
      Add a current stimulus at each location in locations.
      
      5. place(self: arbor.cable_cell, locations: str, detector: arbor.spike_detector) -> None
      
      Add a voltage spike detector at each location in locations.
      
   
   .. py:method:: cable_cell.set_ion(self: arbor.cable_cell, ion: str, int_con: Optional[float] = None, ext_con: Optional[float] = None, rev_pot: Optional[float] = None, method: Optional[arb::mechanism_desc] = None) -> None
      :module: arbor
   
      Set the propoerties of ion species named 'ion' that will be applied
      by default everywhere on the cell. Species concentrations and reversal
      potential can be overridden on specific regions using the paint interface, 
      while the method for calculating reversal potential is global for all
      compartments in the cell, and can't be overriden locally.
       ion:     name of ion species.
       int_con: initial internal concentration [mM].
       ext_con: initial external concentration [mM].
       rev_pot: reversal potential [mV].
       method:  method for calculating reversal potential.
      
   
   .. py:method:: cable_cell.set_properties(self: arbor.cable_cell, Vm: Optional[float] = None, cm: Optional[float] = None, rL: Optional[float] = None, tempK: Optional[float] = None) -> None
      :module: arbor
   
      Set default values for cable and cell properties. These values can be overridden on specific regions using the paint interface.
       Vm:    initial membrane voltage [mV].
       cm:    membrane capacitance [F/m²].
       rL:    axial resistivity [Ω·cm].
       tempK: temperature [Kelvin].
      

.. py:class:: cable_global_properties
   :module: arbor

   
   .. py:method:: cable_global_properties.__init__(*args, **kwargs)
      :module: arbor
   
      Overloaded function.
      
      1. __init__(self: arbor.cable_global_properties) -> None
      
      2. __init__(self: arbor.cable_global_properties, arg0: arbor.cable_global_properties) -> None
      
   
   .. py:attribute:: cable_global_properties.__module__
      :module: arbor
      :value: 'arbor'
   
   
   .. py:method:: cable_global_properties.__new__(**kwargs)
      :module: arbor
   
      Create and return a new object.  See help(type) for accurate signature.
      
   
   .. py:method:: cable_global_properties.__str__(self: arbor.cable_global_properties) -> str
      :module: arbor
   
   
   .. py:method:: cable_global_properties.catalogue
      :module: arbor
      :property:
   
      The mechanism catalogue.
      
   
   .. py:method:: cable_global_properties.check(self: arbor.cable_global_properties) -> None
      :module: arbor
   
      Test whether all default parameters and ion specids properties have been set.
      
   
   .. py:method:: cable_global_properties.foo(self: arbor.cable_global_properties, x: float, method: object = None) -> None
      :module: arbor
   
   
   .. py:method:: cable_global_properties.set_ion(self: arbor.cable_global_properties, ion: str, int_con: Optional[float] = None, ext_con: Optional[float] = None, rev_pot: Optional[float] = None, rev_pot_method: object = None) -> None
      :module: arbor
   
      Set the global default propoerties of ion species named 'ion'.
      Species concentrations and reversal potential can be overridden on
      specific regions using the paint interface, while the method for calculating
      reversal potential is global for all compartments in the cell, and can't be
      overriden locally.
       ion:     name of ion species.
       int_con: initial internal concentration [mM].
       ext_con: initial external concentration [mM].
       rev_pot: initial reversal potential [mV].
       rev_pot_method:  method for calculating reversal potential.
      
   
   .. py:method:: cable_global_properties.set_properties(self: arbor.cable_global_properties, Vm: Optional[float] = None, cm: Optional[float] = None, rL: Optional[float] = None, tempK: Optional[float] = None) -> None
      :module: arbor
   
      Set global default values for cable and cell properties.
       Vm:    initial membrane voltage [mV].
       cm:    membrane capacitance [F/m²].
       rL:    axial resistivity [Ω·cm].
       tempK: temperature [Kelvin].
      

.. py:function:: cable_probe(kind: str, location: arbor.location) -> arb::probe_info
   :module: arbor

   Description of a probe at a location available for monitoring data of kind where kind is one of 'voltage' or 'ionic current density'.
   

.. py:class:: cell_kind
   :module: arbor

   Enumeration used to identify the cell kind, used by the model to group equal kinds in the same cell group.
   
   Members:
   
     benchmark : Proxy cell used for benchmarking.
   
     cable : A cell with morphology described by branching 1D cable segments.
   
     lif : Leaky-integrate and fire neuron.
   
     spike_source : Proxy cell that generates spikes from a spike sequence provided by the user.
   
   
   .. py:method:: cell_kind.__eq__
      :module: arbor
   
      (self: object, arg0: object) -> bool
      
   
   .. py:method:: cell_kind.__getstate__
      :module: arbor
   
      (self: object) -> int_
      
   
   .. py:method:: cell_kind.__hash__
      :module: arbor
   
      (self: object) -> int_
      
   
   .. py:method:: cell_kind.__init__(self: arbor.cell_kind, arg0: int) -> None
      :module: arbor
   
   
   .. py:method:: cell_kind.__int__(self: arbor.cell_kind) -> int
      :module: arbor
   
   
   .. py:attribute:: cell_kind.__members__
      :module: arbor
      :value: {'benchmark': cell_kind.benchmark, 'cable': cell_kind.cable, 'lif': cell_kind.lif, 'spike_source': cell_kind.spike_source}
   
   
   .. py:attribute:: cell_kind.__module__
      :module: arbor
      :value: 'arbor'
   
   
   .. py:method:: cell_kind.__ne__
      :module: arbor
   
      (self: object, arg0: object) -> bool
      
   
   .. py:method:: cell_kind.__new__(**kwargs)
      :module: arbor
   
      Create and return a new object.  See help(type) for accurate signature.
      
   
   .. py:method:: cell_kind.__repr__
      :module: arbor
   
      (self: handle) -> str
      
   
   .. py:method:: cell_kind.__setstate__
      :module: arbor
   
      (self: arbor.cell_kind, arg0: int) -> None
      
   
   .. py:attribute:: cell_kind.benchmark
      :module: arbor
      :value: cell_kind.benchmark
   
   
   .. py:attribute:: cell_kind.cable
      :module: arbor
      :value: cell_kind.cable
   
   
   .. py:attribute:: cell_kind.lif
      :module: arbor
      :value: cell_kind.lif
   
   
   .. py:method:: cell_kind.name
      :module: arbor
      :property:
   
      (self: handle) -> str
      
   
   .. py:attribute:: cell_kind.spike_source
      :module: arbor
      :value: cell_kind.spike_source
   

.. py:class:: cell_member
   :module: arbor

   For global identification of a cell-local item.
   
   Items of cell_member must:
     (1) be associated with a unique cell, identified by the member gid;
     (2) identify an item within a cell-local collection by the member index.
   
   
   .. py:method:: cell_member.__init__(self: arbor.cell_member, gid: int, index: int) -> None
      :module: arbor
   
      Construct a cell member with arguments:
        gid:     The global identifier of the cell.
        index:   The cell-local index of the item.
      
   
   .. py:attribute:: cell_member.__module__
      :module: arbor
      :value: 'arbor'
   
   
   .. py:method:: cell_member.__new__(**kwargs)
      :module: arbor
   
      Create and return a new object.  See help(type) for accurate signature.
      
   
   .. py:method:: cell_member.__repr__(self: arbor.cell_member) -> str
      :module: arbor
   
   
   .. py:method:: cell_member.__str__(self: arbor.cell_member) -> str
      :module: arbor
   
   
   .. py:method:: cell_member.gid
      :module: arbor
      :property:
   
      The global identifier of the cell.
      
   
   .. py:method:: cell_member.index
      :module: arbor
      :property:
   
      Cell-local index of the item.
      

.. py:function:: config() -> dict
   :module: arbor

   Get Arbor's configuration.
   

.. py:class:: connection
   :module: arbor

   Describes a connection between two cells:
   Defined by source and destination end points (that is pre-synaptic and post-synaptic respectively), a connection weight and a delay time.
   
   
   .. py:method:: connection.__init__(self: arbor.connection, source: arbor.cell_member, dest: arbor.cell_member, weight: float, delay: float) -> None
      :module: arbor
   
      Construct a connection with arguments:
        source:      The source end point of the connection.
        dest:        The destination end point of the connection.
        weight:      The weight delivered to the target synapse (unit defined by the type of synapse target).
        delay:       The delay of the connection [ms].
      
   
   .. py:attribute:: connection.__module__
      :module: arbor
      :value: 'arbor'
   
   
   .. py:method:: connection.__new__(**kwargs)
      :module: arbor
   
      Create and return a new object.  See help(type) for accurate signature.
      
   
   .. py:method:: connection.__repr__(self: arbor.connection) -> str
      :module: arbor
   
   
   .. py:method:: connection.__str__(self: arbor.connection) -> str
      :module: arbor
   
   
   .. py:method:: connection.delay
      :module: arbor
      :property:
   
      The delay time of the connection [ms].
      
   
   .. py:method:: connection.dest
      :module: arbor
      :property:
   
      The destination of the connection.
      
   
   .. py:method:: connection.source
      :module: arbor
      :property:
   
      The source of the connection.
      
   
   .. py:method:: connection.weight
      :module: arbor
      :property:
   
      The weight of the connection.
      

.. py:class:: context
   :module: arbor

   An opaque handle for the hardware resources used in a simulation.
   
   
   .. py:method:: context.__init__(*args, **kwargs)
      :module: arbor
   
      Overloaded function.
      
      1. __init__(self: arbor.context) -> None
      
      Construct a local context with one thread, no GPU, no MPI by default.
      
      
      2. __init__(self: arbor.context, alloc: arbor.proc_allocation) -> None
      
      Construct a local context with argument:
        alloc:   The computational resources to be used for the simulation.
      
      
      3. __init__(self: arbor.context, threads: int = 1, gpu_id: object = None) -> None
      
      Construct a local context with arguments:
        threads: The number of threads available locally for execution, 1 by default.
        gpu_id:  The identifier of the GPU to use, None by default.
      
   
   .. py:attribute:: context.__module__
      :module: arbor
      :value: 'arbor'
   
   
   .. py:method:: context.__new__(**kwargs)
      :module: arbor
   
      Create and return a new object.  See help(type) for accurate signature.
      
   
   .. py:method:: context.__repr__(self: arbor.context) -> str
      :module: arbor
   
   
   .. py:method:: context.__str__(self: arbor.context) -> str
      :module: arbor
   
   
   .. py:method:: context.has_gpu
      :module: arbor
      :property:
   
      Whether the context has a GPU.
      
   
   .. py:method:: context.has_mpi
      :module: arbor
      :property:
   
      Whether the context uses MPI for distributed communication.
      
   
   .. py:method:: context.rank
      :module: arbor
      :property:
   
      The numeric id of the local domain (equivalent to MPI rank).
      
   
   .. py:method:: context.ranks
      :module: arbor
      :property:
   
      The number of distributed domains (equivalent to the number of MPI ranks).
      
   
   .. py:method:: context.threads
      :module: arbor
      :property:
   
      The number of threads in the context's thread pool.
      

.. py:function:: default_catalogue() -> arbor.mechanism_catalogue
   :module: arbor


.. py:class:: domain_decomposition
   :module: arbor

   The domain decomposition is responsible for describing the distribution of cells across cell groups and domains.
   
   
   .. py:method:: domain_decomposition.__init__(self: arbor.domain_decomposition) -> None
      :module: arbor
   
   
   .. py:attribute:: domain_decomposition.__module__
      :module: arbor
      :value: 'arbor'
   
   
   .. py:method:: domain_decomposition.__new__(**kwargs)
      :module: arbor
   
      Create and return a new object.  See help(type) for accurate signature.
      
   
   .. py:method:: domain_decomposition.__repr__(self: arbor.domain_decomposition) -> str
      :module: arbor
   
   
   .. py:method:: domain_decomposition.__str__(self: arbor.domain_decomposition) -> str
      :module: arbor
   
   
   .. py:method:: domain_decomposition.domain_id
      :module: arbor
      :property:
   
      The index of the local domain.
      Always 0 for non-distributed models, and corresponds to the MPI rank for distributed runs.
      
   
   .. py:method:: domain_decomposition.gid_domain(self: arbor.domain_decomposition, gid: int) -> int
      :module: arbor
   
      Query the domain id that a cell assigned to (using global identifier gid).
      
   
   .. py:method:: domain_decomposition.groups
      :module: arbor
      :property:
   
      Descriptions of the cell groups on the local domain.
      
   
   .. py:method:: domain_decomposition.num_domains
      :module: arbor
      :property:
   
      Number of domains that the model is distributed over.
      
   
   .. py:method:: domain_decomposition.num_global_cells
      :module: arbor
      :property:
   
      Total number of cells in the global model (sum of num_local_cells over all domains).
      
   
   .. py:method:: domain_decomposition.num_local_cells
      :module: arbor
      :property:
   
      Total number of cells in the local domain.
      

.. py:class:: event_generator
   :module: arbor

   
   .. py:method:: event_generator.__init__(*args, **kwargs)
      :module: arbor
   
      Overloaded function.
      
      1. __init__(self: arbor.event_generator, target: arb::cell_member_type, weight: float, sched: pyarb::regular_schedule_shim) -> None
      
      Construct an event generator with arguments:
        target: The target synapse (gid, local_id).
        weight: The weight of events to deliver.
        sched:  A regular schedule of the events.
      
      2. __init__(self: arbor.event_generator, target: arb::cell_member_type, weight: float, sched: pyarb::explicit_schedule_shim) -> None
      
      Construct an event generator with arguments:
        target: The target synapse (gid, local_id).
        weight: The weight of events to deliver.
        sched:  An explicit schedule of the events.
      
      3. __init__(self: arbor.event_generator, target: arb::cell_member_type, weight: float, sched: pyarb::poisson_schedule_shim) -> None
      
      Construct an event generator with arguments:
        target: The target synapse (gid, local_id).
        weight: The weight of events to deliver.
        sched:  A poisson schedule of the events.
      
   
   .. py:attribute:: event_generator.__module__
      :module: arbor
      :value: 'arbor'
   
   
   .. py:method:: event_generator.__new__(**kwargs)
      :module: arbor
   
      Create and return a new object.  See help(type) for accurate signature.
      
   
   .. py:method:: event_generator.__repr__(self: arbor.event_generator) -> str
      :module: arbor
   
   
   .. py:method:: event_generator.__str__(self: arbor.event_generator) -> str
      :module: arbor
   
   
   .. py:method:: event_generator.target
      :module: arbor
      :property:
   
      The target synapse (gid, local_id).
      
   
   .. py:method:: event_generator.weight
      :module: arbor
      :property:
   
      The weight of events to deliver.
      

.. py:class:: explicit_schedule
   :module: arbor

   Describes an explicit schedule at a predetermined (sorted) sequence of times.
   
   
   .. py:method:: explicit_schedule.__init__(*args, **kwargs)
      :module: arbor
   
      Overloaded function.
      
      1. __init__(self: arbor.explicit_schedule) -> None
      
      Construct an empty explicit schedule.
      
      
      2. __init__(self: arbor.explicit_schedule, times: List[float]) -> None
      
      Construct an explicit schedule with argument:
        times: A list of times [ms], [] by default.
      
   
   .. py:attribute:: explicit_schedule.__module__
      :module: arbor
      :value: 'arbor'
   
   
   .. py:method:: explicit_schedule.__new__(**kwargs)
      :module: arbor
   
      Create and return a new object.  See help(type) for accurate signature.
      
   
   .. py:method:: explicit_schedule.__repr__(self: arbor.explicit_schedule) -> str
      :module: arbor
   
   
   .. py:method:: explicit_schedule.__str__(self: arbor.explicit_schedule) -> str
      :module: arbor
   
   
   .. py:method:: explicit_schedule.events(self: arbor.explicit_schedule, arg0: float, arg1: float) -> List[float]
      :module: arbor
   
      A view of monotonically increasing time values in the half-open interval [t0, t1).
      
   
   .. py:method:: explicit_schedule.times
      :module: arbor
      :property:
   
      A list of times [ms].
      

.. py:class:: flat_cell_builder
   :module: arbor

   
   .. py:method:: flat_cell_builder.__init__(self: arbor.flat_cell_builder) -> None
      :module: arbor
   
   
   .. py:attribute:: flat_cell_builder.__module__
      :module: arbor
      :value: 'arbor'
   
   
   .. py:method:: flat_cell_builder.__new__(**kwargs)
      :module: arbor
   
      Create and return a new object.  See help(type) for accurate signature.
      
   
   .. py:method:: flat_cell_builder.add_cable(*args, **kwargs)
      :module: arbor
   
      Overloaded function.
      
      1. add_cable(self: arbor.flat_cell_builder, length: float, radius: object, name: str, ncomp: int = 1) -> int
      
      2. add_cable(self: arbor.flat_cell_builder, parent: int, length: float, radius: object, name: str, ncomp: int = 1) -> int
      
   
   .. py:method:: flat_cell_builder.add_label(self: arbor.flat_cell_builder, name: str, description: str) -> None
      :module: arbor
   
   
   .. py:method:: flat_cell_builder.build(self: arbor.flat_cell_builder) -> arbor.cable_cell
      :module: arbor
   
   
   .. py:method:: flat_cell_builder.labels
      :module: arbor
      :property:
   
   
   .. py:method:: flat_cell_builder.morphology
      :module: arbor
      :property:
   
   
   .. py:method:: flat_cell_builder.segments
      :module: arbor
      :property:
   

.. py:class:: gap_junction
   :module: arbor

   For marking a location on a cell morphology as a gap junction site.
   
   
   .. py:method:: gap_junction.__init__(self: arbor.gap_junction) -> None
      :module: arbor
   
   
   .. py:attribute:: gap_junction.__module__
      :module: arbor
      :value: 'arbor'
   
   
   .. py:method:: gap_junction.__new__(**kwargs)
      :module: arbor
   
      Create and return a new object.  See help(type) for accurate signature.
      
   
   .. py:method:: gap_junction.__repr__(self: arbor.gap_junction) -> str
      :module: arbor
   
   
   .. py:method:: gap_junction.__str__(self: arbor.gap_junction) -> str
      :module: arbor
   

.. py:class:: gap_junction_connection
   :module: arbor

   Describes a gap junction between two gap junction sites.
   
   
   .. py:method:: gap_junction_connection.__init__(self: arbor.gap_junction_connection, local: arbor.cell_member, peer: arbor.cell_member, ggap: float) -> None
      :module: arbor
   
      Construct a gap junction connection with arguments:
        local: One half of the gap junction connection.
        peer:  Other half of the gap junction connection.
        ggap:  Gap junction conductance [μS].
      
   
   .. py:attribute:: gap_junction_connection.__module__
      :module: arbor
      :value: 'arbor'
   
   
   .. py:method:: gap_junction_connection.__new__(**kwargs)
      :module: arbor
   
      Create and return a new object.  See help(type) for accurate signature.
      
   
   .. py:method:: gap_junction_connection.__repr__(self: arbor.gap_junction_connection) -> str
      :module: arbor
   
   
   .. py:method:: gap_junction_connection.__str__(self: arbor.gap_junction_connection) -> str
      :module: arbor
   
   
   .. py:method:: gap_junction_connection.ggap
      :module: arbor
      :property:
   
      Gap junction conductance [μS].
      
   
   .. py:method:: gap_junction_connection.local
      :module: arbor
      :property:
   
      One half of the gap junction connection.
      
   
   .. py:method:: gap_junction_connection.peer
      :module: arbor
      :property:
   
      Other half of the gap junction connection.
      

.. py:class:: group_description
   :module: arbor

   The indexes of a set of cells of the same kind that are grouped together in a cell group.
   
   
   .. py:method:: group_description.__init__(self: arbor.group_description, kind: arb::cell_kind, gids: List[int], backend: arb::backend_kind) -> None
      :module: arbor
   
      Construct a group description with cell kind, list of gids, and backend kind.
      
   
   .. py:attribute:: group_description.__module__
      :module: arbor
      :value: 'arbor'
   
   
   .. py:method:: group_description.__new__(**kwargs)
      :module: arbor
   
      Create and return a new object.  See help(type) for accurate signature.
      
   
   .. py:method:: group_description.__repr__(self: arbor.group_description) -> str
      :module: arbor
   
   
   .. py:method:: group_description.__str__(self: arbor.group_description) -> str
      :module: arbor
   
   
   .. py:method:: group_description.backend
      :module: arbor
      :property:
   
      The hardware backend on which the cell group will run.
      
   
   .. py:method:: group_description.gids
      :module: arbor
      :property:
   
      The list of gids of the cells in the group.
      
   
   .. py:method:: group_description.kind
      :module: arbor
      :property:
   
      The type of cell in the cell group.
      

.. py:class:: iclamp
   :module: arbor

   A current clamp, for injecting a single pulse of current with fixed duration and current.
   
   
   .. py:method:: iclamp.__init__(self: arbor.iclamp, tstart: float = 0, duration: float = 0, current: float = 0) -> None
      :module: arbor
   
   
   .. py:attribute:: iclamp.__module__
      :module: arbor
      :value: 'arbor'
   
   
   .. py:method:: iclamp.__new__(**kwargs)
      :module: arbor
   
      Create and return a new object.  See help(type) for accurate signature.
      
   
   .. py:method:: iclamp.__repr__(self: arbor.iclamp) -> str
      :module: arbor
   
   
   .. py:method:: iclamp.__str__(self: arbor.iclamp) -> str
      :module: arbor
   
   
   .. py:method:: iclamp.current
      :module: arbor
      :property:
   
      Amplitude of the injected current [nA]
      
   
   .. py:method:: iclamp.duration
      :module: arbor
      :property:
   
      Duration of the current injection [ms]
      
   
   .. py:method:: iclamp.tstart
      :module: arbor
      :property:
   
      Time at which current starts [ms]
      

.. py:class:: ion
   :module: arbor

   For setting ion properties (internal and external concentration and reversal potential) on cells and regions.
   
   
   .. py:method:: ion.__init__(self: arbor.ion, ion_name: str, int_con: Optional[float] = Intial internal concentration [mM], ext_con: Optional[float] = Intial external concentration [mM], rev_pot: Optional[float] = Intial reversal potential [mV]) -> None
      :module: arbor
   
      If concentrations or reversal potential are specified as 'None', cell default or global default value will be used, in that order if set.
      
   
   .. py:attribute:: ion.__module__
      :module: arbor
      :value: 'arbor'
   
   
   .. py:method:: ion.__new__(**kwargs)
      :module: arbor
   
      Create and return a new object.  See help(type) for accurate signature.
      

.. py:class:: ion_dependency
   :module: arbor

   Information about a mechanism's dependence on an ion species.
   
   
   .. py:method:: ion_dependency.__init__(self: arbor.ion_dependency, arg0: arbor.ion_dependency) -> None
      :module: arbor
   
   
   .. py:attribute:: ion_dependency.__module__
      :module: arbor
      :value: 'arbor'
   
   
   .. py:method:: ion_dependency.__new__(**kwargs)
      :module: arbor
   
      Create and return a new object.  See help(type) for accurate signature.
      
   
   .. py:method:: ion_dependency.__repr__(self: arbor.ion_dependency) -> str
      :module: arbor
   
   
   .. py:method:: ion_dependency.__str__(self: arbor.ion_dependency) -> str
      :module: arbor
   
   
   .. py:method:: ion_dependency.read_rev_pot
      :module: arbor
      :property:
   
   
   .. py:method:: ion_dependency.write_ext_con
      :module: arbor
      :property:
   
   
   .. py:method:: ion_dependency.write_int_con
      :module: arbor
      :property:
   
   
   .. py:method:: ion_dependency.write_rev_pot
      :module: arbor
      :property:
   

.. py:class:: label_dict
   :module: arbor

   A dictionary of labelled region and locset definitions, with a
   unique label is assigned to each definition.
   
   
   .. py:method:: label_dict.__getitem__(self: arbor.label_dict, arg0: str) -> str
      :module: arbor
   
   
   .. py:method:: label_dict.__init__(*args, **kwargs)
      :module: arbor
   
      Overloaded function.
      
      1. __init__(self: arbor.label_dict) -> None
      
      Create an empty label dictionary.
      
      2. __init__(self: arbor.label_dict, arg0: Dict[str, str]) -> None
      
      Initialize a label dictionary from a dictionary with string labels as keys, and corresponding definitions as strings.
      
   
   .. py:method:: label_dict.__iter__(self: arbor.label_dict) -> iterator
      :module: arbor
   
   
   .. py:method:: label_dict.__len__(self: arbor.label_dict) -> int
      :module: arbor
   
   
   .. py:attribute:: label_dict.__module__
      :module: arbor
      :value: 'arbor'
   
   
   .. py:method:: label_dict.__new__(**kwargs)
      :module: arbor
   
      Create and return a new object.  See help(type) for accurate signature.
      
   
   .. py:method:: label_dict.__repr__(self: arbor.label_dict) -> str
      :module: arbor
   
   
   .. py:method:: label_dict.__setitem__(self: arbor.label_dict, arg0: str, arg1: str) -> None
      :module: arbor
   
   
   .. py:method:: label_dict.__str__(self: arbor.label_dict) -> str
      :module: arbor
   
   
   .. py:method:: label_dict.locsets
      :module: arbor
      :property:
   
      The locset definitions.
      
   
   .. py:method:: label_dict.regions
      :module: arbor
      :property:
   
      The region definitions.
      

.. py:class:: lif_cell
   :module: arbor

   A benchmarking cell, used by Arbor developers to test communication performance.
   
   
   .. py:method:: lif_cell.C_m
      :module: arbor
      :property:
   
      Membrane capacitance [pF].
      
   
   .. py:method:: lif_cell.E_L
      :module: arbor
      :property:
   
      Resting potential [mV].
      
   
   .. py:method:: lif_cell.V_m
      :module: arbor
      :property:
   
      Initial value of the Membrane potential [mV].
      
   
   .. py:method:: lif_cell.V_reset
      :module: arbor
      :property:
   
      Reset potential [mV].
      
   
   .. py:method:: lif_cell.V_th
      :module: arbor
      :property:
   
      Firing threshold [mV].
      
   
   .. py:method:: lif_cell.__init__(self: arbor.lif_cell) -> None
      :module: arbor
   
   
   .. py:attribute:: lif_cell.__module__
      :module: arbor
      :value: 'arbor'
   
   
   .. py:method:: lif_cell.__new__(**kwargs)
      :module: arbor
   
      Create and return a new object.  See help(type) for accurate signature.
      
   
   .. py:method:: lif_cell.__repr__(self: arbor.lif_cell) -> str
      :module: arbor
   
   
   .. py:method:: lif_cell.__str__(self: arbor.lif_cell) -> str
      :module: arbor
   
   
   .. py:method:: lif_cell.t_ref
      :module: arbor
      :property:
   
      Refractory period [ms].
      
   
   .. py:method:: lif_cell.tau_m
      :module: arbor
      :property:
   
      Membrane potential decaying constant [ms].
      

.. py:function:: load_swc(arg0: str) -> arbor.segment_tree
   :module: arbor

   Load an swc file and as a segment_tree.
   

.. py:function:: load_swc_allen(filename: str, no_gaps: bool = False) -> arbor.segment_tree
   :module: arbor

   Generate a segment tree from an SWC file following the rules prescribed by
   AllenDB and Sonata. Specifically:
   * The first sample (the root) is treated as the center of the soma.
   * The first morphology is translated such that the soma is centered at (0,0,0).
   * The first sample has tag 1 (soma).
   * All other samples have tags 2, 3 or 4 (axon, apic and dend respectively)
   SONATA prescribes that there should be no gaps, however the models in AllenDB
   have gaps between the start of sections and the soma. The flag no_gaps can be
   used to enforce this requirement.
   
   Arbor does not support modelling the soma as a sphere, so a cylinder with length
   equal to the soma diameter is used. The cylinder is centered on the origin, and
   aligned along the z axis.
   Axons and apical dendrites are attached to the proximal end of the cylinder, and
   dendrites to the distal end, with a gap between the start of each branch and the
   end of the soma cylinder to which it is attached.
   

.. py:class:: location
   :module: arbor

   A location on a cable cell.
   
   
   .. py:method:: location.__init__(self: arbor.location, branch: int, pos: float) -> None
      :module: arbor
   
      Construct a location specification holding:
        branch:   The id of the branch.
        pos:      The relative position (from 0., proximal, to 1., distal) on the branch.
      
   
   .. py:attribute:: location.__module__
      :module: arbor
      :value: 'arbor'
   
   
   .. py:method:: location.__new__(**kwargs)
      :module: arbor
   
      Create and return a new object.  See help(type) for accurate signature.
      
   
   .. py:method:: location.__repr__(self: arbor.location) -> str
      :module: arbor
   
   
   .. py:method:: location.__str__(self: arbor.location) -> str
      :module: arbor
   
   
   .. py:method:: location.branch
      :module: arbor
      :property:
   
      The id of the branch.
      
   
   .. py:method:: location.pos
      :module: arbor
      :property:
   
      The relative position on the branch (∈ [0.,1.], where 0. means proximal and 1. distal).
      

.. py:class:: mechanism
   :module: arbor

   
   .. py:method:: mechanism.__init__(*args, **kwargs)
      :module: arbor
   
      Overloaded function.
      
      1. __init__(self: arbor.mechanism, arg0: str) -> None
      
      2. __init__(self: arbor.mechanism, name: str, params: Dict[str, float]) -> None
      
      Example usage setting parameters:
        m = arbor.mechanism('expsyn', {'tau': 1.4})
      will create parameters for the 'expsyn' mechanism, with the provided value
      for 'tau' overrides the default. If a parameter is not set, the default
      (as defined in NMODL) is used.
      
      Example overriding a global parameter:
        m = arbor.mechanism('nernst/R=8.3145,F=96485')
      
   
   .. py:attribute:: mechanism.__module__
      :module: arbor
      :value: 'arbor'
   
   
   .. py:method:: mechanism.__new__(**kwargs)
      :module: arbor
   
      Create and return a new object.  See help(type) for accurate signature.
      
   
   .. py:method:: mechanism.__repr__(self: arbor.mechanism) -> str
      :module: arbor
   
   
   .. py:method:: mechanism.__str__(self: arbor.mechanism) -> str
      :module: arbor
   
   
   .. py:method:: mechanism.name
      :module: arbor
      :property:
   
      The name of the mechanism.
      
   
   .. py:method:: mechanism.set(self: arbor.mechanism, name: str, value: float) -> None
      :module: arbor
   
      Set parameter value.
      
   
   .. py:method:: mechanism.values
      :module: arbor
      :property:
   
      A dictionary of parameter values with parameter name as key.
      

.. py:class:: mechanism_catalogue
   :module: arbor

   
   .. py:method:: mechanism_catalogue.__getitem__(self: arbor.mechanism_catalogue, arg0: str) -> arbor.mechanism_info
      :module: arbor
   
   
   .. py:method:: mechanism_catalogue.__init__(self: arbor.mechanism_catalogue, arg0: arbor.mechanism_catalogue) -> None
      :module: arbor
   
   
   .. py:attribute:: mechanism_catalogue.__module__
      :module: arbor
      :value: 'arbor'
   
   
   .. py:method:: mechanism_catalogue.__new__(**kwargs)
      :module: arbor
   
      Create and return a new object.  See help(type) for accurate signature.
      
   
   .. py:method:: mechanism_catalogue.__repr__(self: arbor.mechanism_catalogue) -> str
      :module: arbor
   
   
   .. py:method:: mechanism_catalogue.__str__(self: arbor.mechanism_catalogue) -> str
      :module: arbor
   
   
   .. py:method:: mechanism_catalogue.derive(self: arbor.mechanism_catalogue, name: str, parent: str, globals: Dict[str, float] = {}, ions: Dict[str, str] = {}) -> None
      :module: arbor
   
   
   .. py:method:: mechanism_catalogue.extend(self: arbor.mechanism_catalogue, other: arbor.mechanism_catalogue, prefix: str) -> None
      :module: arbor
   
      Import another catalogue, possibly with a prefix. Will overwrite in case of name collisions.
      
   
   .. py:method:: mechanism_catalogue.has(self: arbor.mechanism_catalogue, name: str) -> bool
      :module: arbor
   
      Is 'name' in the catalogue?
      
   
   .. py:method:: mechanism_catalogue.is_derived(self: arbor.mechanism_catalogue, name: str) -> bool
      :module: arbor
   
      Is 'name' a derived mechanism or can it be implicitly derived?
      

.. py:class:: mechanism_field
   :module: arbor

   Basic information about a mechanism field.
   
   
   .. py:method:: mechanism_field.__init__(self: arbor.mechanism_field, arg0: arbor.mechanism_field) -> None
      :module: arbor
   
   
   .. py:attribute:: mechanism_field.__module__
      :module: arbor
      :value: 'arbor'
   
   
   .. py:method:: mechanism_field.__new__(**kwargs)
      :module: arbor
   
      Create and return a new object.  See help(type) for accurate signature.
      
   
   .. py:method:: mechanism_field.__repr__(self: arbor.mechanism_field) -> str
      :module: arbor
   
   
   .. py:method:: mechanism_field.__str__(self: arbor.mechanism_field) -> str
      :module: arbor
   
   
   .. py:method:: mechanism_field.default
      :module: arbor
      :property:
   
   
   .. py:method:: mechanism_field.max
      :module: arbor
      :property:
   
   
   .. py:method:: mechanism_field.min
      :module: arbor
      :property:
   
   
   .. py:method:: mechanism_field.units
      :module: arbor
      :property:
   

.. py:class:: mechanism_info
   :module: arbor

   Meta data about a mechanism's fields and ion dependendencies.
   
   
   .. py:method:: mechanism_info.__init__(self: arbor.mechanism_info, arg0: arbor.mechanism_info) -> None
      :module: arbor
   
   
   .. py:attribute:: mechanism_info.__module__
      :module: arbor
      :value: 'arbor'
   
   
   .. py:method:: mechanism_info.__new__(**kwargs)
      :module: arbor
   
      Create and return a new object.  See help(type) for accurate signature.
      
   
   .. py:method:: mechanism_info.__repr__(self: arbor.mechanism_info) -> str
      :module: arbor
   
   
   .. py:method:: mechanism_info.__str__(self: arbor.mechanism_info) -> str
      :module: arbor
   
   
   .. py:method:: mechanism_info.globals
      :module: arbor
      :property:
   
      Global fields have one value common to an instance of a mechanism, are constant in time and set at instantiation.
      
   
   .. py:method:: mechanism_info.ions
      :module: arbor
      :property:
   
      Ion dependencies.
      
   
   .. py:method:: mechanism_info.linear
      :module: arbor
      :property:
   
      True if a synapse mechanism has linear current contributions so that multiple instances on the same compartment can be coalesed.
      
   
   .. py:method:: mechanism_info.parameters
      :module: arbor
      :property:
   
      Parameter fields may vary across the extent of a mechanism, but are constant in time and set at instantiation.
      
   
   .. py:method:: mechanism_info.state
      :module: arbor
      :property:
   
      State fields vary in time and across the extent of a mechanism, and potentially can be sampled at run-time.
      

.. py:class:: meter_manager
   :module: arbor

   Manage metering by setting checkpoints and starting the timing region.
   
   
   .. py:method:: meter_manager.__init__(self: arbor.meter_manager) -> None
      :module: arbor
   
   
   .. py:attribute:: meter_manager.__module__
      :module: arbor
      :value: 'arbor'
   
   
   .. py:method:: meter_manager.__new__(**kwargs)
      :module: arbor
   
      Create and return a new object.  See help(type) for accurate signature.
      
   
   .. py:method:: meter_manager.__repr__(self: arbor.meter_manager) -> str
      :module: arbor
   
   
   .. py:method:: meter_manager.__str__(self: arbor.meter_manager) -> str
      :module: arbor
   
   
   .. py:method:: meter_manager.checkpoint(self: arbor.meter_manager, name: str, context: arbor.context) -> None
      :module: arbor
   
      Create a new checkpoint. Records the time since the last checkpoint             (or the call to start if no previous checkpoints exist),             and restarts the timer for the next checkpoint.
      
   
   .. py:method:: meter_manager.checkpoint_names
      :module: arbor
      :property:
   
      A list of all metering checkpoint names.
      
   
   .. py:method:: meter_manager.start(self: arbor.meter_manager, context: arbor.context) -> None
      :module: arbor
   
      Start the metering. Records a time stamp,             that marks the start of the first checkpoint timing region.
      
   
   .. py:method:: meter_manager.times
      :module: arbor
      :property:
   
      A list of all metering times.
      

.. py:class:: meter_report
   :module: arbor

   Summarises the performance meter results, used to print a report to screen or file.
   If a distributed context is used, the report will contain a summary of results from all MPI ranks.
   
   
   .. py:method:: meter_report.__init__(self: arbor.meter_report, manager: arbor.meter_manager, context: arbor.context) -> None
      :module: arbor
   
   
   .. py:attribute:: meter_report.__module__
      :module: arbor
      :value: 'arbor'
   
   
   .. py:method:: meter_report.__new__(**kwargs)
      :module: arbor
   
      Create and return a new object.  See help(type) for accurate signature.
      
   
   .. py:method:: meter_report.__repr__(self: arbor.meter_report) -> str
      :module: arbor
   
   
   .. py:method:: meter_report.__str__(self: arbor.meter_report) -> str
      :module: arbor
   

.. py:class:: morphology
   :module: arbor

   
   .. py:method:: morphology.__init__(self: arbor.morphology, arg0: arbor.segment_tree) -> None
      :module: arbor
   
   
   .. py:attribute:: morphology.__module__
      :module: arbor
      :value: 'arbor'
   
   
   .. py:method:: morphology.__new__(**kwargs)
      :module: arbor
   
      Create and return a new object.  See help(type) for accurate signature.
      
   
   .. py:method:: morphology.__str__(self: arbor.morphology) -> str
      :module: arbor
   
   
   .. py:method:: morphology.branch_children(self: arbor.morphology, i: int) -> List[int]
      :module: arbor
   
      The child branches of branch i.
      
   
   .. py:method:: morphology.branch_parent(self: arbor.morphology, i: int) -> int
      :module: arbor
   
      The parent branch of branch i.
      
   
   .. py:method:: morphology.branch_segments(self: arbor.morphology, i: int) -> List[arbor.msegment]
      :module: arbor
   
      A list of the segments in branch i, ordered from proximal to distal ends of the branch.
      
   
   .. py:method:: morphology.empty
      :module: arbor
      :property:
   
      Whether the morphology is empty.
      
   
   .. py:method:: morphology.num_branches
      :module: arbor
      :property:
   
      The number of branches in the morphology.
      

.. py:class:: mpoint
   :module: arbor

   
   .. py:method:: mpoint.__init__(self: arbor.mpoint, x: float, y: float, z: float, radius: float) -> None
      :module: arbor
   
      All values in μm.
      
   
   .. py:attribute:: mpoint.__module__
      :module: arbor
      :value: 'arbor'
   
   
   .. py:method:: mpoint.__new__(**kwargs)
      :module: arbor
   
      Create and return a new object.  See help(type) for accurate signature.
      
   
   .. py:method:: mpoint.__repr__(self: arbor.mpoint) -> str
      :module: arbor
   
   
   .. py:method:: mpoint.__str__(self: arbor.mpoint) -> str
      :module: arbor
   
   
   .. py:method:: mpoint.radius
      :module: arbor
      :property:
   
      Radius of cable at sample location centered at coordinates [μm].
      
   
   .. py:method:: mpoint.x
      :module: arbor
      :property:
   
      X coordinate [μm].
      
   
   .. py:method:: mpoint.y
      :module: arbor
      :property:
   
      Y coordinate [μm].
      
   
   .. py:method:: mpoint.z
      :module: arbor
      :property:
   
      Z coordinate [μm].
      

.. py:class:: msegment
   :module: arbor

   
   .. py:method:: msegment.__init__(*args, **kwargs)
      :module: arbor
   
      Initialize self.  See help(type(self)) for accurate signature.
      
   
   .. py:attribute:: msegment.__module__
      :module: arbor
      :value: 'arbor'
   
   
   .. py:method:: msegment.__new__(**kwargs)
      :module: arbor
   
      Create and return a new object.  See help(type) for accurate signature.
      
   
   .. py:method:: msegment.dist
      :module: arbor
      :property:
   
      the location and radius of the distal end.
      
   
   .. py:method:: msegment.prox
      :module: arbor
      :property:
   
      the location and radius of the proximal end.
      
   
   .. py:method:: msegment.tag
      :module: arbor
      :property:
   
      tag meta-data.
      

.. py:class:: partition_hint
   :module: arbor

   Provide a hint on how the cell groups should be partitioned.
   
   
   .. py:method:: partition_hint.__init__(self: arbor.partition_hint, cpu_group_size: int = 1, gpu_group_size: int = 18446744073709551615, prefer_gpu: bool = True) -> None
      :module: arbor
   
      Construct a partition hint with arguments:
        cpu_group_size: The size of cell group assigned to CPU, each cell in its own group by default.
                        Must be positive, else set to default value.
        gpu_group_size: The size of cell group assigned to GPU, all cells in one group by default.
                        Must be positive, else set to default value.
        prefer_gpu:     Whether GPU is preferred, True by default.
      
   
   .. py:attribute:: partition_hint.__module__
      :module: arbor
      :value: 'arbor'
   
   
   .. py:method:: partition_hint.__new__(**kwargs)
      :module: arbor
   
      Create and return a new object.  See help(type) for accurate signature.
      
   
   .. py:method:: partition_hint.__repr__(self: arbor.partition_hint) -> str
      :module: arbor
   
   
   .. py:method:: partition_hint.__str__(self: arbor.partition_hint) -> str
      :module: arbor
   
   
   .. py:method:: partition_hint.cpu_group_size
      :module: arbor
      :property:
   
      The size of cell group assigned to CPU.
      
   
   .. py:method:: partition_hint.gpu_group_size
      :module: arbor
      :property:
   
      The size of cell group assigned to GPU.
      
   
   .. py:attribute:: partition_hint.max_size
      :module: arbor
      :value: 18446744073709551615
   
   
   .. py:method:: partition_hint.prefer_gpu
      :module: arbor
      :property:
   
      Whether GPU usage is preferred.
      

.. py:function:: partition_load_balance(recipe: pyarb::py_recipe, context: arbor.context, hints: Dict[arb::cell_kind, arbor.partition_hint] = {}) -> arbor.domain_decomposition
   :module: arbor

   Construct a domain_decomposition that distributes the cells in the model described by recipe
   over the distributed and local hardware resources described by context.
   Optionally, provide a dictionary of partition hints for certain cell kinds, by default empty.
   

.. py:class:: poisson_schedule
   :module: arbor

   Describes a schedule according to a Poisson process.
   
   
   .. py:method:: poisson_schedule.__init__(self: arbor.poisson_schedule, tstart: float = 0.0, freq: float = 10.0, seed: int = 0) -> None
      :module: arbor
   
      Construct a Poisson schedule with arguments:
        tstart: The delivery time of the first event in the sequence [ms], 0 by default.
        freq:   The expected frequency [Hz], 10 by default.
        seed:   The seed for the random number generator, 0 by default.
      
   
   .. py:attribute:: poisson_schedule.__module__
      :module: arbor
      :value: 'arbor'
   
   
   .. py:method:: poisson_schedule.__new__(**kwargs)
      :module: arbor
   
      Create and return a new object.  See help(type) for accurate signature.
      
   
   .. py:method:: poisson_schedule.__repr__(self: arbor.poisson_schedule) -> str
      :module: arbor
   
   
   .. py:method:: poisson_schedule.__str__(self: arbor.poisson_schedule) -> str
      :module: arbor
   
   
   .. py:method:: poisson_schedule.events(self: arbor.poisson_schedule, arg0: float, arg1: float) -> List[float]
      :module: arbor
   
      A view of monotonically increasing time values in the half-open interval [t0, t1).
      
   
   .. py:method:: poisson_schedule.freq
      :module: arbor
      :property:
   
      The expected frequency [Hz].
      
   
   .. py:method:: poisson_schedule.seed
      :module: arbor
      :property:
   
      The seed for the random number generator.
      
   
   .. py:method:: poisson_schedule.tstart
      :module: arbor
      :property:
   
      The delivery time of the first event in the sequence [ms].
      

.. py:function:: print_config(arg0: dict) -> None
   :module: arbor

   Print Arbor's configuration.
   

.. py:class:: probe
   :module: arbor

   
   .. py:method:: probe.__init__(*args, **kwargs)
      :module: arbor
   
      Initialize self.  See help(type(self)) for accurate signature.
      
   
   .. py:attribute:: probe.__module__
      :module: arbor
      :value: 'arbor'
   
   
   .. py:method:: probe.__new__(**kwargs)
      :module: arbor
   
      Create and return a new object.  See help(type) for accurate signature.
      
   
   .. py:method:: probe.__repr__(self: arbor.probe) -> str
      :module: arbor
   
   
   .. py:method:: probe.__str__(self: arbor.probe) -> str
      :module: arbor
   

.. py:class:: proc_allocation
   :module: arbor

   Enumerates the computational resources on a node to be used for simulation.
   
   
   .. py:method:: proc_allocation.__init__(self: arbor.proc_allocation, threads: int = 1, gpu_id: object = None) -> None
      :module: arbor
   
      Construct an allocation with arguments:
        threads: The number of threads available locally for execution, 1 by default.
        gpu_id:  The identifier of the GPU to use, None by default.
      
   
   .. py:attribute:: proc_allocation.__module__
      :module: arbor
      :value: 'arbor'
   
   
   .. py:method:: proc_allocation.__new__(**kwargs)
      :module: arbor
   
      Create and return a new object.  See help(type) for accurate signature.
      
   
   .. py:method:: proc_allocation.__repr__(self: arbor.proc_allocation) -> str
      :module: arbor
   
   
   .. py:method:: proc_allocation.__str__(self: arbor.proc_allocation) -> str
      :module: arbor
   
   
   .. py:method:: proc_allocation.gpu_id
      :module: arbor
      :property:
   
      The identifier of the GPU to use.
      Corresponds to the integer parameter used to identify GPUs in CUDA API calls.
      
   
   .. py:method:: proc_allocation.has_gpu
      :module: arbor
      :property:
   
      Whether a GPU is being used (True/False).
      
   
   .. py:method:: proc_allocation.threads
      :module: arbor
      :property:
   
      The number of threads available locally for execution.
      

.. py:class:: recipe
   :module: arbor

   A description of a model, describing the cells and the network via a cell-centric interface.
   
   
   .. py:attribute:: recipe.__dict__
      :module: arbor
      :value: mappingproxy({'__init__': <instancemethod __init__>, '__dict__': <attribute '__dict__' of 'arbor.recipe' objects>, '__doc__': 'A description of a model, describing the cells and the network via a cell-centric interface.', '__module__': 'arbor', 'num_cells': <instancemethod num_cells>, 'cell_description': <instancemethod cell_description>, 'cell_kind': <instancemethod cell_kind>, 'num_sources': <instancemethod num_sources>, 'num_targets': <instancemethod num_targets>, 'num_gap_junction_sites': <instancemethod num_gap_junction_sites>, 'event_generators': <instancemethod event_generators>, 'connections_on': <instancemethod connections_on>, 'gap_junctions_on': <instancemethod gap_junctions_on>, 'get_probes': <instancemethod get_probes>, '__str__': <instancemethod __str__>, '__repr__': <instancemethod __repr__>})
   
   
   .. py:method:: recipe.__init__(self: arbor.recipe) -> None
      :module: arbor
   
   
   .. py:attribute:: recipe.__module__
      :module: arbor
      :value: 'arbor'
   
   
   .. py:method:: recipe.__new__(**kwargs)
      :module: arbor
   
      Create and return a new object.  See help(type) for accurate signature.
      
   
   .. py:method:: recipe.__repr__(self: arbor.recipe) -> str
      :module: arbor
   
   
   .. py:method:: recipe.__str__(self: arbor.recipe) -> str
      :module: arbor
   
   
   .. py:method:: recipe.cell_description(self: arbor.recipe, gid: int) -> object
      :module: arbor
   
      High level description of the cell with global identifier gid.
      
   
   .. py:method:: recipe.cell_kind(self: arbor.recipe, gid: int) -> arbor.cell_kind
      :module: arbor
   
      The kind of cell with global identifier gid.
      
   
   .. py:method:: recipe.connections_on(self: arbor.recipe, gid: int) -> List[arbor.connection]
      :module: arbor
   
      A list of all the incoming connections to gid, [] by default.
      
   
   .. py:method:: recipe.event_generators(self: arbor.recipe, gid: int) -> List[object]
      :module: arbor
   
      A list of all the event generators that are attached to gid, [] by default.
      
   
   .. py:method:: recipe.gap_junctions_on(self: arbor.recipe, gid: int) -> List[arbor.gap_junction_connection]
      :module: arbor
   
      A list of the gap junctions connected to gid, [] by default.
      
   
   .. py:method:: recipe.get_probes(self: arbor.recipe, gid: int) -> List[arb::probe_info]
      :module: arbor
   
      The probes to allow monitoring.
      
   
   .. py:method:: recipe.num_cells(self: arbor.recipe) -> int
      :module: arbor
   
      The number of cells in the model.
      
   
   .. py:method:: recipe.num_gap_junction_sites(self: arbor.recipe, gid: int) -> int
      :module: arbor
   
      The number of gap junction sites on gid, 0 by default.
      
   
   .. py:method:: recipe.num_sources(self: arbor.recipe, gid: int) -> int
      :module: arbor
   
      The number of spike sources on gid, 0 by default.
      
   
   .. py:method:: recipe.num_targets(self: arbor.recipe, gid: int) -> int
      :module: arbor
   
      The number of post-synaptic sites on gid, 0 by default.
      

.. py:class:: regular_schedule
   :module: arbor

   Describes a regular schedule with multiples of dt within the interval [tstart, tstop).
   
   
   .. py:method:: regular_schedule.__init__(self: arbor.regular_schedule, tstart: object = None, dt: float = 0.0, tstop: object = None) -> None
      :module: arbor
   
      Construct a regular schedule with arguments:
        tstart: The delivery time of the first event in the sequence [ms], None by default.
        dt:     The interval between time points [ms], 0 by default.
        tstop:  No events delivered after this time [ms], None by default.
      
   
   .. py:attribute:: regular_schedule.__module__
      :module: arbor
      :value: 'arbor'
   
   
   .. py:method:: regular_schedule.__new__(**kwargs)
      :module: arbor
   
      Create and return a new object.  See help(type) for accurate signature.
      
   
   .. py:method:: regular_schedule.__repr__(self: arbor.regular_schedule) -> str
      :module: arbor
   
   
   .. py:method:: regular_schedule.__str__(self: arbor.regular_schedule) -> str
      :module: arbor
   
   
   .. py:method:: regular_schedule.dt
      :module: arbor
      :property:
   
      The interval between time points [ms].
      
   
   .. py:method:: regular_schedule.events(self: arbor.regular_schedule, arg0: float, arg1: float) -> List[float]
      :module: arbor
   
      A view of monotonically increasing time values in the half-open interval [t0, t1).
      
   
   .. py:method:: regular_schedule.tstart
      :module: arbor
      :property:
   
      The delivery time of the first event in the sequence [ms].
      
   
   .. py:method:: regular_schedule.tstop
      :module: arbor
      :property:
   
      No events delivered after this time [ms].
      

.. py:class:: sampler
   :module: arbor

   
   .. py:method:: sampler.__init__(self: arbor.sampler) -> None
      :module: arbor
   
   
   .. py:attribute:: sampler.__module__
      :module: arbor
      :value: 'arbor'
   
   
   .. py:method:: sampler.__new__(**kwargs)
      :module: arbor
   
      Create and return a new object.  See help(type) for accurate signature.
      
   
   .. py:method:: sampler.clear(self: arbor.sampler) -> None
      :module: arbor
   
      Clear all recorded samples.
      
   
   .. py:method:: sampler.samples(self: arbor.sampler, probe_id: arbor.cell_member) -> List[arbor.trace_sample]
      :module: arbor
   
      A list of the recorded samples of a probe with probe id.
      

.. py:class:: segment_tree
   :module: arbor

   
   .. py:method:: segment_tree.__init__(self: arbor.segment_tree) -> None
      :module: arbor
   
   
   .. py:attribute:: segment_tree.__module__
      :module: arbor
      :value: 'arbor'
   
   
   .. py:method:: segment_tree.__new__(**kwargs)
      :module: arbor
   
      Create and return a new object.  See help(type) for accurate signature.
      
   
   .. py:method:: segment_tree.__str__(self: arbor.segment_tree) -> str
      :module: arbor
   
   
   .. py:method:: segment_tree.append(*args, **kwargs)
      :module: arbor
   
      Overloaded function.
      
      1. append(self: arbor.segment_tree, parent: int, prox: arbor.mpoint, dist: arbor.mpoint, tag: int) -> int
      
      Append a segment to the tree.
      
      2. append(self: arbor.segment_tree, parent: int, dist: arbor.mpoint, tag: int) -> int
      
      Append a segment to the tree.
      
      3. append(self: arbor.segment_tree, parent: int, x: float, y: float, z: float, radius: float, tag: int) -> int
      
      Append a segment to the tree, using the distal location of the parent segment as the proximal end.
      
   
   .. py:method:: segment_tree.empty
      :module: arbor
      :property:
   
      Indicates whether the tree is empty (i.e. whether it has size 0)
      
   
   .. py:method:: segment_tree.parents
      :module: arbor
      :property:
   
      A list with the parent index of each segment.
      
   
   .. py:method:: segment_tree.reserve(self: arbor.segment_tree, arg0: int) -> None
      :module: arbor
   
   
   .. py:method:: segment_tree.segments
      :module: arbor
      :property:
   
      A list of the segments.
      
   
   .. py:method:: segment_tree.size
      :module: arbor
      :property:
   
      The number of segments in the tree.
      

.. py:class:: simulation
   :module: arbor

   The executable form of a model.
   A simulation is constructed from a recipe, and then used to update and monitor model state.
   
   
   .. py:method:: simulation.__init__(self: arbor.simulation, recipe: arbor.recipe, domain_decomposition: arbor.domain_decomposition, context: arbor.context) -> None
      :module: arbor
   
      Initialize the model described by a recipe, with cells and network distributed
      according to the domain decomposition and computational resources described by a context.
      
   
   .. py:attribute:: simulation.__module__
      :module: arbor
      :value: 'arbor'
   
   
   .. py:method:: simulation.__new__(**kwargs)
      :module: arbor
   
      Create and return a new object.  See help(type) for accurate signature.
      
   
   .. py:method:: simulation.__repr__(self: arbor.simulation) -> str
      :module: arbor
   
   
   .. py:method:: simulation.__str__(self: arbor.simulation) -> str
      :module: arbor
   
   
   .. py:method:: simulation.reset(self: arbor.simulation) -> None
      :module: arbor
   
      Reset the state of the simulation to its initial state.
      
   
   .. py:method:: simulation.run(self: arbor.simulation, tfinal: float, dt: float = 0.025) -> float
      :module: arbor
   
      Run the simulation from current simulation time to tfinal [ms], with maximum time step size dt [ms].
      
   
   .. py:method:: simulation.set_binning_policy(self: arbor.simulation, policy: arbor.binning, bin_interval: float) -> None
      :module: arbor
   
      Set the binning policy for event delivery, and the binning time interval if applicable [ms].
      

.. py:class:: single_cell_model
   :module: arbor

   Wrapper for simplified description, and execution, of single cell models.
   
   
   .. py:method:: single_cell_model.__init__(self: arbor.single_cell_model, cell: arbor.cable_cell) -> None
      :module: arbor
   
      Initialise a single cell model for a cable cell.
      
   
   .. py:attribute:: single_cell_model.__module__
      :module: arbor
      :value: 'arbor'
   
   
   .. py:method:: single_cell_model.__new__(**kwargs)
      :module: arbor
   
      Create and return a new object.  See help(type) for accurate signature.
      
   
   .. py:method:: single_cell_model.__repr__(self: arbor.single_cell_model) -> str
      :module: arbor
   
   
   .. py:method:: single_cell_model.__str__(self: arbor.single_cell_model) -> str
      :module: arbor
   
   
   .. py:method:: single_cell_model.probe(*args, **kwargs)
      :module: arbor
   
      Overloaded function.
      
      1. probe(self: arbor.single_cell_model, what: str, where: str, frequency: float) -> None
      
      Sample a variable on the cell.
       what:      Name of the variable to record (currently only 'voltage').
       where:     Location on cell morphology at which to sample the variable.
       frequency: The target frequency at which to sample [Hz].
      
      2. probe(self: arbor.single_cell_model, what: str, where: arbor.location, frequency: float) -> None
      
      Sample a variable on the cell.
       what:      Name of the variable to record (currently only 'voltage').
       where:     Location on cell morphology at which to sample the variable.
       frequency: The target frequency at which to sample [Hz].
      
   
   .. py:method:: single_cell_model.properties
      :module: arbor
      :property:
   
      Global properties.
      
   
   .. py:method:: single_cell_model.run(self: arbor.single_cell_model, tfinal: float, dt: float = 0.025) -> None
      :module: arbor
   
      Run model from t=0 to t=tfinal ms.
      
   
   .. py:method:: single_cell_model.spikes
      :module: arbor
      :property:
   
      Holds spike times [ms] after a call to run().
      
   
   .. py:method:: single_cell_model.traces
      :module: arbor
      :property:
   
      Holds sample traces after a call to run().
      

.. py:class:: spike
   :module: arbor

   
   .. py:method:: spike.__init__(self: arbor.spike) -> None
      :module: arbor
   
   
   .. py:attribute:: spike.__module__
      :module: arbor
      :value: 'arbor'
   
   
   .. py:method:: spike.__new__(**kwargs)
      :module: arbor
   
      Create and return a new object.  See help(type) for accurate signature.
      
   
   .. py:method:: spike.__repr__(self: arbor.spike) -> str
      :module: arbor
   
   
   .. py:method:: spike.__str__(self: arbor.spike) -> str
      :module: arbor
   
   
   .. py:method:: spike.source
      :module: arbor
      :property:
   
      The spike source (type: cell_member).
      
   
   .. py:method:: spike.time
      :module: arbor
      :property:
   
      The spike time [ms].
      

.. py:class:: spike_detector
   :module: arbor

   A spike detector, generates a spike when voltage crosses a threshold.
   
   
   .. py:method:: spike_detector.__init__(self: arbor.spike_detector, threshold: float) -> None
      :module: arbor
   
   
   .. py:attribute:: spike_detector.__module__
      :module: arbor
      :value: 'arbor'
   
   
   .. py:method:: spike_detector.__new__(**kwargs)
      :module: arbor
   
      Create and return a new object.  See help(type) for accurate signature.
      
   
   .. py:method:: spike_detector.__repr__(self: arbor.spike_detector) -> str
      :module: arbor
   
   
   .. py:method:: spike_detector.__str__(self: arbor.spike_detector) -> str
      :module: arbor
   
   
   .. py:method:: spike_detector.threshold
      :module: arbor
      :property:
   
      Voltage threshold of spike detector [ms]
      

.. py:class:: spike_recorder
   :module: arbor

   
   .. py:method:: spike_recorder.__init__(self: arbor.spike_recorder) -> None
      :module: arbor
   
   
   .. py:attribute:: spike_recorder.__module__
      :module: arbor
      :value: 'arbor'
   
   
   .. py:method:: spike_recorder.__new__(**kwargs)
      :module: arbor
   
      Create and return a new object.  See help(type) for accurate signature.
      
   
   .. py:method:: spike_recorder.spikes
      :module: arbor
      :property:
   
      A list of the recorded spikes.
      

.. py:class:: spike_source_cell
   :module: arbor

   A spike source cell, that generates a user-defined sequence of spikes that act as inputs for other cells in the network.
   
   
   .. py:method:: spike_source_cell.__init__(*args, **kwargs)
      :module: arbor
   
      Overloaded function.
      
      1. __init__(self: arbor.spike_source_cell, schedule: pyarb::regular_schedule_shim) -> None
      
      Construct a spike source cell that generates spikes at regular intervals.
      
      2. __init__(self: arbor.spike_source_cell, schedule: pyarb::explicit_schedule_shim) -> None
      
      Construct a spike source cell that generates spikes at a sequence of user-defined times.
      
      3. __init__(self: arbor.spike_source_cell, schedule: pyarb::poisson_schedule_shim) -> None
      
      Construct a spike source cell that generates spikes at times defined by a Poisson sequence.
      
   
   .. py:attribute:: spike_source_cell.__module__
      :module: arbor
      :value: 'arbor'
   
   
   .. py:method:: spike_source_cell.__new__(**kwargs)
      :module: arbor
   
      Create and return a new object.  See help(type) for accurate signature.
      
   
   .. py:method:: spike_source_cell.__repr__(self: arbor.spike_source_cell) -> str
      :module: arbor
   
   
   .. py:method:: spike_source_cell.__str__(self: arbor.spike_source_cell) -> str
      :module: arbor
   

.. py:class:: trace
   :module: arbor

   Values and meta-data for a sample-trace on a single cell model.
   
   
   .. py:method:: trace.__init__(*args, **kwargs)
      :module: arbor
   
      Initialize self.  See help(type(self)) for accurate signature.
      
   
   .. py:attribute:: trace.__module__
      :module: arbor
      :value: 'arbor'
   
   
   .. py:method:: trace.__new__(**kwargs)
      :module: arbor
   
      Create and return a new object.  See help(type) for accurate signature.
      
   
   .. py:method:: trace.__repr__(self: arbor.trace) -> str
      :module: arbor
   
   
   .. py:method:: trace.__str__(self: arbor.trace) -> str
      :module: arbor
   
   
   .. py:method:: trace.location
      :module: arbor
      :property:
   
      Location on cell morphology.
      
   
   .. py:method:: trace.time
      :module: arbor
      :property:
   
      Time stamps of samples [ms].
      
   
   .. py:method:: trace.value
      :module: arbor
      :property:
   
      Sample values.
      
   
   .. py:method:: trace.variable
      :module: arbor
      :property:
   
      Name of the variable being recorded.
      

.. py:class:: trace_sample
   :module: arbor

   
   .. py:method:: trace_sample.__init__(*args, **kwargs)
      :module: arbor
   
      Initialize self.  See help(type(self)) for accurate signature.
      
   
   .. py:attribute:: trace_sample.__module__
      :module: arbor
      :value: 'arbor'
   
   
   .. py:method:: trace_sample.__new__(**kwargs)
      :module: arbor
   
      Create and return a new object.  See help(type) for accurate signature.
      
   
   .. py:method:: trace_sample.__repr__(self: arbor.trace_sample) -> str
      :module: arbor
   
   
   .. py:method:: trace_sample.__str__(self: arbor.trace_sample) -> str
      :module: arbor
   
   
   .. py:method:: trace_sample.time
      :module: arbor
      :property:
   
      The sample time [ms] at a specific probe.
      
   
   .. py:method:: trace_sample.value
      :module: arbor
      :property:
   
      The sample record at a specific probe.
      
