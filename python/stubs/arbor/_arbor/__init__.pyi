"""
arbor: multicompartment neural network models.
"""
from __future__ import annotations
import typing
from . import env
from . import units
__all__ = ['ArbFileNotFoundError', 'ArbValueError', 'MechCatItemIterator', 'MechCatKeyIterator', 'MechCatValueIterator', 'allen_catalogue', 'asc_morphology', 'axial_resistivity', 'backend', 'bbp_catalogue', 'benchmark_cell', 'cable', 'cable_cell', 'cable_component', 'cable_global_properties', 'cable_probe_axial_current', 'cable_probe_density_state', 'cable_probe_density_state_cell', 'cable_probe_ion_current_cell', 'cable_probe_ion_current_density', 'cable_probe_ion_diff_concentration', 'cable_probe_ion_diff_concentration_cell', 'cable_probe_ion_ext_concentration', 'cable_probe_ion_ext_concentration_cell', 'cable_probe_ion_int_concentration', 'cable_probe_ion_int_concentration_cell', 'cable_probe_membrane_voltage', 'cable_probe_membrane_voltage_cell', 'cable_probe_point_info', 'cable_probe_point_state', 'cable_probe_point_state_cell', 'cable_probe_stimulus_current_cell', 'cable_probe_total_current_cell', 'cable_probe_total_ion_current_cell', 'cable_probe_total_ion_current_density', 'catalogue', 'cell_address', 'cell_cv_data', 'cell_global_label', 'cell_kind', 'cell_local_label', 'cell_member', 'component_meta_data', 'config', 'connection', 'context', 'cv_data', 'cv_policy', 'cv_policy_every_segment', 'cv_policy_explicit', 'cv_policy_fixed_per_branch', 'cv_policy_max_extent', 'cv_policy_single', 'decor', 'default_catalogue', 'density', 'domain_decomposition', 'env', 'event_generator', 'explicit_schedule', 'ext_concentration', 'extent', 'gap_junction_connection', 'group_description', 'iclamp', 'int_concentration', 'intersect_region', 'ion_data', 'ion_dependency', 'ion_diffusivity', 'ion_settings', 'isometry', 'junction', 'label_dict', 'lif_cell', 'lif_probe_metadata', 'lif_probe_voltage', 'load_asc', 'load_catalogue', 'load_component', 'load_swc_arbor', 'load_swc_neuron', 'location', 'mechanism', 'mechanism_field', 'mechanism_info', 'membrane_capacitance', 'membrane_potential', 'meter_manager', 'meter_report', 'mnpos', 'morphology', 'morphology_provider', 'mpoint', 'msegment', 'neuroml', 'neuroml_morph_data', 'neuron_cable_properties', 'partition_by_group', 'partition_hint', 'partition_load_balance', 'place_pwlin', 'poisson_schedule', 'print_config', 'probe', 'proc_allocation', 'recipe', 'regular_schedule', 'reversal_potential', 'reversal_potential_method', 'scaled_mechanism', 'schedule_base', 'segment_tree', 'selection_policy', 'simulation', 'single_cell_model', 'spike', 'spike_recording', 'spike_source_cell', 'stochastic_catalogue', 'synapse', 'temperature', 'threshold_detector', 'trace', 'units', 'voltage_process', 'write_component']
class ArbFileNotFoundError(FileNotFoundError):
    pass
class ArbValueError(ValueError):
    pass
class MechCatItemIterator:
    def __iter__(self) -> MechCatItemIterator:
        ...
    def __next__(self) -> tuple[str, mechanism_info]:
        ...
class MechCatKeyIterator:
    def __iter__(self) -> MechCatKeyIterator:
        ...
    def __next__(self) -> str:
        ...
class MechCatValueIterator:
    def __iter__(self) -> MechCatValueIterator:
        ...
    def __next__(self) -> mechanism_info:
        ...
class asc_morphology:
    """
    The morphology and label dictionary meta-data loaded from a Neurolucida ASCII (.asc) file.
    """
    @property
    def labels(self) -> label_dict:
        """
        The four canonical regions are labeled 'soma', 'axon', 'dend' and 'apic'.
        """
    @property
    def morphology(self) -> morphology:
        """
        The cable cell morphology.
        """
    @property
    def segment_tree(self) -> segment_tree:
        """
        The raw segment tree.
        """
class axial_resistivity:
    """
    Setting the axial resistivity.
    """
    def __init__(self, arg0: units.quantity) -> None:
        ...
    def __repr__(self) -> str:
        ...
class backend:
    """
    Enumeration used to indicate which hardware backend to execute a cell group on.
    
    Members:
    
      gpu : Use GPU backend.
    
      multicore : Use multicore backend.
    """
    __members__: typing.ClassVar[dict[str, backend]]  # value = {'gpu': <backend.gpu: 1>, 'multicore': <backend.multicore: 0>}
    gpu: typing.ClassVar[backend]  # value = <backend.gpu: 1>
    multicore: typing.ClassVar[backend]  # value = <backend.multicore: 0>
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class benchmark_cell:
    """
    A benchmarking cell, used by Arbor developers to test communication performance.
    A benchmark cell generates spikes at a user-defined sequence of time points, and
    the time taken to integrate a cell can be tuned by setting the realtime_ratio,
    for example if realtime_ratio=2, a cell will take 2 seconds of CPU time to
    simulate 1 second.
    """
    @typing.overload
    def __init__(self, source_label: str, target_label: str, schedule: regular_schedule, realtime_ratio: float = 1.0) -> None:
        """
        Construct a benchmark cell that generates spikes on 'source_label' at regular intervals.
        The cell has one source labeled 'source_label', and one target labeled 'target_label'.
        """
    @typing.overload
    def __init__(self, source_label: str, target_label: str, schedule: explicit_schedule, realtime_ratio: float = 1.0) -> None:
        """
        Construct a benchmark cell that generates spikes on 'source_label' at a sequence of user-defined times.
        The cell has one source labeled 'source_label', and one target labeled 'target_label'.
        """
    @typing.overload
    def __init__(self, source_label: str, target_label: str, schedule: poisson_schedule, realtime_ratio: float = 1.0) -> None:
        """
        Construct a benchmark cell that generates spikeson 'source_label' at times defined by a Poisson sequence.
        The cell has one source labeled 'source_label', and one target labeled 'target_label'.
        """
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
class cable:
    __hash__: typing.ClassVar[None] = None
    def __eq__(self, arg0: cable) -> bool:
        ...
    def __init__(self, branch: int, prox: float, dist: float) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    @property
    def branch(self) -> int:
        """
        The id of the branch on which the cable lies.
        """
    @property
    def dist(self) -> float:
        """
        The relative position of the distal end of the cable on its branch ∈ [0,1].
        """
    @property
    def prox(self) -> float:
        """
        The relative position of the proximal end of the cable on its branch ∈ [0,1].
        """
class cable_cell:
    """
    Represents morphologically-detailed cell models, with morphology represented as a
    tree of one-dimensional cable segments.
    """
    @typing.overload
    def __init__(self, morphology: morphology, decor: decor, labels: label_dict | None = None) -> None:
        """
        Construct with a morphology, decor, and label dictionary.
        """
    @typing.overload
    def __init__(self, segment_tree: segment_tree, decor: decor, labels: label_dict | None = None) -> None:
        """
        Construct with a morphology derived from a segment tree, decor, and label dictionary.
        """
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    def cables(self, label: str) -> list[cable]:
        """
        The cable segments of the cell morphology for a region label.
        """
    def locations(self, label: str) -> list[location]:
        """
        The locations of the cell morphology for a locset label.
        """
    @property
    def num_branches(self) -> int:
        """
        The number of unbranched cable sections in the morphology.
        """
class cable_component:
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    @property
    def component(self) -> morphology | label_dict | decor | cable_cell:
        """
        cable-cell component.
        """
    @property
    def meta_data(self) -> component_meta_data:
        """
        cable-cell component meta-data.
        """
    @meta_data.setter
    def meta_data(self, arg0: component_meta_data) -> None:
        ...
class cable_global_properties:
    membrane_voltage_limit: float | None
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: cable_global_properties) -> None:
        ...
    def __str__(self) -> str:
        ...
    def check(self) -> None:
        """
        Test whether all default parameters and ion species properties have been set.
        """
    def set_ion(self, ion: str, valence: int | None = None, int_con: units.quantity | None = None, ext_con: units.quantity | None = None, rev_pot: units.quantity | None = None, method: typing.Any = None, diff: units.quantity | None = None) -> None:
        """
        Set the global default properties of ion species named 'ion'.
         * valence: valence of the ion species [e].
         * int_con: initial internal concentration [mM].
         * ext_con: initial external concentration [mM].
         * rev_pot: reversal potential [mV].
         * method:  mechanism for calculating reversal potential.
         * diff:   diffusivity [m^2/s].
        There are 3 ion species predefined in arbor: 'ca', 'na' and 'k'.
        If 'ion' in not one of these ions it will be added to the list, making it
        available to mechanisms. The user has to provide the valence of a previously
        undefined ion the first time this function is called with it as an argument.
        Species concentrations and reversal potential can be overridden on
        specific regions using the paint interface, while the method for calculating
        reversal potential is global for all compartments in the cell, and can't be
        overriden locally.
        """
    def set_property(self, Vm: units.quantity | None = None, cm: units.quantity | None = None, rL: units.quantity | None = None, tempK: units.quantity | None = None) -> None:
        """
        Set global default values for cable and cell properties.
         * Vm:    initial membrane voltage [mV].
         * cm:    membrane capacitance [F/m²].
         * rL:    axial resistivity [Ω·cm].
         * tempK: temperature [Kelvin].
        These values can be overridden on specific regions using the paint interface.
        """
    def unset_ion(self, arg0: str) -> None:
        """
        Remove ion species from properties.
        """
    @property
    def axial_resistivity(self) -> float | None:
        ...
    @axial_resistivity.setter
    def axial_resistivity(self, arg1: float) -> None:
        ...
    @property
    def catalogue(self) -> catalogue:
        """
        The mechanism catalogue.
        """
    @catalogue.setter
    def catalogue(self, arg0: catalogue) -> None:
        ...
    @property
    def coalesce_synapses(self) -> bool:
        """
        Flag for enabling/disabling linear syanpse coalescing.
        """
    @coalesce_synapses.setter
    def coalesce_synapses(self, arg0: bool) -> None:
        ...
    @property
    def ion_data(self) -> dict[str, ion_data]:
        ...
    @property
    def ion_reversal_potential(self) -> dict[str, mechanism]:
        ...
    @property
    def ion_valence(self) -> dict[str, int]:
        ...
    @property
    def ions(self) -> dict[str, ion_settings]:
        """
        Return a view of all ion settings.
        """
    @property
    def membrane_capacitance(self) -> float | None:
        ...
    @membrane_capacitance.setter
    def membrane_capacitance(self, arg1: float) -> None:
        ...
    @property
    def membrane_potential(self) -> float | None:
        ...
    @membrane_potential.setter
    def membrane_potential(self, arg1: float) -> None:
        ...
    @property
    def temperature(self) -> float | None:
        ...
    @temperature.setter
    def temperature(self, arg1: float) -> None:
        ...
class cable_probe_point_info:
    """
    Probe metadata associated with a cable cell probe for point process state.
    """
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    @property
    def location(self) -> location:
        """
        Location of point process instance on cell.
        """
    @location.setter
    def location(self, arg0: location) -> None:
        ...
    @property
    def multiplicity(self) -> int:
        """
        Number of coalesced point processes (linear synapses) associated with this instance.
        """
    @multiplicity.setter
    def multiplicity(self, arg0: int) -> None:
        ...
    @property
    def target(self) -> int:
        """
        The target index of the point process instance on the cell.
        """
    @target.setter
    def target(self, arg0: int) -> None:
        ...
class catalogue:
    def __contains__(self, name: str) -> bool:
        """
        Is 'name' in the catalogue?
        """
    def __getitem__(self, arg0: str) -> mechanism_info:
        ...
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: catalogue) -> None:
        ...
    def __iter__(self) -> MechCatKeyIterator:
        """
        Return an iterator over all mechanism names in this catalogues.
        """
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    def derive(self, name: str, parent: str, globals: dict[str, float] = {}, ions: dict[str, str] = {}) -> None:
        ...
    def extend(self, other: catalogue, prefix: str) -> None:
        """
        Import another catalogue, possibly with a prefix. Will overwrite in case of name collisions.
        """
    def is_derived(self, name: str) -> bool:
        """
        Is 'name' a derived mechanism or can it be implicitly derived?
        """
    def items(self) -> MechCatItemIterator:
        """
        Return an iterator over all (name, mechanism) tuples  in this catalogues.
        """
    def keys(self) -> MechCatKeyIterator:
        """
        Return an iterator over all mechanism names in this catalogues.
        """
    def values(self) -> MechCatValueIterator:
        """
        Return an iterator over all mechanism info values in this catalogues.
        """
class cell_address:
    gid: int
    tag: str
class cell_cv_data:
    """
    Provides information on the CVs representing the discretization of a cable-cell.
    """
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    def cables(self, index: int) -> list[cable]:
        """
        Return a list of cables representing the CV at the given index.
        """
    def children(self, index: int) -> list[int]:
        """
        Return a list of indices of the CVs representing the children of the CV at the given index.
        """
    def parent(self, index: int) -> int:
        """
        Return the index of the CV representing the parent of the CV at the given index.
        """
    @property
    def num_cv(self) -> int:
        """
        Return the number of CVs in the cell.
        """
class cell_global_label:
    """
    For global identification of an item.
    
    cell_global_label members:
    (1) a unique cell identified by its gid.
    (2) a cell_local_label, referring to a labeled group of items on the cell and a policy for selecting a single item out of the group.
    """
    @typing.overload
    def __init__(self, gid: int, label: str) -> None:
        """
        Construct a cell_global_label identifier from a gid and a label argument identifying an item on the cell.
        The default round_robin policy is used for selecting one of possibly multiple items on the cell associated with the label.
        """
    @typing.overload
    def __init__(self, gid: int, label: cell_local_label) -> None:
        """
        Construct a cell_global_label identifier with arguments:
          gid:   The global identifier of the cell.
          label: The cell_local_label representing the label and selection policy of an item on the cell.
        """
    @typing.overload
    def __init__(self, arg0: tuple) -> None:
        """
        Construct a cell_global_label identifier with tuple argument (gid, label):
          gid:   The global identifier of the cell.
          label: The cell_local_label representing the label and selection policy of an item on the cell.
        """
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    @property
    def gid(self) -> int:
        """
        The global identifier of the cell.
        """
    @gid.setter
    def gid(self, arg0: int) -> None:
        ...
    @property
    def label(self) -> cell_local_label:
        """
        The cell_local_label representing the label and selection policy of an item on the cell.
        """
    @label.setter
    def label(self, arg0: cell_local_label) -> None:
        ...
class cell_kind:
    """
    Enumeration used to identify the cell kind, used by the model to group equal kinds in the same cell group.
    
    Members:
    
      benchmark : Proxy cell used for benchmarking.
    
      cable : A cell with morphology described by branching 1D cable segments.
    
      lif : Leaky-integrate and fire neuron.
    
      spike_source : Proxy cell that generates spikes from a spike sequence provided by the user.
    """
    __members__: typing.ClassVar[dict[str, cell_kind]]  # value = {'benchmark': <cell_kind.benchmark: 3>, 'cable': <cell_kind.cable: 0>, 'lif': <cell_kind.lif: 1>, 'spike_source': <cell_kind.spike_source: 2>}
    benchmark: typing.ClassVar[cell_kind]  # value = <cell_kind.benchmark: 3>
    cable: typing.ClassVar[cell_kind]  # value = <cell_kind.cable: 0>
    lif: typing.ClassVar[cell_kind]  # value = <cell_kind.lif: 1>
    spike_source: typing.ClassVar[cell_kind]  # value = <cell_kind.spike_source: 2>
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class cell_local_label:
    """
    For local identification of an item.
    
    cell_local_label identifies:
    (1) a labeled group of one or more items on one or more locations on the cell.
    (2) a policy for selecting one of the items.
    """
    @typing.overload
    def __init__(self, label: str) -> None:
        """
        Construct a cell_local_label identifier from a label argument identifying a group of one or more items on a cell.
        The default round_robin policy is used for selecting one of possibly multiple items associated with the label.
        """
    @typing.overload
    def __init__(self, label: str, policy: selection_policy) -> None:
        """
        Construct a cell_local_label identifier with arguments:
          label:  The identifier of a group of one or more items on a cell.
          policy: The policy for selecting one of possibly multiple items associated with the label.
        """
    @typing.overload
    def __init__(self, arg0: tuple) -> None:
        """
        Construct a cell_local_label identifier with tuple argument (label, policy):
          label:  The identifier of a group of one or more items on a cell.
          policy: The policy for selecting one of possibly multiple items associated with the label.
        """
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    @property
    def label(self) -> str:
        """
        The identifier of a a group of one or more items on a cell.
        """
    @label.setter
    def label(self, arg0: str) -> None:
        ...
    @property
    def policy(self) -> selection_policy:
        """
        The policy for selecting one of possibly multiple items associated with the label.
        """
    @policy.setter
    def policy(self, arg0: selection_policy) -> None:
        ...
class cell_member:
    """
    For global identification of a cell-local item.
    
    Items of cell_member must:
      (1) be associated with a unique cell, identified by the member gid;
      (2) identify an item within a cell-local collection by the member index.
    """
    @typing.overload
    def __init__(self, gid: int, index: int) -> None:
        """
        Construct a cell member identifier with arguments:
          gid:     The global identifier of the cell.
          index:   The cell-local index of the item.
        """
    @typing.overload
    def __init__(self, arg0: tuple) -> None:
        """
        Construct a cell member identifier with tuple argument (gid, index):
          gid:     The global identifier of the cell.
          index:   The cell-local index of the item.
        """
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    @property
    def gid(self) -> int:
        """
        The global identifier of the cell.
        """
    @gid.setter
    def gid(self, arg0: int) -> None:
        ...
    @property
    def index(self) -> int:
        """
        Cell-local index of the item.
        """
    @index.setter
    def index(self, arg0: int) -> None:
        ...
class component_meta_data:
    @property
    def version(self) -> str:
        """
        cable-cell component version.
        """
    @version.setter
    def version(self, arg0: str) -> None:
        ...
class connection:
    """
    Describes a connection between two cells:
      Defined by source and destination end points (that is pre-synaptic and post-synaptic respectively), a connection weight and a delay time.
    """
    def __init__(self, source: cell_global_label, dest: cell_local_label, weight: float, delay: units.quantity) -> None:
        """
        Construct a connection with arguments:
          source:      The source end point of the connection.
          dest:        The destination end point of the connection.
          weight:      The weight delivered to the target synapse (unit defined by the type of synapse target).
          delay:       The delay of the connection [ms].
        """
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    @property
    def delay(self) -> float:
        """
        The delay time of the connection [ms].
        """
    @delay.setter
    def delay(self, arg0: float) -> None:
        ...
    @property
    def dest(self) -> cell_local_label:
        """
        The destination label of the connection.
        """
    @dest.setter
    def dest(self, arg0: cell_local_label) -> None:
        ...
    @property
    def source(self) -> cell_global_label:
        """
        The source gid and label of the connection.
        """
    @source.setter
    def source(self, arg0: cell_global_label) -> None:
        ...
    @property
    def weight(self) -> float:
        """
        The weight of the connection.
        """
    @weight.setter
    def weight(self, arg0: float) -> None:
        ...
class context:
    """
    An opaque handle for the hardware resources used in a simulation.
    """
    @typing.overload
    def __init__(self) -> None:
        """
        Construct a local context with proc_allocation = env.default_allocation().
        """
    @typing.overload
    def __init__(self, *, threads: int = 1, gpu_id: typing.Any = None, mpi: typing.Any = None, inter: typing.Any = None, bind_procs: bool = False, bind_threads: bool = False) -> None:
        """
        Construct a context with arguments:
          threads: The number of threads available locally for execution. Must be set to 1 at minimum. 1 by default.
          gpu_id:  The identifier of the GPU to use, None by default. Only available if arbor.__config__['gpu']!="none".
          mpi:     The MPI communicator, None by default. Only available if arbor.__config__['mpi']==True.
          inter:   An MPI intercommunicator used to connect to external simulations, None by default. Only available if arbor.__config__['mpi']==True.
          bind_procs:   Create process binding mask.
          bind_threads: Create thread binding mask.
        """
    @typing.overload
    def __init__(self, alloc: proc_allocation, *, mpi: typing.Any = None, inter: typing.Any = None) -> None:
        """
        Construct a context with arguments:
          alloc:   The computational resources to be used for the simulation.
          mpi:     The MPI communicator, None by default. Only available if arbor.__config__['mpi']==True.
          inter:   An MPI intercommunicator used to connect to external simulations, None by default. Only available if arbor.__config__['mpi']==True.
        """
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    @property
    def has_gpu(self) -> bool:
        """
        Whether the context has a GPU.
        """
    @property
    def has_mpi(self) -> bool:
        """
        Whether the context uses MPI for distributed communication.
        """
    @property
    def rank(self) -> int:
        """
        The numeric id of the local domain (equivalent to MPI rank).
        """
    @property
    def ranks(self) -> int:
        """
        The number of distributed domains (equivalent to the number of MPI ranks).
        """
    @property
    def threads(self) -> int:
        """
        The number of threads in the context's thread pool.
        """
class cv_policy:
    """
    Describes the rules used to discretize (compartmentalise) a cable cell morphology.
    """
    def __add__(self, arg0: cv_policy) -> cv_policy:
        ...
    def __init__(self, expression: str) -> None:
        """
        A valid CV policy expression
        """
    def __or__(self, arg0: cv_policy) -> cv_policy:
        ...
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    @property
    def domain(self) -> str:
        """
        The domain on which the policy is applied.
        """
class decor:
    """
    Description of the decorations to be applied to a cable cell, that is the painted,
    placed and defaulted properties, mecahanisms, ion species etc.
    """
    @typing.overload
    def __init__(self) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: decor) -> None:
        ...
    def defaults(self) -> list[membrane_potential | axial_resistivity | temperature | membrane_capacitance | ion_diffusivity | int_concentration | ext_concentration | reversal_potential | reversal_potential_method | cv_policy]:
        """
        Return a view of all defaults.
        """
    @typing.overload
    def discretization(self, policy: cv_policy) -> decor:
        """
        A cv_policy used to discretise the cell into compartments for simulation
        """
    @typing.overload
    def discretization(self, policy: str) -> decor:
        """
        An s-expression string representing a cv_policy used to discretise the cell into compartments for simulation
        """
    @typing.overload
    def paint(self, region: str, mechanism: density) -> decor:
        """
        Associate a density mechanism with a region.
        """
    @typing.overload
    def paint(self, region: str, mechanism: voltage_process) -> decor:
        """
        Associate a voltage process mechanism with a region.
        """
    @typing.overload
    def paint(self, region: str, mechanism: scaled_mechanism) -> None:
        """
        Associate a scaled density mechanism with a region.
        """
    @typing.overload
    def paint(self, region: str, Vm: units.quantity | str | None = None, cm: units.quantity | str | None = None, rL: units.quantity | str | None = None, tempK: units.quantity | str | None = None) -> decor:
        """
        Set cable properties on a region.
        Set global default values for cable and cell properties.
         * Vm:    initial membrane voltage [mV].
         * cm:    membrane capacitance [F/m²].
         * rL:    axial resistivity [Ω·cm].
         * tempK: temperature [Kelvin].
        """
    @typing.overload
    def paint(self, region: str, *, ion: str, int_con: units.quantity | None = None, ext_con: units.quantity | None = None, rev_pot: units.quantity | None = None, diff: units.quantity | None = None) -> decor:
        """
        Set ion species properties conditions on a region.
         * int_con: initial internal concentration [mM].
         * ext_con: initial external concentration [mM].
         * rev_pot: reversal potential [mV].
         * method:  mechanism for calculating reversal potential.
         * diff:   diffusivity [m^2/s].
        """
    def paintings(self) -> list[tuple[str, membrane_potential | axial_resistivity | temperature | membrane_capacitance | ion_diffusivity | int_concentration | ext_concentration | reversal_potential | density | voltage_process | scaled_mechanism]]:
        """
        Return a view of all painted items.
        """
    @typing.overload
    def place(self, locations: str, synapse: synapse, label: str) -> decor:
        """
        Place one instance of 'synapse' on each location in 'locations'.The group of synapses has the label 'label', used for forming connections between cells.
        """
    @typing.overload
    def place(self, locations: str, junction: junction, label: str) -> decor:
        """
        Place one instance of 'junction' on each location in 'locations'.The group of junctions has the label 'label', used for forming gap-junction connections between cells.
        """
    @typing.overload
    def place(self, locations: str, iclamp: iclamp, label: str) -> decor:
        """
        Add a current stimulus at each location in locations.The group of current stimuli has the label 'label'.
        """
    @typing.overload
    def place(self, locations: str, detector: threshold_detector, label: str) -> decor:
        """
        Add a voltage spike detector at each location in locations.The group of spike detectors has the label 'label', used for forming connections between cells.
        """
    def placements(self) -> list[tuple[str, iclamp | threshold_detector | synapse | junction, str]]:
        """
        Return a view of all placed items.
        """
    def set_ion(self, ion: str, int_con: units.quantity | None = None, ext_con: units.quantity | None = None, rev_pot: units.quantity | None = None, method: typing.Any = None, diff: units.quantity | None = None) -> decor:
        """
        Set the cell-level properties of ion species named 'ion'.
         * int_con: initial internal concentration [mM].
         * ext_con: initial external concentration [mM].
         * rev_pot: reversal potential [mV].
         * method:  mechanism for calculating reversal potential.
         * diff:    diffusivity [m^2/s].
        There are 3 ion species predefined in arbor: 'ca', 'na' and 'k'.
        If 'ion' in not one of these ions it will be added to the list, making it
        available to mechanisms. The user has to provide the valence of a previously
        undefined ion the first time this function is called with it as an argument.
        Species concentrations and reversal potential can be overridden on
        specific regions using the paint interface, while the method for calculating
        reversal potential is global for all compartments in the cell, and can't be
        overriden locally.
        """
    def set_property(self, Vm: units.quantity | None = None, cm: units.quantity | None = None, rL: units.quantity | None = None, tempK: units.quantity | None = None) -> decor:
        """
        Set default values for cable and cell properties:
         * Vm:    initial membrane voltage [mV].
         * cm:    membrane capacitance [F/m²].
         * rL:    axial resistivity [Ω·cm].
         * tempK: temperature [Kelvin].
        These values can be overridden on specific regions using the paint interface.
        """
class density:
    """
    For painting a density mechanism on a region.
    """
    @typing.overload
    def __init__(self, arg0: str) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: mechanism) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: str, arg1: dict[str, float]) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: mechanism, arg1: dict[str, float]) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: str, **kwargs) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: mechanism, **kwargs) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    @property
    def mech(self) -> mechanism:
        """
        The underlying mechanism.
        """
class domain_decomposition:
    """
    The domain decomposition is responsible for describing the distribution of cells across cell groups and domains.
    """
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    def gid_domain(self, gid: int) -> int:
        """
        Query the domain id that a cell assigned to (using global identifier gid).
        """
    @property
    def domain_id(self) -> int:
        """
        The index of the local domain.
        Always 0 for non-distributed models, and corresponds to the MPI rank for distributed runs.
        """
    @property
    def groups(self) -> list[group_description]:
        """
        Descriptions of the cell groups on the local domain.
        """
    @property
    def num_domains(self) -> int:
        """
        Number of domains that the model is distributed over.
        """
    @property
    def num_global_cells(self) -> int:
        """
        Total number of cells in the global model (sum of num_local_cells over all domains).
        """
    @property
    def num_groups(self) -> int:
        """
        Total number of cell groups in the local domain.
        """
    @property
    def num_local_cells(self) -> int:
        """
        Total number of cells in the local domain.
        """
class event_generator:
    def __init__(self, target: cell_local_label, weight: float, sched: schedule_base) -> None:
        """
        Construct an event generator with arguments:
          target: The target synapse label and selection policy.
          weight: The weight of events to deliver.
          sched:  A schedule of the events.
        """
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    @property
    def target(self) -> cell_local_label:
        """
        The target synapse (gid, local_id).
        """
    @target.setter
    def target(self, arg0: cell_local_label) -> None:
        ...
    @property
    def weight(self) -> float:
        """
        The weight of events to deliver.
        """
    @weight.setter
    def weight(self, arg0: float) -> None:
        ...
class explicit_schedule(schedule_base):
    """
    Describes an explicit schedule at a predetermined (sorted) sequence of times.
    """
    @typing.overload
    def __init__(self) -> None:
        """
        Construct an empty explicit schedule.
        """
    @typing.overload
    def __init__(self, times: list[units.quantity]) -> None:
        """
        Construct an explicit schedule with argument:
          times: A list of times [ms], [] by default.
        """
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    def events(self, arg0: float, arg1: float) -> list[float]:
        """
        A view of monotonically increasing time values in the half-open interval [t0, t1) in [ms].
        """
    @property
    def times_ms(self) -> list[float]:
        """
        A list of times [ms].
        """
    @times_ms.setter
    def times_ms(self, arg1: list[float]) -> None:
        ...
class ext_concentration:
    """
    Setting the initial external ion concentration.
    """
    def __init__(self, arg0: str, arg1: units.quantity) -> None:
        ...
    def __repr__(self) -> str:
        ...
class extent:
    """
    A potentially empty region on a morphology.
    """
class gap_junction_connection:
    """
    Describes a gap junction between two gap junction sites.
    """
    def __init__(self, peer: cell_global_label, local: cell_local_label, weight: float) -> None:
        """
        Construct a gap junction connection with arguments:
          peer:  remote half of the gap junction connection.
          local: local half of the gap junction connection.
          weight:  Gap junction connection weight [unit-less].
        """
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    @property
    def local(self) -> cell_local_label:
        """
        Local label of the gap junction connection.
        """
    @local.setter
    def local(self, arg0: cell_local_label) -> None:
        ...
    @property
    def peer(self) -> cell_global_label:
        """
        Remote gid and label of the gap junction connection.
        """
    @peer.setter
    def peer(self, arg0: cell_global_label) -> None:
        ...
    @property
    def weight(self) -> float:
        """
        Gap junction connection weight [unit-less].
        """
    @weight.setter
    def weight(self, arg0: float) -> None:
        ...
class group_description:
    """
    The indexes of a set of cells of the same kind that are grouped together in a cell group.
    """
    def __init__(self, kind: cell_kind, gids: list[int], backend: backend) -> None:
        """
        Construct a group description with cell kind, list of gids, and backend kind.
        """
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    @property
    def backend(self) -> backend:
        """
        The hardware backend on which the cell group will run.
        """
    @property
    def gids(self) -> list[int]:
        """
        The list of gids of the cells in the group.
        """
    @property
    def kind(self) -> cell_kind:
        """
        The type of cell in the cell group.
        """
class iclamp:
    """
    A current clamp for injecting a DC or fixed frequency current governed by a piecewise linear envelope.
    """
    @typing.overload
    def __init__(self, tstart: units.quantity, duration: units.quantity, current: units.quantity, *, frequency: units.quantity = ..., phase: units.quantity = ...) -> None:
        """
        Construct finite duration current clamp, constant amplitude
        """
    @typing.overload
    def __init__(self, current: units.quantity, *, frequency: units.quantity = ..., phase: units.quantity = ...) -> None:
        """
        Construct constant amplitude current clamp
        """
    @typing.overload
    def __init__(self, envelope: list[tuple[units.quantity, units.quantity]], *, frequency: units.quantity = ..., phase: units.quantity = ...) -> None:
        """
        Construct current clamp according to (time, amplitude) linear envelope
        """
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    @property
    def envelope(self) -> list[tuple[float, float]]:
        """
        List of (time [ms], amplitude [nA]) points comprising the piecewise linear envelope
        """
    @property
    def frequency(self) -> float:
        """
        Oscillation frequency (kHz), zero implies DC stimulus.
        """
    @property
    def phase(self) -> float:
        """
        Oscillation initial phase (rad)
        """
class int_concentration:
    """
    Setting the initial internal ion concentration.
    """
    def __init__(self, arg0: str, arg1: units.quantity) -> None:
        ...
    def __repr__(self) -> str:
        ...
class ion_data:
    @property
    def charge(self) -> int:
        """
        Valence.
        """
    @property
    def diffusivity(self) -> float | None:
        """
        Diffusivity.
        """
    @property
    def external_concentration(self) -> float | None:
        """
        External concentration.
        """
    @property
    def internal_concentration(self) -> float | None:
        """
        Internal concentration.
        """
    @property
    def reversal_concentration(self) -> float | None:
        """
        Reversal potential.
        """
    @property
    def reversal_potential(self) -> float | None:
        """
        Reversal potential.
        """
    @property
    def reversal_potential_method(self) -> str:
        """
        Reversal potential method.
        """
class ion_dependency:
    """
    Information about a mechanism's dependence on an ion species.
    """
    def __init__(self, arg0: ion_dependency) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    @property
    def read_rev_pot(self) -> bool:
        ...
    @property
    def write_ext_con(self) -> bool:
        ...
    @property
    def write_int_con(self) -> bool:
        ...
    @property
    def write_rev_pot(self) -> bool:
        ...
class ion_diffusivity:
    """
    Setting the ion diffusivity.
    """
    def __init__(self, arg0: str, arg1: units.quantity) -> None:
        ...
    def __repr__(self) -> str:
        ...
class ion_settings:
    pass
class isometry:
    @staticmethod
    @typing.overload
    def rotate(theta: float, x: float, y: float, z: float) -> isometry:
        """
        Construct a rotation isometry of angle theta about the axis in direction (x, y, z).
        """
    @staticmethod
    @typing.overload
    def rotate(theta: float, axis: tuple) -> isometry:
        """
        Construct a rotation isometry of angle theta about the given axis in the direction described by a tuple.
        """
    @staticmethod
    @typing.overload
    def translate(x: float, y: float, z: float) -> isometry:
        """
        Construct a translation isometry from displacements x, y, and z.
        """
    @staticmethod
    @typing.overload
    def translate(arg0: tuple) -> isometry:
        """
        Construct a translation isometry from the first three components of a tuple.
        """
    @staticmethod
    @typing.overload
    def translate(arg0: mpoint) -> isometry:
        """
        Construct a translation isometry from the x, y, and z components of an mpoint.
        """
    @typing.overload
    def __call__(self, arg0: mpoint) -> mpoint:
        """
        Apply isometry to mpoint argument.
        """
    @typing.overload
    def __call__(self, arg0: tuple) -> tuple:
        """
        Apply isometry to first three components of tuple argument.
        """
    def __init__(self) -> None:
        """
        Construct a trivial isometry.
        """
    def __mul__(self, arg0: isometry) -> isometry:
        ...
class junction:
    """
    For placing a gap-junction mechanism on a locset.
    """
    @typing.overload
    def __init__(self, arg0: str) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: mechanism) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: str, arg1: dict[str, float]) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: str, **kwargs) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: mechanism, arg1: dict[str, float]) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: mechanism, **kwargs) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    @property
    def mech(self) -> mechanism:
        """
        The underlying mechanism.
        """
class label_dict:
    """
    A dictionary of labelled region and locset definitions, with a
    unique label assigned to each definition.
    """
    @staticmethod
    def append(*args, **kwargs) -> None:
        """
        Import the entries of a another label dictionary with an optional prefix.
        """
    def __contains__(self, arg0: str) -> bool:
        ...
    def __getitem__(self, arg0: str) -> str:
        ...
    @typing.overload
    def __init__(self) -> None:
        """
        Create an empty label dictionary.
        """
    @typing.overload
    def __init__(self, arg0: dict[str, str]) -> None:
        """
        Initialize a label dictionary from a dictionary with string labels as keys, and corresponding definitions as strings.
        """
    @typing.overload
    def __init__(self, arg0: label_dict) -> None:
        """
        Initialize a label dictionary from another one
        """
    @typing.overload
    def __init__(self, arg0: typing.Iterator) -> None:
        """
        Initialize a label dictionary from an iterable of key, definition pairs
        """
    def __iter__(self) -> typing.Iterator:
        ...
    def __len__(self) -> int:
        ...
    def __repr__(self) -> str:
        ...
    def __setitem__(self, arg0: str, arg1: str) -> None:
        ...
    def __str__(self) -> str:
        ...
    def add_swc_tags(self) -> label_dict:
        """
        Add standard SWC tagged regions.
         - soma: (tag 1)
         - axon: (tag 2)
         - dend: (tag 3)
         - apic: (tag 4)
        """
    def items(self) -> typing.Iterator:
        ...
    def keys(self) -> typing.Iterator:
        ...
    def update(self, other: label_dict) -> None:
        """
        The label_dict to be importedImport the entries of a another label dictionary.
        """
    def values(self) -> typing.Iterator:
        ...
    @property
    def locsets(self) -> list[str]:
        """
        The locset definitions.
        """
    @property
    def regions(self) -> list[str]:
        """
        The region definitions.
        """
class lif_cell:
    """
    A leaky integrate-and-fire cell.
    """
    def __init__(self, source_label: str, target_label: str, *, tau_m: units.quantity | None = None, V_th: units.quantity | None = None, C_m: units.quantity | None = None, E_L: units.quantity | None = None, V_m: units.quantity | None = None, t_ref: units.quantity | None = None) -> None:
        """
        Construct a lif cell with one source labeled 'source_label', and one target labeled 'target_label'.
        """
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    @property
    def C_m(self) -> units.quantity:
        """
        Membrane capacitance [pF].
        """
    @C_m.setter
    def C_m(self, arg0: units.quantity) -> None:
        ...
    @property
    def E_L(self) -> units.quantity:
        """
        Resting potential [mV].
        """
    @E_L.setter
    def E_L(self, arg0: units.quantity) -> None:
        ...
    @property
    def E_R(self) -> units.quantity:
        """
        Reset potential [mV].
        """
    @E_R.setter
    def E_R(self, arg0: units.quantity) -> None:
        ...
    @property
    def V_m(self) -> units.quantity:
        """
        Initial value of the Membrane potential [mV].
        """
    @V_m.setter
    def V_m(self, arg0: units.quantity) -> None:
        ...
    @property
    def V_th(self) -> units.quantity:
        """
        Firing threshold [mV].
        """
    @V_th.setter
    def V_th(self, arg0: units.quantity) -> None:
        ...
    @property
    def source(self) -> str:
        """
        Label of the single build-in source on the cell.
        """
    @source.setter
    def source(self, arg0: str) -> None:
        ...
    @property
    def t_ref(self) -> units.quantity:
        """
        Refractory period [ms].
        """
    @t_ref.setter
    def t_ref(self, arg0: units.quantity) -> None:
        ...
    @property
    def target(self) -> str:
        """
        Label of the single build-in target on the cell.
        """
    @target.setter
    def target(self, arg0: str) -> None:
        ...
    @property
    def tau_m(self) -> units.quantity:
        """
        Membrane potential decaying constant [ms].
        """
    @tau_m.setter
    def tau_m(self, arg0: units.quantity) -> None:
        ...
class lif_probe_metadata:
    """
    Probe metadata associated with a LIF cell probe.
    """
class location:
    """
    A location on a cable cell.
    """
    __hash__: typing.ClassVar[None] = None
    def __eq__(self, arg0: location) -> bool:
        ...
    def __init__(self, branch: int, pos: float) -> None:
        """
        Construct a location specification holding:
          branch:   The id of the branch.
          pos:      The relative position (from 0., proximal, to 1., distal) on the branch.
        """
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    @property
    def branch(self) -> int:
        """
        The id of the branch.
        """
    @property
    def pos(self) -> float:
        """
        The relative position on the branch (∈ [0.,1.], where 0. means proximal and 1. distal).
        """
class mechanism:
    @typing.overload
    def __init__(self, name: str) -> None:
        """
        The name of the mechanism
        """
    @typing.overload
    def __init__(self, name: str, params: dict[str, float]) -> None:
        """
        Example usage setting parameters:
          m = arbor.mechanism('expsyn', {'tau': 1.4})
        will create parameters for the 'expsyn' mechanism, with the provided value
        for 'tau' overrides the default. If a parameter is not set, the default
        (as defined in NMODL) is used.
        
        Example overriding a global parameter:
          m = arbor.mechanism('nernst/R=8.3145,F=96485')
        """
    @typing.overload
    def __init__(self, name: str, **kwargs) -> None:
        """
        Example usage setting parameters:
          m = arbor.mechanism('expsyn', tau=1.4})
        will create parameters for the 'expsyn' mechanism, with the provided value
        for 'tau' overrides the default. If a parameter is not set, the default
        (as defined in NMODL) is used.
        
        Example overriding a global parameter:
          m = arbor.mechanism('nernst/R=8.3145,F=96485')
        """
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    def set(self, name: str, value: float) -> None:
        """
        Set parameter value.
        """
    @property
    def name(self) -> str:
        """
        The name of the mechanism.
        """
    @property
    def values(self) -> dict[str, float]:
        """
        A dictionary of parameter values with parameter name as key.
        """
class mechanism_field:
    """
    Basic information about a mechanism field.
    """
    def __init__(self, arg0: mechanism_field) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    @property
    def default(self) -> float:
        ...
    @property
    def max(self) -> float:
        ...
    @property
    def min(self) -> float:
        ...
    @property
    def units(self) -> str:
        ...
class mechanism_info:
    """
    Meta data about a mechanism's fields and ion dependendencies.
    """
    def __init__(self, arg0: mechanism_info) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    @property
    def globals(self) -> dict[str, mechanism_field]:
        """
        Global fields have one value common to an instance of a mechanism, are constant in time and set at instantiation.
        """
    @property
    def ions(self) -> dict[str, ion_dependency]:
        """
        Ion dependencies.
        """
    @property
    def kind(self) -> str:
        """
        String representation of the kind of the mechanism.
        """
    @property
    def linear(self) -> bool:
        """
        True if a synapse mechanism has linear current contributions so that multiple instances on the same compartment can be coalesced.
        """
    @property
    def parameters(self) -> dict[str, mechanism_field]:
        """
        Parameter fields may vary across the extent of a mechanism, but are constant in time and set at instantiation.
        """
    @property
    def post_events(self) -> bool:
        """
        True if a synapse mechanism has a `POST_EVENT` procedure defined.
        """
    @property
    def state(self) -> dict[str, mechanism_field]:
        """
        State fields vary in time and across the extent of a mechanism, and potentially can be sampled at run-time.
        """
class membrane_capacitance:
    """
    Setting the membrane capacitance.
    """
    def __init__(self, arg0: units.quantity) -> None:
        ...
    def __repr__(self) -> str:
        ...
class membrane_potential:
    """
    Setting the initial membrane voltage.
    """
    def __init__(self, arg0: units.quantity, arg1: str | None) -> None:
        ...
    def __repr__(self) -> str:
        ...
class meter_manager:
    """
    Manage metering by setting checkpoints and starting the timing region.
    """
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    def checkpoint(self, name: str, context: context) -> None:
        """
        Create a new checkpoint. Records the time since the last checkpoint(or the call to start if no previous checkpoints exist),and restarts the timer for the next checkpoint.
        """
    def start(self, context: context) -> None:
        """
        Start the metering. Records a time stamp,             that marks the start of the first checkpoint timing region.
        """
    @property
    def checkpoint_names(self) -> list[str]:
        """
        A list of all metering checkpoint names.
        """
    @property
    def times(self) -> list[float]:
        """
        A list of all metering times.
        """
class meter_report:
    """
    Summarises the performance meter results, used to print a report to screen or file.
    If a distributed context is used, the report will contain a summary of results from all MPI ranks.
    """
    def __init__(self, manager: meter_manager, context: context) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
class morphology:
    """
    A cell morphology.
    """
    def __init__(self, arg0: segment_tree) -> None:
        ...
    def __str__(self) -> str:
        ...
    def branch_children(self, i: int) -> list[int]:
        """
        The child branches of branch i.
        """
    def branch_parent(self, i: int) -> int:
        """
        The parent branch of branch i.
        """
    def branch_segments(self, i: int) -> list[msegment]:
        """
        A list of the segments in branch i, ordered from proximal to distal ends of the branch.
        """
    def to_segment_tree(self) -> segment_tree:
        """
        Convert this morphology to a segment_tree.
        """
    @property
    def empty(self) -> bool:
        """
        Whether the morphology is empty.
        """
    @property
    def num_branches(self) -> int:
        """
        The number of branches in the morphology.
        """
class morphology_provider:
    def __init__(self, morphology: morphology) -> None:
        """
        Construct a morphology provider.
        """
    def reify_locset(self, arg0: str) -> list[location]:
        """
        Turn a locset into a list of locations.
        """
    def reify_region(self, arg0: str) -> extent:
        """
        Turn a region into an extent.
        """
class mpoint:
    __hash__: typing.ClassVar[None] = None
    def __eq__(self, arg0: mpoint) -> bool:
        ...
    @typing.overload
    def __init__(self, x: float, y: float, z: float, radius: float) -> None:
        """
        Create an mpoint object from parameters x, y, z, and radius, specified in µm.
        """
    @typing.overload
    def __init__(self, arg0: tuple) -> None:
        """
        Create an mpoint object from a tuple (x, y, z, radius), specified in µm.
        """
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    @property
    def radius(self) -> float:
        """
        Radius of cable at sample location centred at coordinates [μm].
        """
    @property
    def x(self) -> float:
        """
        X coordinate [μm].
        """
    @property
    def y(self) -> float:
        """
        Y coordinate [μm].
        """
    @property
    def z(self) -> float:
        """
        Z coordinate [μm].
        """
class msegment:
    @property
    def dist(self) -> mpoint:
        """
        the location and radius of the distal end.
        """
    @property
    def prox(self) -> mpoint:
        """
        the location and radius of the proximal end.
        """
    @property
    def tag(self) -> int:
        """
        tag meta-data.
        """
class neuroml:
    def __init__(self, arg0: typing.Any) -> None:
        """
        Construct NML morphology from filename or stream.
        """
    def cell_ids(self) -> list[str]:
        """
        Query top-level cells.
        """
    def cell_morphology(self, cell_id: str, allow_spherical_root: bool = False) -> neuroml_morph_data | None:
        """
        Retrieve nml_morph_data associated with cell_id.
        """
    def morphology(self, morph_id: str, allow_spherical_root: bool = False) -> neuroml_morph_data | None:
        """
        Retrieve top-level nml_morph_data associated with morph_id.
        """
    def morphology_ids(self) -> list[str]:
        """
        Query top-level standalone morphologies.
        """
class neuroml_morph_data:
    def groups(self) -> label_dict:
        """
        Label dictionary containing one region expression for each segmentGroup id.
        """
    def named_segments(self) -> label_dict:
        """
        Label dictionary containing one region expression for each name applied to one or more segments.
        """
    def segments(self) -> label_dict:
        """
        Label dictionary containing one region expression for each segment id.
        """
    @property
    def cell_id(self) -> str | None:
        """
        Cell id, or empty if morphology was taken from a top-level <morphology> element.
        """
    @property
    def group_segments(self) -> dict[str, list[int]]:
        """
        Map from segmentGroup ids to their corresponding segment ids.
        """
    @property
    def id(self) -> str:
        """
        Morphology id.
        """
    @property
    def morphology(self) -> morphology:
        """
        Morphology constructed from a signle NeuroML <morphology> element.
        """
class partition_hint:
    """
    Provide a hint on how the cell groups should be partitioned.
    """
    max_size: typing.ClassVar[int] = 18446744073709551615
    def __init__(self, cpu_group_size: int = 1, gpu_group_size: int = 18446744073709551615, prefer_gpu: bool = True) -> None:
        """
        Construct a partition hint with arguments:
          cpu_group_size: The size of cell group assigned to CPU, each cell in its own group by default.
                          Must be positive, else set to default value.
          gpu_group_size: The size of cell group assigned to GPU, all cells in one group by default.
                          Must be positive, else set to default value.
          prefer_gpu:     Whether GPU is preferred, True by default.
        """
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    @property
    def cpu_group_size(self) -> int:
        """
        The size of cell group assigned to CPU.
        """
    @cpu_group_size.setter
    def cpu_group_size(self, arg0: int) -> None:
        ...
    @property
    def gpu_group_size(self) -> int:
        """
        The size of cell group assigned to GPU.
        """
    @gpu_group_size.setter
    def gpu_group_size(self, arg0: int) -> None:
        ...
    @property
    def prefer_gpu(self) -> bool:
        """
        Whether GPU usage is preferred.
        """
    @prefer_gpu.setter
    def prefer_gpu(self, arg0: bool) -> None:
        ...
class place_pwlin:
    def __init__(self, morphology: morphology, isometry: isometry = ...) -> None:
        """
        Construct a piecewise-linear placement object from the given morphology and optional isometry.
        """
    def all_at(self, location: location) -> list[mpoint]:
        """
        Return list of all possible interpolated mpoints corresponding to the location argument.
        """
    def all_segments(self, arg0: list[cable]) -> list[msegment]:
        """
        Return maximal list of non-overlapping full or partial msegments whose union is coterminous with the extent of the given list of cables.
        """
    def at(self, location: location) -> mpoint:
        """
        Return an interpolated mpoint corresponding to the location argument.
        """
    def closest(self, arg0: float, arg1: float, arg2: float) -> tuple:
        """
        Find the location on the morphology that is closest to a 3d point. Returns the location and its distance from the point.
        """
    def segments(self, arg0: list[cable]) -> list[msegment]:
        """
        Return minimal list of full or partial msegments whose union is coterminous with the extent of the given list of cables.
        """
class poisson_schedule(schedule_base):
    """
    Describes a schedule according to a Poisson process within the interval [tstart, tstop).
    """
    def __init__(self, freq: units.quantity, *, tstart: units.quantity = ..., seed: int = 0, tstop: units.quantity | None = None) -> None:
        """
        Construct a Poisson schedule with arguments:
          tstart: The delivery time of the first event in the sequence [ms], 0 by default.
          freq:   The expected frequency [kHz].
          seed:   The seed for the random number generator, 0 by default.
          tstop:  No events delivered after this time [ms], None by default.
        """
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    def events(self, arg0: units.quantity, arg1: units.quantity) -> list[float]:
        """
        A view of monotonically increasing time values in the half-open interval [t0, t1).
        """
    @property
    def freq(self) -> units.quantity:
        """
        The expected frequency [kHz].
        """
    @freq.setter
    def freq(self, arg1: units.quantity) -> None:
        ...
    @property
    def seed(self) -> int:
        """
        The seed for the random number generator.
        """
    @seed.setter
    def seed(self, arg0: int) -> None:
        ...
    @property
    def tstart(self) -> units.quantity:
        """
        The delivery time of the first event in the sequence [ms].
        """
    @tstart.setter
    def tstart(self, arg1: units.quantity) -> None:
        ...
    @property
    def tstop(self) -> units.quantity:
        """
        No events delivered after this time [ms].
        """
    @tstop.setter
    def tstop(self, arg1: units.quantity) -> None:
        ...
class probe:
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
class proc_allocation:
    """
    Enumerates the computational resources on a node to be used for simulation.
    """
    def __init__(self, *, threads: int = 1, gpu_id: typing.Any = None, bind_procs: bool = False, bind_threads: bool = False) -> None:
        """
        Construct an allocation with arguments:
          threads:      The number of threads available locally for execution. Must be set to 1 at minimum. 1 by default.
          gpu_id:       The identifier of the GPU to use, None by default.
          bind_procs:   Create process binding mask.
          bind_threads: Create thread binding mask.
        """
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    @property
    def bind_procs(self) -> bool:
        """
        Try to bind MPI procs?
        """
    @bind_procs.setter
    def bind_procs(self, arg1: bool) -> None:
        ...
    @property
    def bind_threads(self) -> bool:
        """
        Try to bind threads?
        """
    @bind_threads.setter
    def bind_threads(self, arg1: bool) -> None:
        ...
    @property
    def gpu_id(self) -> int | None:
        """
        The identifier of the GPU to use.
        Corresponds to the integer parameter used to identify GPUs in CUDA API calls.
        """
    @gpu_id.setter
    def gpu_id(self, arg1: typing.Any) -> None:
        ...
    @property
    def has_gpu(self) -> bool:
        """
        Whether a GPU is being used (True/False).
        """
    @property
    def threads(self) -> int:
        """
        The number of threads available locally for execution.
        """
    @threads.setter
    def threads(self, arg1: int) -> None:
        ...
class recipe:
    """
    A description of a model, describing the cells and the network via a cell-centric interface.
    """
    def __init__(self) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    def cell_description(self, gid: int) -> typing.Any:
        """
        High level description of the cell with global identifier gid.
        """
    def cell_kind(self, gid: int) -> cell_kind:
        """
        The kind of cell with global identifier gid.
        """
    def connections_on(self, gid: int) -> list[connection]:
        """
        A list of all the incoming connections to gid, [] by default.
        """
    def event_generators(self, gid: int) -> list[typing.Any]:
        """
        A list of all the event generators that are attached to gid, [] by default.
        """
    def external_connections_on(self, gid: int) -> list[connection]:
        """
        A list of all the incoming connections from _remote_ locations to gid, [] by default.
        """
    def gap_junctions_on(self, gid: int) -> list[gap_junction_connection]:
        """
        A list of the gap junctions connected to gid, [] by default.
        """
    def global_properties(self, kind: cell_kind) -> typing.Any:
        """
        The default properties applied to all cells of type 'kind' in the model.
        """
    def num_cells(self) -> int:
        """
        The number of cells in the model.
        """
    def probes(self, gid: int) -> list[probe]:
        """
        The probes to allow monitoring.
        """
class regular_schedule(schedule_base):
    """
    Describes a regular schedule with multiples of dt within the interval [tstart, tstop).
    """
    @typing.overload
    def __init__(self, tstart: units.quantity, dt: units.quantity, tstop: units.quantity | None = None) -> None:
        """
        Construct a regular schedule with arguments:
          tstart: The delivery time of the first event in the sequence [ms].
          dt:     The interval between time points [ms].
          tstop:  No events delivered after this time [ms], None by default.
        """
    @typing.overload
    def __init__(self, dt: units.quantity) -> None:
        """
        Construct a regular schedule, starting from t = 0 and never terminating, with arguments:
          dt:     The interval between time points [ms].
        """
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    def events(self, arg0: float, arg1: float) -> list[float]:
        """
        A view of monotonically increasing time values in the half-open interval [t0, t1).
        """
    @property
    def dt(self) -> units.quantity:
        """
        The interval between time points [ms].
        """
    @dt.setter
    def dt(self, arg1: units.quantity) -> None:
        ...
    @property
    def tstart(self) -> units.quantity:
        """
        The delivery time of the first event in the sequence [ms].
        """
    @tstart.setter
    def tstart(self, arg1: units.quantity) -> None:
        ...
    @property
    def tstop(self) -> units.quantity | None:
        """
        No events delivered after this time [ms].
        """
    @tstop.setter
    def tstop(self, arg1: units.quantity | None) -> None:
        ...
class reversal_potential:
    """
    Setting the initial reversal potential.
    """
    def __init__(self, arg0: str, arg1: units.quantity) -> None:
        ...
    def __repr__(self) -> str:
        ...
class reversal_potential_method:
    """
    Describes the mechanism used to compute eX for ion X.
    """
    def __init__(self, arg0: str, arg1: mechanism) -> None:
        ...
    def __repr__(self) -> str:
        ...
class scaled_mechanism:
    """
    For painting a scaled density mechanism on a region.
    """
    @typing.overload
    def __init__(self, arg0: density) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: density, arg1: dict[str, str]) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: density, **kwargs) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    def scale(self, name: str, ex: str) -> scaled_mechanism:
        """
        Add a scaling expression to a parameter.
        """
class schedule_base:
    """
    Schedule abstract base class.
    """
class segment_tree:
    def __init__(self) -> None:
        ...
    def __str__(self) -> str:
        ...
    @typing.overload
    def append(self, parent: int, prox: mpoint, dist: mpoint, tag: int) -> int:
        """
        Append a segment to the tree.
        """
    @typing.overload
    def append(self, parent: int, dist: mpoint, tag: int) -> int:
        """
        Append a segment to the tree.
        """
    @typing.overload
    def append(self, parent: int, x: float, y: float, z: float, radius: float, tag: int) -> int:
        """
        Append a segment to the tree, using the distal location of the parent segment as the proximal end.
        """
    def apply_isometry(self, arg0: isometry) -> segment_tree:
        """
        Apply an isometry to all segments in the tree.
        """
    def equivalent(self, arg0: segment_tree) -> bool:
        """
        Two trees are equivalent, but not neccessarily identical, ie they have the same segments and structure.
        """
    def is_fork(self, i: int) -> bool:
        """
        True if segment has more than one child.
        """
    def is_root(self, i: int) -> bool:
        """
        True if segment has no parent.
        """
    def is_terminal(self, i: int) -> bool:
        """
        True if segment has no children.
        """
    def join_at(self, arg0: int, arg1: segment_tree) -> segment_tree:
        """
        Join two subtrees at a given id, such that said id becomes the parent of the inserted sub-tree.
        """
    def reserve(self, arg0: int) -> None:
        ...
    def split_at(self, arg0: int) -> tuple[segment_tree, segment_tree]:
        """
        Split into a pair of trees at the given id, such that one tree is the subtree rooted at id and the other is the original tree without said subtree.
        """
    def tag_roots(self, arg0: int) -> list[int]:
        """
        Get roots of tag region of this segment tree.
        """
    @property
    def empty(self) -> bool:
        """
        Indicates whether the tree is empty (i.e. whether it has size 0)
        """
    @property
    def parents(self) -> list[int]:
        """
        A list with the parent index of each segment.
        """
    @property
    def segments(self) -> list[msegment]:
        """
        A list of the segments.
        """
    @property
    def size(self) -> int:
        """
        The number of segments in the tree.
        """
class selection_policy:
    """
    Enumeration used to identify a selection policy, used by the model for selecting one of possibly multiple locations on the cell associated with a labeled item.
    
    Members:
    
      round_robin : Iterate round-robin over all possible locations.
    
      round_robin_halt : Halts at the current location until the round_robin policy is called (again).
    
      univalent : Assert that there is only one possible location associated with a labeled item on the cell. The model throws an exception if the assertion fails.
    """
    __members__: typing.ClassVar[dict[str, selection_policy]]  # value = {'round_robin': <selection_policy.round_robin: 0>, 'round_robin_halt': <selection_policy.round_robin_halt: 1>, 'univalent': <selection_policy.univalent: 2>}
    round_robin: typing.ClassVar[selection_policy]  # value = <selection_policy.round_robin: 0>
    round_robin_halt: typing.ClassVar[selection_policy]  # value = <selection_policy.round_robin_halt: 1>
    univalent: typing.ClassVar[selection_policy]  # value = <selection_policy.univalent: 2>
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class simulation:
    """
    The executable form of a model.
    A simulation is constructed from a recipe, and then used to update and monitor model state.
    """
    @staticmethod
    def deserialize(*args, **kwargs) -> None:
        ...
    def __init__(self, recipe: recipe, context: context | None = None, domains: domain_decomposition | None = None, seed: int = 0) -> None:
        """
        Initialize the model described by a recipe, with cells and network distributed
        according to the domain decomposition and computational resources described by a
        context. Initialize PRNG using seed
        """
    def clear_samplers(self) -> None:
        """
        Clearing spike and sample information. restoring memory
        """
    @typing.overload
    def probe_metadata(self, probeset_id: cell_address) -> list:
        """
        Retrieve metadata associated with given probe id.
        """
    @typing.overload
    def probe_metadata(self, addr: tuple[int, str]) -> list:
        """
        Retrieve metadata associated with given probe id.
        """
    @typing.overload
    def probe_metadata(self, gid: int, tag: str) -> list:
        """
        Retrieve metadata associated with given probe id.
        """
    def progress_banner(self) -> None:
        """
        Show a text progress bar during simulation.
        """
    def record(self, arg0: spike_recording) -> None:
        """
        Disable or enable local or global spike recording.
        """
    def remove_all_samplers(self, arg0: int) -> None:
        """
        Remove all sampling on the simulatr.
        """
    def remove_sampler(self, handle: int) -> None:
        """
        Remove sampling associated with the given handle.
        """
    def reset(self) -> None:
        """
        Reset the state of the simulation to its initial state.
        """
    def run(self, tfinal: units.quantity, dt: units.quantity = ...) -> float:
        """
        Run the simulation from current simulation time to tfinal [ms], with maximum time step size dt [ms].
        """
    @typing.overload
    def sample(self, probeset_id: cell_address, schedule: schedule_base) -> int:
        """
        Record data from probes with given probeset_id according to supplied schedule.
        Returns handle for retrieving data or removing the sampling.
        """
    @typing.overload
    def sample(self, gid: int, tag: str, schedule: schedule_base) -> int:
        """
        Record data from probes with given probeset_id=(gid, tag) according to supplied schedule.
        Returns handle for retrieving data or removing the sampling.
        """
    @typing.overload
    def sample(self, probeset_id: tuple[int, str], schedule: schedule_base) -> int:
        """
        Record data from probes with given probeset_id=(gid, tag) according to supplied schedule.
        Returns handle for retrieving data or removing the sampling.
        """
    def samples(self, handle: int) -> list:
        """
        Retrieve sample data as a list, one element per probe associated with the query.
        """
    def serialize(self) -> str:
        """
        Serialize the simulation object to a JSON string.
        """
    def set_remote_spike_filter(self, pred: typing.Callable[[spike], bool]) -> None:
        """
        Add a callback to filter spikes going out over external connections. `pred` isa callable on the `spike` type. **Caution**: This will be extremely slow; use C++ if you want to make use of this.
        """
    def spikes(self) -> typing.Any:
        """
        Retrieve recorded spikes as numpy array.
        """
    def update(self, recipe: recipe) -> None:
        """
        Rebuild the connection table from recipe::connections_on and the eventgenerators based on recipe::event_generators.
        """
class single_cell_model:
    """
    Wrapper for simplified description, and execution, of single cell models.
    """
    @typing.overload
    def __init__(self, tree: segment_tree, decor: decor, labels: label_dict = ...) -> None:
        """
        Build single cell model from cable cell components
        """
    @typing.overload
    def __init__(self, morph: morphology, decor: decor, labels: label_dict = ...) -> None:
        """
        Build single cell model from cable cell components
        """
    @typing.overload
    def __init__(self, cell: cable_cell) -> None:
        """
        Initialise a single cell model for a cable cell.
        """
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    def event_generator(self, event_generator: event_generator) -> None:
        """
        Register an event generator.
         event_generator: An Arbor event generator.
        """
    @typing.overload
    def probe(self, what: str, where: str, tag: str, frequency: units.quantity) -> None:
        """
        Sample a variable on the cell.
         what:      Name of the variable to record (currently only 'voltage').
         where:     Location on cell morphology at which to sample the variable.
         tag:       Unique name for this probe.
         frequency: The target frequency at which to sample [kHz].
        """
    @typing.overload
    def probe(self, what: str, where: location, tag: str, frequency: units.quantity) -> None:
        """
        Sample a variable on the cell.
         what:      Name of the variable to record (currently only 'voltage').
         where:     Location on cell morphology at which to sample the variable.
         tag:       Unique name for this probe.
         frequency: The target frequency at which to sample [kHz].
        """
    def run(self, tfinal: units.quantity, dt: units.quantity = ...) -> None:
        """
        Run model from t=0 to t=tfinal ms.
        """
    @property
    def cable_cell(self) -> cable_cell:
        """
        The cable cell held by this model.
        """
    @property
    def properties(self) -> cable_global_properties:
        """
        Global properties.
        """
    @properties.setter
    def properties(self, arg0: cable_global_properties) -> None:
        ...
    @property
    def spikes(self) -> list[float]:
        """
        Holds spike times [ms] after a call to run().
        """
    @property
    def traces(self) -> list[trace]:
        """
        Holds sample traces after a call to run().
        """
class spike:
    def __init__(self, arg0: cell_member, arg1: float) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    @property
    def source(self) -> cell_member:
        """
        The global identifier of the cell.
        """
    @source.setter
    def source(self, arg0: cell_member) -> None:
        ...
    @property
    def time(self) -> float:
        """
        The time of spike.
        """
    @time.setter
    def time(self, arg0: float) -> None:
        ...
class spike_recording:
    """
    Members:
    
      off
    
      local
    
      all
    """
    __members__: typing.ClassVar[dict[str, spike_recording]]  # value = {'off': <spike_recording.off: 0>, 'local': <spike_recording.local: 1>, 'all': <spike_recording.all: 2>}
    all: typing.ClassVar[spike_recording]  # value = <spike_recording.all: 2>
    local: typing.ClassVar[spike_recording]  # value = <spike_recording.local: 1>
    off: typing.ClassVar[spike_recording]  # value = <spike_recording.off: 0>
    def __eq__(self, other: typing.Any) -> bool:
        ...
    def __getstate__(self) -> int:
        ...
    def __hash__(self) -> int:
        ...
    def __index__(self) -> int:
        ...
    def __init__(self, value: int) -> None:
        ...
    def __int__(self) -> int:
        ...
    def __ne__(self, other: typing.Any) -> bool:
        ...
    def __repr__(self) -> str:
        ...
    def __setstate__(self, state: int) -> None:
        ...
    def __str__(self) -> str:
        ...
    @property
    def name(self) -> str:
        ...
    @property
    def value(self) -> int:
        ...
class spike_source_cell:
    """
    A spike source cell, that generates a user-defined sequence of spikes that act as inputs for other cells in the network.
    """
    @typing.overload
    def __init__(self, source_label: str, schedule: regular_schedule) -> None:
        """
        Construct a spike source cell with a single source labeled 'source_label'.
        The cell generates spikes on 'source_label' at regular intervals.
        """
    @typing.overload
    def __init__(self, source_label: str, schedule: explicit_schedule) -> None:
        """
        Construct a spike source cell with a single source labeled 'source_label'.
        The cell generates spikes on 'source_label' at a sequence of user-defined times.
        """
    @typing.overload
    def __init__(self, source_label: str, schedule: poisson_schedule) -> None:
        """
        Construct a spike source cell with a single source labeled 'source_label'.
        The cell generates spikes on 'source_label' at times defined by a Poisson sequence.
        """
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
class synapse:
    """
    For placing a synaptic mechanism on a locset.
    """
    @typing.overload
    def __init__(self, arg0: str) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: mechanism) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: str, arg1: dict[str, float]) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: mechanism, arg1: dict[str, float]) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: str, **kwargs) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: mechanism, **kwargs) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    @property
    def mech(self) -> mechanism:
        """
        The underlying mechanism.
        """
class temperature:
    """
    Setting the temperature.
    """
    def __init__(self, arg0: units.quantity) -> None:
        ...
    def __repr__(self) -> str:
        ...
class threshold_detector:
    """
    A spike detector, generates a spike when voltage crosses a threshold. Can be used as source endpoint for an arbor.connection.
    """
    def __init__(self, threshold: units.quantity) -> None:
        """
        Voltage threshold of spike detector [mV]
        """
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    @property
    def threshold(self) -> float:
        """
        Voltage threshold of spike detector [mV]
        """
class trace:
    """
    Values and meta-data for a sample-trace on a single cell model.
    """
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    @property
    def location(self) -> location:
        """
        Location on cell morphology.
        """
    @property
    def time(self) -> list[float]:
        """
        Time stamps of samples [ms].
        """
    @property
    def value(self) -> list[float]:
        """
        Sample values.
        """
    @property
    def variable(self) -> str:
        """
        Name of the variable being recorded.
        """
class voltage_process:
    """
    For painting a voltage_process mechanism on a region.
    """
    @typing.overload
    def __init__(self, arg0: str) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: mechanism) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: str, arg1: dict[str, float]) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: mechanism, arg1: dict[str, float]) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: mechanism, **kwargs) -> None:
        ...
    @typing.overload
    def __init__(self, arg0: str, **kwargs) -> None:
        ...
    def __repr__(self) -> str:
        ...
    def __str__(self) -> str:
        ...
    @property
    def mech(self) -> mechanism:
        """
        The underlying mechanism.
        """
def allen_catalogue() -> catalogue:
    ...
def bbp_catalogue() -> catalogue:
    ...
def cable_probe_axial_current(where: str, tag: str) -> probe:
    """
    Probe specification for cable cell axial current at points in a location set.
    """
def cable_probe_density_state(where: str, mechanism: str, state: str, tag: str) -> probe:
    """
    Probe specification for a cable cell density mechanism state variable at points in a location set.
    """
def cable_probe_density_state_cell(mechanism: str, state: str, tag: str) -> probe:
    """
    Probe specification for a cable cell density mechanism state variable on each cable in each CV where defined.
    """
def cable_probe_ion_current_cell(ion: str, tag: str) -> probe:
    """
    Probe specification for cable cell ionic current across each cable in each CV.
    """
def cable_probe_ion_current_density(where: str, ion: str, tag: str) -> probe:
    """
    Probe specification for cable cell ionic current density at points in a location set.
    """
def cable_probe_ion_diff_concentration(where: str, ion: str, tag: str) -> probe:
    """
    Probe specification for cable cell diffusive ionic concentration at points in a location set.
    """
def cable_probe_ion_diff_concentration_cell(ion: str, tag: str) -> probe:
    """
    Probe specification for cable cell diffusive ionic concentration for each cable in each CV.
    """
def cable_probe_ion_ext_concentration(where: str, ion: str, tag: str) -> probe:
    """
    Probe specification for cable cell external ionic concentration at points in a location set.
    """
def cable_probe_ion_ext_concentration_cell(ion: str, tag: str) -> probe:
    """
    Probe specification for cable cell external ionic concentration for each cable in each CV.
    """
def cable_probe_ion_int_concentration(where: str, ion: str, tag: str) -> probe:
    """
    Probe specification for cable cell internal ionic concentration at points in a location set.
    """
def cable_probe_ion_int_concentration_cell(ion: str, tag: str) -> probe:
    """
    Probe specification for cable cell internal ionic concentration for each cable in each CV.
    """
def cable_probe_membrane_voltage(where: str, tag: str) -> probe:
    """
    Probe specification for cable cell membrane voltage interpolated at points in a location set.
    """
def cable_probe_membrane_voltage_cell(tag: str) -> probe:
    """
    Probe specification for cable cell membrane voltage associated with each cable in each CV.
    """
def cable_probe_point_state(target: int, mechanism: str, state: str, tag: str) -> probe:
    """
    Probe specification for a cable cell point mechanism state variable value at a given target index.
    """
def cable_probe_point_state_cell(mechanism: str, state: str, tag: str) -> probe:
    """
    Probe specification for a cable cell point mechanism state variable value at every corresponding target.
    """
def cable_probe_stimulus_current_cell(tag: str) -> probe:
    """
    Probe specification for cable cell stimulus current across each cable in each CV.
    """
def cable_probe_total_current_cell(tag: str) -> probe:
    """
    Probe specification for cable cell total transmembrane current for each cable in each CV.
    """
def cable_probe_total_ion_current_cell(tag: str) -> probe:
    """
    Probe specification for cable cell total transmembrane current excluding capacitive currents for each cable in each CV.
    """
def cable_probe_total_ion_current_density(where: str, tag: str) -> probe:
    """
    Probe specification for cable cell total transmembrane current density excluding capacitive currents at points in a location set.
    """
def config() -> dict:
    """
    Get Arbor's configuration.
    """
def cv_data(cell: cable_cell) -> cell_cv_data | None:
    """
    Returns a cell_cv_data object representing the CVs comprising the cable-cell according to the discretization policy provided in the decor of the cell. Returns None if no CV-policy was provided in the decor.
    """
def cv_policy_every_segment(domain: str = '(all)') -> cv_policy:
    """
    Policy to create one compartment per component of a region.
    """
def cv_policy_explicit(locset: str, domain: str = '(all)') -> cv_policy:
    """
    Policy to create compartments at explicit locations.
    """
def cv_policy_fixed_per_branch(n: int, domain: str = '(all)') -> cv_policy:
    """
    Policy to use the same number of CVs for each branch.
    """
def cv_policy_max_extent(length: float, domain: str = '(all)') -> cv_policy:
    """
    Policy to use as many CVs as required to ensure that no CV has a length longer than a given value.
    """
def cv_policy_single(domain: str = '(all)') -> cv_policy:
    """
    Policy to create one compartment per component of a region.
    """
def default_catalogue() -> catalogue:
    ...
def intersect_region(reg: str, data: cell_cv_data, integrate_along: str) -> list[tuple]:
    """
    Returns a list of [index, proportion] tuples identifying the CVs present in the region.
    `index` is the index of the CV in the cell_cv_data object provided as an argument.
    `proportion` is the proportion of the CV (itegrated by area or length) included in the region.
    """
def lif_probe_voltage(tag: str) -> probe:
    """
    Probe specification for LIF cell membrane voltage.
    """
def load_asc(filename_or_stream: typing.Any, raw: bool = False) -> segment_tree | asc_morphology:
    """
    Load a morphology or segment_tree (raw=True) and meta data from a Neurolucida ASCII .asc file.
    """
def load_catalogue(arg0: typing.Any) -> catalogue:
    ...
def load_component(filename_or_descriptor: typing.Any) -> cable_component:
    """
    Load arbor-component (decor, morphology, label_dict, cable_cell) from file.
    """
def load_swc_arbor(filename_or_stream: typing.Any, raw: bool = False) -> segment_tree | morphology:
    """
    Generate a morphology/segment_tree (raw=False/True) from an SWC file following the rules prescribed by Arbor.
    Specifically:
     * Single-segment somas are disallowed.
     * There are no special rules related to somata. They can be one or multiple branches
       and other segments can connect anywhere along them.
     * A segment is always created between a sample and its parent, meaning there
       are no gaps in the resulting morphology.
    """
def load_swc_neuron(filename_or_stream: typing.Any, raw: bool = False) -> segment_tree | morphology:
    """
    Generate a morphology/segment_tree (raw=False/True) from an SWC file following the rules prescribed by NEURON.
    See the documentation https://docs.arbor-sim.org/en/latest/fileformat/swc.html
    for a detailed description of the interpretation.
    """
def neuron_cable_properties() -> cable_global_properties:
    """
    default NEURON cable_global_properties
    """
def partition_by_group(recipe: recipe, context: context, groups: list[group_description]) -> domain_decomposition:
    """
    Construct a domain_decomposition that assigned the groups of cell provided as argument 
    to the local hardware resources described by context on the calling rank.
    The cell_groups are guaranteed to be present on the calling rank.
    """
def partition_load_balance(recipe: recipe, context: context, hints: dict[cell_kind, partition_hint] = {}) -> domain_decomposition:
    """
    Construct a domain_decomposition that distributes the cells in the model described by recipe
    over the distributed and local hardware resources described by context.
    Optionally, provide a dictionary of partition hints for certain cell kinds, by default empty.
    """
def print_config() -> None:
    """
    Print Arbor's configuration.
    """
def stochastic_catalogue() -> catalogue:
    ...
@typing.overload
def write_component(object: cable_component, filename_or_descriptor: typing.Any) -> None:
    """
    Write cable_component to file.
    """
@typing.overload
def write_component(object: decor, filename_or_descriptor: typing.Any) -> None:
    """
    Write decor to file.
    """
@typing.overload
def write_component(object: label_dict, filename_or_descriptor: typing.Any) -> None:
    """
    Write label_dict to file.
    """
@typing.overload
def write_component(object: morphology, filename_or_descriptor: typing.Any) -> None:
    """
    Write morphology to file.
    """
@typing.overload
def write_component(object: cable_cell, filename_or_descriptor: typing.Any) -> None:
    """
    Write cable_cell to file.
    """
__version__: str = '0.9.1-dev'
mnpos: int = 4294967295
