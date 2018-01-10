Python Wrapper
==============

All of the code relating to the Python wrapper is in the :file:`python` path of the source code repository.

Arbor uses pybind11_, a header only C++ library, to generate Python wrappers.
When the :class:`ARB_WITH_PYTHON` option is selected during building, the pybind11_ repository is automatically checked out from GitHub into the :file:`python/pybind11` directory
pybind11_ provides the CMake infrastructure used to build the python bindings.

Python Module
-------------

The workflow for installing the module generated from the C++ bindings has not been implemented (suggestions welcome from the pythonistas in the audience.)
The Python module is currently installed in :file:`build/lib/`. On my system, it is installed as :file:`build/lib/pyarb.cpython-36m-x86_64-linux-gnu.so`.

C++ Bindings
------------

For most of types and functions, Arbor uses pybind11 to generate thin wrappers
that expose them directly to Python.
As an example of this approach, take the C++ type :class:`node_info` and the helper
function :func:`get_node_info`, defined in :file:`src/hardware/node_info.hpp`:

.. code-block:: cpp

    struct node_info {
        node_info() = default;
        node_info(unsigned c, unsigned g): num_cpu_cores(c), num_gpus(g) {}

        unsigned num_cpu_cores = 1;
        unsigned num_gpus = 0;
    };

    node_info get_node_info();

The type and function are wrapped as follows using pybind11 in :file:`python/arb.cpp`

.. code-block:: cpp

    // wrap node_info
    pb::class_<arb::hw::node_info> node_info(m, "node_info",
        "Describes the resources on a compute node.");
    node_info
        .def(pb::init<>())
        .def(pb::init<unsigned, unsigned>())
        .def_readwrite("num_cpu_cores", &arb::hw::node_info::num_cpu_cores,
                "The number of available CPU cores.")
        .def_readwrite("num_gpus", &arb::hw::node_info::num_gpus,
                "The number of available GPUs.")
        .def("__str__",  &node_info_string)
        .def("__repr__", &node_info_string);

    // wrap get_node_info() function call
    m.def("get_node_info", &arb::hw::get_node_info,
        "Returns a description of the hardware resources available on the host compute node.");

This is all that is required to wrap both :class:`node_info` and :func:`get_node_info` for use in Python

.. code-block:: pycon

    >>> import pyarb
    >>> node = pyarb.get_node_info()
    >>> print(node)
    <node_info: 4 cpus; 0 gpus>

.. note::

    In the above example, Python can pretty print :class:`node_info` because
    the :class:`__str__` and :class:`__repr__` attributes were defined using
    pybind11. In this case, they use the :func:`node_info_string` function,
    defined in :file:`python/print.hpp`.

Recipes
-------

Arbor provides an interface for user defined recipes in Python, which matches
that used to define recipes in C++.
To hide some C++-specific details from Python users, a shim is used
instead of a directly exposing the C++ :class:`recipe` class to
Python. All of the recipe wrapping is in :file:`python/recipe.hpp`.

The C++ :class:`recipe` interface uses :class:`util::any` and :class:`util::unique_any`
(implementations of C++17's :class:`std::any` type-erased container) as return types on some calls.
Such dynamic type-erasure is performed naturally in Python's :class:`PyObject` storage class.
The shim interface is used to translate :class:`PyObject` types to :class:`util::{unique_}any`,
such that all use of :class:`util::{unique_}any` is restricted to the C++ side.

The following steps are used to interface the C++ and Python recipe definitions

1. The :class:`arb::recipe` interface is replicated in another C++ class
   :class:`arb::py::recipe`. This interface that returns :class:`pybind11::object`
   (equivalent to :class:`PyObject` types) from calls that use type erasure.
2. This :class:`arb::py::recipe` is wrapped for Python, so that user-defined recipes return
   native :class:`PyObject` types. A "trampoline" class `arb::py::recipe_trampoline`
   is used to wrap (see the pybind11_ docs for more information).
3. A shim :class:`py_recipe_shim` class that derives from :class:`arb::recipe`
   holds a :class:`std::shared_ptr` to the `arb::py::recipe`. Calls to 
   The shim forwards calls to `arb::recipe` to a python-side
   arb::py::recipe implementation, and translates the output of the
   arb::py::recipe return values to those used by arb::recipe.

When an C++ interface that takes a :class:`recipe` is wrapped for Python, we use a lambda to
wrap the Python recipe for forwarding to the underlying C++ call. For example, take the
load balancing function :func:`partition_load_balance`, which uses cell kind information
and model size information from the recipe:

.. code-block:: cpp

    // prototype from src/load_balance.hpp
    domain_decomposition partition_load_balance(const recipe& rec, hw::node_info nd) {

    // wrapping in python/arb.cpp
    m.def("partition_load_balance",
        [](std::shared_ptr<arb::py::recipe>& r, const arb::hw::node_info& ni) {
            return arb::partition_load_balance(arb::py_recipe_shim(r), ni);
        },
        "Simple load balancer.", "recipe"_a, "node"_a);

Beware the GIL
--------------

Recipe code written in Python is called from multithreaded C++ code.
To avoid deadlocks, the global interpreter lock, or GIL, must be released when
the C++ is called from Python.

Examples of calls where GIL deadlocks might happen include:

1. calls to :func:`model::run`, where multiple threads might call a user-defined
   :class:`event_generator`.
2. calls to the :class:`model` constructor, which builds cell groups in parallel,
   with multipe threads concurrently querying :class:`arb::py::recipe` for cell decriptions.

Pybind11 provides facilities for releasing and aquiring the GIL.
The pybind11 mechanism used by Arbor for calling back into Python
(see :class:`PYBIND11_OVERLOAD_PURE`) automatically acquires the GIL.
Hence, we are only responsible for releasing the GIL when calling into
multithreaded C++ code that may call back Python, for which we use
:class:`pybind11::gil_scoped_release`, as illustrated for calls to
:func:`model::model` and :func:`model::run`:

.. code-block:: cpp

    namespace pb = pybind11;

    model.def(pb::init(
                // Constructor is wrapped in a lambda to wrap python recipe definition
                [](std::shared_ptr<arb::py::recipe>& r, const arb::domain_decomposition& d) {
                    return new arb::model(arb::py_recipe_shim(r), d);
              }),
              pb::call_guard<pb::gil_scoped_release>(),
         .def("run", &arb::model::run, pb::call_guard<pb::gil_scoped_release>());

.. _pybind11: http://pybind11.readthedocs.io
