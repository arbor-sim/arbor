.. _dev-version:

Version and build information
=============================

The Arbor library records version and configuration information in
two ways:

* The ``version.hpp`` header has preprocessor defines with the prefix ``ARB_``.

* The library presents this information in variables within the ``arb::`` namespace.

Version information
-------------------

The Arbor version string is in the format MAJOR.MINOR.PATCH,
or for development versions MAJOR.MINOR.PATCH-DEV, where DEV
is a string, usually literally "dev".

:c:macro:`ARB_VERSION`
    Full Arbor version string. Available as :cpp:var:`arb::version`.

:c:macro:`ARB_VERSION_MAJOR`
    Major version number. Available as :cpp:var:`arb::version_major`.

:c:macro:`ARB_VERSION_MINOR`
    Major version number. Available as :cpp:var:`arb::version_minor`.

:c:macro:`ARB_VERSION_PATCH`
    Major version number. Available as :cpp:var:`arb::version_patch`.

:c:macro:`ARB_VERSION_DEV`
    Development version suffix string. Only defined if Arbor is a development version.
    Available as :cpp:var:`arb::version_dev`, which will be an empty string
    if :c:macro:`ARB_VERSION_DEV` is not defined.

Source information
------------------

:c:macro:`ARB_SOURCE_ID`
   The source id contains the git commit time stamp, the commit hash,
   and if there are uncommitted changes in the source tree, a suffix "modified",
   e.g. ``"2020-01-02T03:04:05+06:00 b1946ac92492d2347c6235b4d2611184 modified"``.
   Available as :cpp:var:`arb::source_id`.

Build information
-----------------

Arbor can be built in the default 'Release' configuration, or in an unoptimized
'Debug' configuration that is useful for development. Additionally, it can be
built for a particular CPU architecture given by the ``ARB_ARCH`` CMake configuration
variable.

:c:macro:`ARB_BUILD_CONFIG`
    Configuration string, all uppercase. Will be ``"DEBUG"`` or ``"RELEASE"``.
    Available as :cpp:var:`arb::build_config`.

:c:macro:`ARB_ARCH`
    Value of the ``ARB_ARCH`` configuration variable, e.g. ``"native"``.
    Available as :cpp:var:`arb::arch`.

Features
--------

Configuration-time features are enabled in Arbor via CMake configuration variables
such as ``ARB_WITH_MPI`` and ``ARB_WITH_PYTHON``. Each enabled feature has
a corresponding preprocessor symbol in ``version.hpp`` of the form ``ARB_FEATURENAME_ENABLED``.
Examples include :c:macro:`ARB_MPI_ENABLED`, :c:macro:`ARB_ASSERT_ENABLED`.

Full build information
----------------------

A single string containing all the identification information for an Arbor build
is available in the macro :c:macro:`ARB_FULL_BUILD_ID` and in the variable
:cpp:var:`arb::full_build_id`. This string contains the source id, the full version,
the build configuration, the target architecture, and a list of enabled features.

