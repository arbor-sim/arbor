.. _profiling:

Profiling Arbor
===============

Arbor has support for live profiling using `Tracy <https://github.com/wolfpld/>`_ by means of annotation in the source
code. By default, these macros expand to nothing to avoid overheads. For a complete picture, refer to the Tracy
documentation, but here are the central examples:

.. code-block:: C++

    void foo(const std::string& s) {
         // Will show a zone 'foo' in the profiler
         PROFILE_ZONE();
         bar(s);
    }


    void bar(const std::string& s) {
         // Will show a zone 'bar' in the profiler
         PROFILE_ZONE();
         // Will add a custom text to the zone.
         ANNOTATE_ZONE(s.data(), s.size());

         {
                // Will show a zone 'interesting' in the profiler _under_ zone 'bar'
                PROFILE_NAMED_ZONE("interesting");

         }

    }

    int main() {
        // Zones: foo/bar(via foo)/interesting
        foo("via foo");
        // Zones: bar(just bar)/interesting
        bar("just bar");
    }

To enable profiling `ARB_WITH_PROFILING` needs to set at build time. Then, all applications built on top of Arbor will
connect to the Tracy server and stream their results.
