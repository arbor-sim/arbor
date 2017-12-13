from libc.stdlib cimport malloc, free
from libcpp.vector cimport vector
from libcpp.string cimport string
from cpython cimport bool as cbool
from libc.stdint cimport uint32_t
from cpython cimport bool

####### _miniapp #################################
# C function defined in miniapp-base
# miniapp main -> callable function
cdef extern from "miniapp-base.hpp" namespace "arb::io":
    cdef:
        cppclass options:
            pass
        
        cppclass options_interface:
            options_interface() except +
            
            options get_options() except +
            void set_cargs(vector[string]) except +
            void set_cells(uint32_t) except +
            void set_synapses_per_cell(uint32_t) except +
            void set_syn_type(string) except +
            void set_compartments_per_segment(uint32_t) except +
            void set_morphologies(string) except +
            void set_morph_rr(cbool) except +
            void set_all_to_all(cbool) except +
            void set_ring(cbool) except +
            void set_tfinal(double) except +
            void set_dt(double) except +
            void set_bin_regular(cbool) except +
            void set_bin_dt(double) except +
            void set_sample_dt(double) except +
            void set_probe_soma_only(cbool) except +
            void set_probe_ratio(double) except +
            void set_trace_prefix(string) except +
            void set_trace_max_gid(unsigned) except +
            void set_trace_format(string) except +
            void set_spike_file_output(cbool) except +
            void set_single_file_per_rank(cbool) except +
            void set_over_write(cbool) except +
            void set_output_path(string) except +
            void set_file_name(string) except +
            void set_file_extension(string) except +
            void set_spike_file_input(cbool) except +
            void set_input_spike_path(string) except +
            void set_dry_run_ranks(int) except +
            void set_profile_only_zero(cbool) except +
            void set_report_compartments(cbool) except +
            void set_verbose(cbool) except +

        int miniapp(options) except +

######## Miniapp ###################################
# miniapp(arg1, ...) in python
#  calls the miniapp main function
#  with a argv[0] of "pyarbor"

cdef class Miniapp:
    cdef:
        public int ret
        options_interface opts

    def __cinit__(self, *args, **kwargs):
        self.opts = options_interface()

    def __init__(self, *args, **kwargs):
        self.ret = 0

    def set_cargs(self, list cargs):
        cdef vector[string] cargs_ = cargs
        self.opts.set_cargs(cargs_)

    def set_cells(self, int cells):
        self.opts.set_cells(cells)

    def set_synapses_per_cell(self, int synapses_per_cell):
        self.opts.set_synapses_per_cell(synapses_per_cell)

    def set_syn_type(self, str syn_type):
        self.opts.set_syn_type(syn_type)

    def set_compartments_per_segment(self,
                                     int compartments_per_segment):
        self.opts.set_compartments_per_segment(
            compartments_per_segment
        )

    def set_morphologies(self, str morphologies):
        self.opts.set_morphologies(morphologies)

    def set_morph_rr(self, bool morph_rr):
        self.opts.set_morph_rr(morph_rr)

    def set_all_to_all(self, bool all_to_all):
        self.opts.set_all_to_all(all_to_all)

    def set_ring(self, bool ring):
        self.opts.set_ring(ring)

    def set_tfinal(self, double tfinal):
        self.opts.set_tfinal(tfinal)

    def set_dt(self, double dt):
        self.opts.set_dt(dt)

    def set_bin_regular(self, bool bin_regular):
        self.opts.set_bin_regular(bin_regular)

    def set_bin_dt(self, double bin_dt):
        self.opts.set_bin_dt(bin_dt)

    def set_sample_dt(self, double sample_dt):
        self.opts.set_sample_dt(sample_dt)

    def set_probe_soma_only(self, bool probe_soma_only):
        self.opts.set_probe_soma_only(probe_soma_only)

    def set_probe_ratio(self, double probe_ratio):
        self.opts.set_probe_ratio(probe_ratio)

    def set_trace_prefix(self, string trace_prefix):
        self.opts.set_trace_prefix(trace_prefix)

    def set_trace_max_gid(self, unsigned trace_max_gid):
        self.opts.set_trace_max_gid(trace_max_gid)

    def set_trace_format(self, string trace_format):
        self.opts.set_trace_format(trace_format)

    def set_spike_file_output(self, bool spike_file_output):
        self.opts.set_spike_file_output(spike_file_output)

    def set_single_file_per_rank(self, bool single_file_per_rank):
        self.opts.set_single_file_per_rank(single_file_per_rank)

    def set_over_write(self, bool over_write):
        self.opts.set_over_write(over_write)

    def set_output_path(self, str output_path):
        self.opts.set_output_path(output_path)

    def set_file_name(self, str file_name):
        self.opts.set_file_name(file_name)

    def set_file_extension(self, str file_extension):
        self.opts.set_file_extension(file_extension)

    def set_spike_file_input(self, bool spike_file_input):
        self.opts.set_spike_file_input(spike_file_input)

    def set_input_spike_path(self, str input_spike_path):
        self.opts.set_input_spike_path(input_spike_path)

    def set_dry_run_ranks(self, int dry_run_ranks):
        self.opts.set_dry_run_ranks(dry_run_ranks)

    def set_profile_only_zero(self, bool profile_only_zero):
        self.opts.set_profile_only_zero(profile_only_zero)

    def set_report_compartments(self, bool report_compartments):
        self.opts.set_report_compartments(report_compartments)

    def set_verbose(self, bool verbose):
        self.opts.set_verbose(verbose)

    def run(self):
        self.ret = miniapp(self.opts.get_options())
        if self.ret:
            raise RuntimeError(
                "Miniapp returned non-zero {}"
                .format(self.ret)
            )
