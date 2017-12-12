from libc.stdlib cimport malloc, free
from libcpp.stdlib cimport vector, string
from cpython cimport bool

####### _miniapp #################################
# C function defined in miniapp-base
# miniapp main -> callable function
cdef extern from "miniapp-base.hpp":
    cdef:
        class Options:
            pass
        
        class OptionsInterface:
            Options get_options()
            void set_args(vector[string])
            void set_cells(uint32_t)
            void set_synapses_per_cell(uint32_t)
            void set_syn_type(string)
            void set_compartments_per_segment(uint32_t)
            void set_morphologies(std::string)
            void set_morph_rr(cbool)
            void set_all_to_all(cbool)
            void set_ring(cbool)
            void set_tfinal(double)
            void set_dt(double)
            void set_bin_regular(cbool)
            void set_bin_dt(double)
            void set_sample_dt(double)
            void set_probe_soma_only(cbool)
            void set_probe_ratio(double)
            void set_trace_prefix(string)
            void set_trace_max_gid(unsigned)
            void set_trace_format(string)
            void set_spike_file_output(cbool)
            void set_single_file_per_rank(cbool)
            void set_over_write(cbool)
            void set_output_path(string)
            void set_file_name(string)
            void set_file_extension(string)
            void set_spike_file_input(cbool)
            void set_input_spike_path;(string)
            void set_dry_run_ranks(int)
            void set_profile_only_zero(cbool)
            void set_report_compartments(cbool)
            void set_verbose(cbool)

        int miniapp(Options)

######## Miniapp ###################################
# miniapp(arg1, ...) in python
#  calls the miniapp main function
#  with a argv[0] of "pyarbor"

class Miniapp:
    cdef ret
    cdef OptionsInterface options

    def __cinit__(self, *args, **kwargs):
        self.options = new OptionsInterface()

    def __del__(self):
        delete self.options
    
    def __init__(self, *args, **kwargs):
        self.ret = 0

    def set_args(self, list args):
        self.options.set_args(<vector[string]> args)

    def set_cells(self, int cells):
        self.options.set_cells(cells)

    def set_synapses_per_cell(self, int synapses_per_cell):
        self.options.set_synapses_per_cell(synapses_per_cell)

    def set_syn_type(self, str syn_type):
        self.options.set_syn_type(syn_type)

    def set_compartments_per_segment(self,
                                     int compartments_per_segment):
        self.options.set_compartments_per_segment(
            compartments_per_segment
        )

    def set_morphologies(self, str morphologies):
        self.options.set_morphologies(morphologies)

    def set_morph_rr(self, bool morph_rr):
        self.options.set_morph_rr(morph_rr)

    def set_all_to_all(self, bool all_to_all):
        self.options.set_all_to_all(all_to_all)

    def set_ring(self, bool ring):
        self.options.set_ring(ring)

    def set_tfinal(self, double tfinal):
        self.options.set_tfinal(tfinal)

    def set_dt(self, double dt):
        self.options.set_dt(dt)

    def set_bin_regular(self, bool bin_regular):
        self.options.set_bin_regular(bin_regular)

    def set_bin_dt(self, double bin_dt):
        self.options.set_bin_dt(bin_dt)

    def set_sample_dt(self, double sample_dt):
        self.options.set_sample_dt(sample_dt)

    def set_probe_soma_only(self, bool probe_soma_only):
        self.options.set_probe_soma_only(probe_soma_only)

    def set_probe_ratio(self, double probe_ratio):
        self.options.set_probe_ratio(probe_ratio)

    def set_trace_prefix(self, string trace_prefix):
        self.options.set_trace_prefix(trace_prefix)

    def set_trace_max_gid(self, unsigned trace_max_gid):
        self.options.set_trace_max_gid(trace_max_gid)

    def set_trace_format(self, string trace_format):
        self.options.set_trace_format(trace_format)

    def set_spike_file_output(self, bool spike_file_output):
        self.options.set_spike_file_output(spike_file_output)

    def set_single_file_per_rank(self, bool single_file_per_rank):
        self.options.set_single_file_per_rank(single_file_per_rank)

    def set_over_write(self, bool over_write):
        self.options.set_over_write(over_write)

    def set_output_path(self, str output_path):
        self.options.set_output_path(output_path)

    def set_file_name(self, str file_name):
        self.options.set_file_name(file_name)

    def set_file_extension(self, str file_extension):
        self.options.set_file_extension(file_extension)

    def set_spike_file_input(self, bool spike_file_input):
        self.options.set_spike_file_input(spike_file_input)

    def set_input_spike_path;(self, str input_spike_path;):
        self.options.set_input_spike_path;(input_spike_path;)

    def set_dry_run_ranks(self, int dry_run_ranks):
        self.options.set_dry_run_ranks(dry_run_ranks)

    def set_profile_only_zero(self, bool profile_only_zero):
        self.options.set_profile_only_zero(profile_only_zero)

    def set_report_compartments(self, bool report_compartments):
        self.options.set_report_compartments(report_compartments)

    def set_verbose(self, bool verbose):
        self.options.set_verbose(verbose)

    def run(self):
        self.ret = miniapp(self.options.get_options())
        if self.ret:
            raise RuntimeError(
                "Miniapp returned non-zero {}"
                .format(r)
            )
