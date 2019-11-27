struct local_param_set_proxy {
    arb::cable_cell_local_parameter_set params;
    void set_membrane_potential(pybind11::object value) {
        params.init_membrane_potential =
            py2optional<double>(value, "membrane potential must be a number.", is_nonneg{});
    }
    void set_temperature(pybind11::object value) {
        params.temperature_K =
            py2optional<double>(value, "temperature in degrees K must be non-negative.", is_nonneg{});
    }
    void set_axial_resistivity(pybind11::object value) {
        params.axial_resistivity =
            py2optional<double>(value, "axial resistivity must be positive.", is_positive{});
    }
    void set_membrane_capacitance(pybind11::object value) {
        params.membrane_capacitance =
            py2optional<double>(value, "membrane capacitance must be positive.", is_positive{});
    }

    auto get_membrane_potential()   const { return params.init_membrane_potential; }
    auto get_temperature()          const { return params.temperature_K; }
    auto get_axial_resistivity()    const { return params.axial_resistivity; }
    auto get_membrane_capacitance() const { return params.axial_resistivity; }

    operator arb::cable_cell_local_parameter_set() const {
        return params;
    }
};

std::string local_parameter_set_str(const local_param_set_proxy& p) {
    auto s = util::pprintf("<arbor.local_parameter_set: V_m {} (mV), temp {} (K), R_L {} (Ω·cm), C_m {} (F/m²), ion_data {}>",
            p.params.init_membrane_potential, p.params.temperature_K,
            p.params.axial_resistivity, p.params.membrane_capacitance,
            util::dictionary_csv(p.params.ion_data));
    return s;
}


    // arb::cable_cell_ion_data
    pybind11::class_<arb::cable_cell_ion_data> ion_data(m, "ion_data");
    ion_data
        .def(pybind11::init(
            [](double ic, double ec, double rp) {
                return arb::cable_cell_ion_data{ic, ec, rp};
            }), "intern_con"_a, "extern_con"_a, "rev_pot"_a)
        .def_readonly("intern_con",
            &arb::cable_cell_ion_data::init_int_concentration,
            "Initial internal concentration of ion species.")
        .def_readonly("extern_con",
            &arb::cable_cell_ion_data::init_ext_concentration,
            "Initial external concentration of ion species.")
        .def_readonly("rev_pot",
            &arb::cable_cell_ion_data::init_reversal_potential,
            "Initial reversal potential of ion species.")
        .def("__repr__", &ion_data_str)
        .def("__str__",  &ion_data_str);


    arb::cable_cell_local_parameter_set

    pybind11::class_<local_param_set_proxy> local_cable_params(m, "local_parameter_set");
    local_cable_params
        .def(pybind11::init<>())
        .def_property("temperature_K",
            &local_param_set_proxy::get_temperature,
            &local_param_set_proxy::set_temperature,
            "Temperature in degrees Kelvin.")
        .def_property("axial_resistivity",
            &local_param_set_proxy::get_axial_resistivity,
            &local_param_set_proxy::set_axial_resistivity,
            "Axial resistivity in Ω·cm.")
        .def_property("init_membrane_potential",
            &local_param_set_proxy::get_membrane_potential,
            &local_param_set_proxy::set_membrane_potential,
            "Initial membrane potential in mV.")
        .def_property("membrane_capacitance",
            &local_param_set_proxy::get_membrane_capacitance,
            &local_param_set_proxy::set_membrane_capacitance,
            "Membrane capacitance in F/m².")
        .def_property_readonly("ion_data",
                [](const local_param_set_proxy& p) {
                    return p.params.ion_data;
                })
        .def("set_ion",
                [](local_param_set_proxy& p, std::string ion, arb::cable_cell_ion_data data) {
                    p.params.ion_data[std::move(ion)] = data;
                },
            "name"_a, "props"_a,
            "Set properties of an ion species with name.")
        .def("__repr__", &local_parameter_set_str)
        .def("__str__",  &local_parameter_set_str);

    //pybind11::class_<local_param_set_proxy> cable_params(m, "local_parameter_set");

    // in cable_cell wrapper
    .def("paint",
        [](arb::cable_cell& c, const char* region, const local_param_set_proxy& p) {
            c.paint(region, (arb::cable_cell_local_parameter_set)p);
        },
        "region"_a, "mechanism"_a,
        "Associate a set of properties with a region. These properties will override the the global or cell-wide default values on the specific region.")
