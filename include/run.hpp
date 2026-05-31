
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include "save.hpp"
#include "scf.hpp"

using json = nlohmann::json;


//---------------------------------------------------------
struct scf_options {
    std::string method = "RHF";

    std::string basis;
    std::string basis_path;
    std::string save_path;
    std::string folder_name = "test";

    int charge = 0;

    int n_alpha = -1;
    int n_beta = -1;

    std::string eri_device = "CPU";

    int max_it = 10000;
    double epsilon = 1e-6;
    double damping = 0.3;
    double screening_threshold = 1e-8;

    bool set_initial_density = false;

    std::vector<std::pair<std::string, vector3>> config;
};


struct bl_scan_options {

    std::string basis;
    std::string basis_path;
    std::string save_path;
    std::string folder_name = "test";

    int charge = 0;

    int n_alpha = -1;
    int n_beta = -1;

    std::string eri_device = "CPU";

    int max_it = 10000;
    double epsilon = 1e-6;
    double damping = 0.3;
    double screening_threshold = 1e-8;

    bool set_initial_density = false;

    std::string central_atom;

    // symbol, {theta, phi}
    std::vector<std::pair<std::string, std::pair<double, double>>> config;

    // {start, stop, n_points}
    std::vector<std::array<double, 3>> distances;
};

//---------------------------------------------------------


std::vector<double> linspace(double start, double stop, int npoints){
    double step = (stop - start) / (npoints - 1);
    std::vector<double> arrray;
    for (int i = 0; i < npoints; ++i)
    {
        arrray.emplace_back(start + i*step);
    }

    return arrray;
}
//---------------------------------------------------------

void require_key(const json& input, const std::string& key)
{
    if (!input.contains(key)) {
        throw std::runtime_error("Missing required JSON field: " + key);
    }
}

//---------------------------------------------------------

scf_options read_scf_options(const std::string& json_file)
{
    std::ifstream file(json_file);

    if (!file.is_open()) {
        throw std::runtime_error("Could not open JSON file: " + json_file);
    }

    json input;
    file >> input;

    require_key(input, "basis_path");
    require_key(input, "save_path");
    require_key(input, "basis");
    require_key(input, "atoms");

    scf_options opt;

    opt.method = input.value("method", "RHF");

    opt.basis = input.at("basis").get<std::string>();
    opt.basis_path = input.at("basis_path").get<std::string>();
    opt.save_path = input.at("save_path").get<std::string>();

    opt.folder_name = input.value("folder_name", "test");
    opt.charge = input.value("charge", 0);
    opt.eri_device = input.value("eri_hardware", "CPU");

    opt.max_it = input.value("max_iterations", 10000);
    opt.epsilon = input.value("epsilon", 1e-6);
    opt.damping = input.value("damping", 0.3);
    opt.screening_threshold = input.value("screening_threshold", 1e-8);

    if (opt.method == "UHF") {
        require_key(input, "spin_occupancy");

        opt.n_alpha = input.at("spin_occupancy").at("alpha").get<int>();
        opt.n_beta  = input.at("spin_occupancy").at("beta").get<int>();
    }

    for (const auto& atom : input.at("atoms")) {
        std::string symbol = atom.at("symbol").get<std::string>();
        std::string coord_type = atom.value("coordinate_type", "cartesian");

        const auto& r = atom.at("position");

        vector3 pos;

        if (coord_type == "cartesian") {
            pos = position_vector(
                r.at(0).get<double>(),
                r.at(1).get<double>(),
                r.at(2).get<double>()
            );
        }
        else if (coord_type == "polar") {
            pos = polar3d(
                r.at(0).get<double>(),
                deg2rad(r.at(1).get<double>()),
                deg2rad(r.at(2).get<double>())
            );
        }
        else {
            throw std::runtime_error("Unknown coordinate_type: " + coord_type);
        }

        opt.config.push_back({symbol, pos});
    }

    return opt;
}


bl_scan_options read_blscan_options(const std::string& json_file)
{
    std::ifstream file(json_file);

    if (!file.is_open()) {
        throw std::runtime_error("Could not open JSON file: " + json_file);
    }

    json input;
    file >> input;

    require_key(input, "basis_path");
    require_key(input, "save_path");
    require_key(input, "basis");
    require_key(input, "atoms");
    require_key(input, "central_atom");
    require_key(input, "distances");

    bl_scan_options opt;

    opt.basis = input.at("basis").get<std::string>();
    opt.basis_path = input.at("basis_path").get<std::string>();
    opt.save_path = input.at("save_path").get<std::string>();
    opt.central_atom = input.at("central_atom").get<std::string>();

    opt.folder_name = input.value("folder_name", "test");
    opt.charge = input.value("charge", 0);
    opt.eri_device = input.value("eri_hardware", "CPU");

    opt.max_it = input.value("max_iterations", 10000);
    opt.epsilon = input.value("epsilon", 1e-6);
    opt.damping = input.value("damping", 0.3);
    opt.screening_threshold = input.value("screening_threshold", 1e-8);


    for (const auto& atom : input.at("atoms")) {
        std::string symbol = atom.at("symbol").get<std::string>();

        require_key(atom, "angular_position");
        const auto& r = atom.at("angular_position");

        if (!r.is_array() || r.size() != 2) {
            throw std::runtime_error(
                "angular_position must be [theta, phi]"
            );
        }

        double theta = r.at(0).get<double>();
        double phi = r.at(1).get<double>();

        opt.config.push_back({symbol, {theta, phi}});
    }

    for (const auto& d : input.at("distances")) {
        if (!d.is_array() || d.size() != 3) {
            throw std::runtime_error(
                "each distance range must be [start, stop, n_points]"
            );
        }

        double start = d.at(0).get<double>();
        double stop = d.at(1).get<double>();
        int n_points = d.at(2).get<int>();

        if (n_points < 2) {
            throw std::runtime_error(
                "distance range n_points must be at least 2"
            );
        }

        if (stop <= start) {
            throw std::runtime_error(
                "distance range stop must be larger than start"
            );
        }

        opt.distances.push_back({
            start,
            stop,
            static_cast<double>(n_points)
        });
    }

    return opt;
}
//---------------------------------------------------------



//---------------------------------------------------------

void run_scf_from_options(const scf_options& opt)
{
    if (opt.method == "RHF") {

        rhf_data result = rhf_solver(
            opt.config,
            opt.basis,
            opt.basis_path,
            opt.charge,
            opt.eri_device,
            false,
            {},
            opt.max_it,
            opt.epsilon,
            opt.damping,
            opt.screening_threshold
        );

        save_rhf_data(opt.save_path, opt.folder_name, result);
    }
    else if (opt.method == "UHF") {

        uhf_data result = uhf_solver(
            opt.config,
            opt.basis,
            opt.basis_path,
            {opt.n_alpha, opt.n_beta},
            opt.charge,
            opt.eri_device,
            false,
            {},
            {},
            opt.max_it,
            opt.epsilon,
            opt.damping,
            opt.screening_threshold
        );

        save_uhf_data(opt.save_path, opt.folder_name, result);
    }
    else {
        throw std::runtime_error("Unknown SCF method: " + opt.method);
    }
}
//---------------------------------------------------------

//---------------------------------------------------------

void run_bl_scan(const bl_scan_options& opt)
{
    auto t0 = std::chrono::high_resolution_clock::now();

    std::vector<double> distances;

    for (const auto& range : opt.distances) {
        double start = range[0];
        double stop  = range[1];
        int n = static_cast<int>(range[2]);

        std::vector<double> segment = linspace(start, stop, n);

        if (!distances.empty() && !segment.empty()) {
            segment.erase(segment.begin());
        }

        distances.insert(distances.end(), segment.begin(), segment.end());
    }

    int n_points = static_cast<int>(distances.size());
    int n_satellite_atoms = static_cast<int>(opt.config.size());

    std::vector<double> E_tot(n_points);
    std::vector<double> E_homo(n_points);
    std::vector<double> E_lumo(n_points);
    std::vector<double> iterations(n_points);

    int K = -1;
    std::vector<double> prev_density = {};
    bool have_prev_density = false;

    for (int i = 0; i < n_points; ++i) {

        std::vector<std::pair<std::string, vector3>> config_i;
        config_i.push_back({
            opt.central_atom,
            position_vector(0.0, 0.0, 0.0)
        });

        for (int j = 0; j < n_satellite_atoms; ++j) {
            std::string symbol = opt.config[j].first;

            double theta = deg2rad(opt.config[j].second.first);
            double phi   = deg2rad(opt.config[j].second.second);

            vector3 pos = polar3d(distances[i], theta, phi);

            config_i.push_back({symbol, pos});
        }



        rhf_data result = rhf_solver(
            config_i,
            opt.basis,
            opt.basis_path,
            opt.charge,
            opt.eri_device,
            have_prev_density,
            prev_density,
            opt.max_it,
            opt.epsilon,
            opt.damping,
            opt.screening_threshold
        );

        prev_density = result.density;
        have_prev_density = true;

        int homo = result.Ne / 2 - 1;
        int lumo = result.Ne / 2;

        if (homo < 0 ||
            lumo >= static_cast<int>(result.mo_energy.size())) {
            throw std::runtime_error("Invalid RHF HOMO/LUMO index");
        }

        E_tot[i] = result.E_tot;
        E_homo[i] = result.mo_energy[homo];
        E_lumo[i] = result.mo_energy[lumo];
        iterations[i] = static_cast<double>(result.iterations);
        K = result.K;


        std::cout << i + 1 << "/" << n_points 
        << " data points collect" << std::endl;
    }


    std::vector<std::string> elements;
    elements.push_back(opt.central_atom);

    for (const auto& atom : opt.config) {
        elements.push_back(atom.first);
    }

    auto t1 = std::chrono::high_resolution_clock::now();

    blscan_data data;

    data.distances = distances;
    data.energies = E_tot;

    data.homo_energies = E_homo;
    data.lumo_energies = E_lumo;

    data.iterations = iterations;

    data.basis = opt.basis;
    data.system = formatElementCounts(elements);

    data.n_points = n_points;
    data.K = K;

    data.min_d = distances.front();
    data.max_d = distances.back();

    data.start = formatTimeC20(t0);
    data.end = formatTimeC20(t1);
    data.duration = std::chrono::duration<double>(t1 - t0).count();

    save_blscan(data, opt.save_path, opt.folder_name);
}

//oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
//oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo
//oooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo

void run_dissociation(const scf_options& opt){

    auto t0 = std::chrono::high_resolution_clock::now();

    if (opt.method != "RHF"){
        throw std::runtime_error(R"(dissociation energy only support
                                 RHF molecules)");
    }

    rhf_data molecule_result = rhf_solver(
        opt.config,
        opt.basis,
        opt.basis_path,
        opt.charge,
        opt.eri_device,
        false,
        {},
        opt.max_it,
        opt.epsilon,
        opt.damping,
        opt.screening_threshold
    );

    disstn_data output;
    output.BasisSet = molecule_result.BasisSet;
    output.K = molecule_result.K;
    output.charge = molecule_result.charge;
    output.Ne = molecule_result.Ne;
    output.coefficient = molecule_result.coefficient;
    output.density = molecule_result.density;
    output.hardware = molecule_result.hardware;
    output.mo_energy = molecule_result.mo_energy;
    output.basis = molecule_result.basis;
    output.energy = molecule_result.E_tot;
    output.system = molecule_result.system;

    std::vector<std::string> atom_list;
    for (int i = 0; i < opt.config.size(); ++i){
        atom_list.emplace_back(opt.config[i].first);
    }

    std::pair<std::vector<std::string>, std::vector<int>>
    atoms_info = countElements(atom_list);

    std::vector<double> atom_energies;
    double total_atom_energy = 0.0;

    for (int i = 0; i < atoms_info.first.size(); ++i){

        std::vector<std::pair<std::string, vector3>>
        single_atom = {{atoms_info.first[i], 
                        position_vector(0.0, 0.0, 0.0)}};

        uhf_data atom_result
        = uhf_solver(single_atom, opt.basis, opt.basis_path, 
                     atom_spin_occupancy(atoms_info.first[i]),
                     0, opt.eri_device, false, {}, {},
                     opt.max_it, opt.epsilon, opt.damping,
                     opt.screening_threshold);

        atom_energies.emplace_back(atom_result.E_tot);
        total_atom_energy += atoms_info.second[i]
                               * atom_result.E_tot;
    }

    auto tf = std::chrono::high_resolution_clock::now();
    double dt = std::chrono::duration<double>(tf - t0).count();

    double E_d = total_atom_energy - molecule_result.E_tot;
    
    output.disstn_energy = E_d;
    output.energies = atom_energies;
    output.atoms = atoms_info.first;
    output.atom_numbers = atoms_info.second;
    output.start = formatTimeC20(t0);
    output.end = formatTimeC20(tf);
    output.duration = dt;

    save_disstn_data(output, opt.save_path, opt.folder_name);
}
