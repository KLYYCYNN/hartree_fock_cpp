#include "scf.hpp"
#include "save.hpp"

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

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


void require_key(const json& input, const std::string& key)
{
    if (!input.contains(key)) {
        throw std::runtime_error("Missing required JSON field: " + key);
    }
}



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


int main(int argc, char** argv)
{
    try {
        if (argc < 2) {
            std::cerr << "Usage: " << argv[0] << " input.json\n";
            return 1;
        }

        scf_options opt = read_scf_options(argv[1]);

        run_scf_from_options(opt);

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}