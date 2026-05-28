#include "scf.hpp"
#include "visual.hpp"
#include "save.hpp"

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " input.json\n";
        return 1;
    }

    std::ifstream file(argv[1]);
    if (!file.is_open()) {
        std::cerr << "Could not open input file: " << argv[1] << "\n";
        return 1;
    }

    json input;
    file >> input;

    std::string param_path = input.at("basis_path").get<std::string>();
    std::string save_path = input.at("save_path").get<std::string>();
    std::string folder_name = input.at("folder_name").get<std::string>();
    std::string basis = input.at("basis").get<std::string>();
    std::string eri_hardware = input.at("eri_hardware").get<std::string>();

    int charge = input.value("charge", 0);

    std::vector<std::pair<std::string, vector3>> config;

    for (const auto& atom : input.at("atoms")) {
        std::string symbol = atom.at("symbol").get<std::string>();
        std::string coord_type = atom.value("coordinate_type", "cartesian");

        vector3 pos;

        if (coord_type == "cartesian") {
            const auto& r = atom.at("position");

            pos = position_vector(
                r.at(0).get<double>(),
                r.at(1).get<double>(),
                r.at(2).get<double>()
            );
        }
        else if (coord_type == "polar") {
            const auto& r = atom.at("position");

            pos = polar3d(
                r.at(0).get<double>(),  // radius
                deg2rad(r.at(1).get<double>()),  // theta
                deg2rad(r.at(2).get<double>() )  // phi
            );
        }
        else {
            throw std::runtime_error(
                "Unknown coordinate_type: " + coord_type
            );
        }

        config.push_back({symbol, pos});
    }

    scf_data result = scf_cycle(
        config,
        basis,
        param_path,
        charge,
        eri_hardware
    );

    save_scf_data(save_path, folder_name, result);

    return 0;
}