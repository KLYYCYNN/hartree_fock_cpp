#include "scf.hpp"
#include "save.hpp"
#include "visual.hpp"

int main() {

    std::string param_path = "/home/jc6224/hf_cpp/basis";
    std::string save_path = "/home/jc6224/hf_cpp/data";
    std::string compute_ERI = "Hybrid";
    std::string basis_set = "cc-pVTZ";

    std::vector<std::pair<std::string, vector3>>
    config = {{"H", position_vector(0.0, 0.0, 0.7)},
              {"H", position_vector(0.0, 0.0, -0.7)}};

    int charge = 0;
    std::vector<double> initial_density = linspace(0.0, 0.0, 900);

    scf_data scf_output = scf_cycle(config, initial_density,
                                    basis_set, param_path,
                                    charge, compute_ERI);

    save_scf_data(save_path, "test3", scf_output);
}
