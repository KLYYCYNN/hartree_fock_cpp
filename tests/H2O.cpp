#include "scf.hpp"
#include "save.hpp"
#include "visual.hpp"

int main() {

    std::string param_path = "/home/jc6224/hf/basis_sets";
    std::string save_path = "/home/jc6224/hf/data";
    std::string compute_ERI = "CPU";
    std::string basis_set = "cc-pVTZ";

    std::vector<std::pair<std::string, vector3>>
    config = {{"H", polar3d(1.81, M_PI / 2, deg2rad(104.5 / 2))},
              {"H", polar3d(1.81, M_PI / 2, -deg2rad(104.5 / 2))},
               {"O", position_vector(0.0, 0.0, 0.0)}};

    int charge = 0;
    std::vector<double> initial_density = linspace(0.0, 0.0, 65 * 65);

    scf_data scf_output = scf_cycle(config, initial_density,
                                    basis_set, param_path,
                                    charge, compute_ERI);

    save_scf_data(save_path, "H2O", scf_output);
}