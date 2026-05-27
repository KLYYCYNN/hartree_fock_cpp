// running molecules with GPU computed ERI, benchmarking.

#include "scf.hpp"
#include "visual.hpp"
#include "save.hpp"

int main(){

    std::string param_path = "/home/jc6224/hf_cpp/basis";
    std::string save_path = "/home/jc6224/hf_cpp/data";
    std::string folder_name = "test13";


    std::vector<std::pair<std::string, vector3>> 
    config = {{"S", position_vector(0.0, 0.0, 0.0)},
              {"O", polar3d(2.706, M_PI / 2, deg2rad(119.5 / 2))},
              {"O", polar3d(2.706, M_PI / 2, -deg2rad(119.5 / 2))}};

    scf_data result = scf_cycle(config, "cc-pVTZ", param_path, 0, "CPU");

    save_scf_data(save_path, folder_name, result);
    return 0;
}