// running molecules with GPU computed ERI, benchmarking.

#include "scf.hpp"
#include "visual.hpp"
#include "save.hpp"

int main(){

    std::string param_path = "/home/jc6224/hf_cpp/basis";
    std::string save_path = "/home/jc6224/hf_cpp/data";
    std::string folder_name = "test25";


    std::vector<std::pair<std::string, vector3>> 
    config = {{"C", position_vector(0.0, 0.0, 0.0)},
              {"O", position_vector(2.196, 0.0, 0.0)},
              {"O", position_vector(-2.196, 0.0, 0.0)}};

    scf_data result = scf_cycle(config, "cc-pVTZ", param_path, 0, "GPU");

    save_scf_data(save_path, folder_name, result);
    return 0;
}