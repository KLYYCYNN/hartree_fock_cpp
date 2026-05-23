#include "cube.hpp"

int main(){

    std::string save_path = "/home/jc6224/hf/data/H2O";
    std::string basis_file = save_path + "/basis.csv";
    std::string param_path = "/home/jc6224/hf/basis_sets";

    std::vector<basis_function> basis 
    = basis_recon(basis_file, param_path,"cc-pVTZ");
    
    rapidcsv::Document density_file(save_path + "/density.csv");
    rapidcsv::Document coeff_file(save_path + "/coefficient.csv");

    std::vector<double> P_matrix = density_file.GetColumn
                                   <double>("density_matrix");

    std::vector<double> C_matrix = coeff_file.GetColumn
                                   <double>("coefficient_matrix");

    vector3 origin = position_vector(0.0, 0.0, 0.0);
    cube_data rho_r = density3d(basis, P_matrix, position_vector(0.5, 0.0, 0.0), 3.0, 500);
    //std::pair<cube_data, cube_data> homo_lumo
    //= rhf_3d_homo_lumo(basis, C_matrix, 10, position_vector(0.5, 0.0, 0.0), 3.0, 500);

    std::vector<atom> molecule = recreate_system(save_path + "/geometry.csv");
    
    write_cube_file(save_path + "/density.cube", rho_r, molecule);
    //write_cube_file(save_path + "/lumo.cube", homo_lumo.second, molecule);


}