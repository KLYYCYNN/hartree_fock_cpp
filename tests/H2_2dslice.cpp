#include "cube.hpp"

int main(){

    std::string save_path = "/home/jc6224/hf/data/test1";
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

    square surface;
    surface.center = origin;
    surface.A = position_vector(0.0, 1.0, 0.0);
    surface.B = position_vector(0.0, 0.0, 1.0);
    surface.L = 3.0;

    rhf_2d_homo_lumo(save_path, "homo_lumo", basis, C_matrix, 2, surface, 1024);
}