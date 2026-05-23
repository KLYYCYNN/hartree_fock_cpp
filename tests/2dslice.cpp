#include "cube.hpp"

int main(){

    std::string save_path = "/home/jc6224/hf_cpp/data/H2O";
    std::string basis_file = save_path + "/basis.csv";
    std::string param_path = "/home/jc6224/hf_cpp/basis";

    std::vector<basis_function> basis 
    = basis_recon(basis_file, param_path,"cc-pVTZ");
    
    rapidcsv::Document density_file(save_path + "/density.csv");
    rapidcsv::Document coeff_file(save_path + "/coefficient.csv");

    std::vector<double> P_matrix = density_file.GetColumn
                                   <double>("density_matrix");

    std::vector<double> C_matrix = coeff_file.GetColumn
                                   <double>("coefficient_matrix");

    square surface;
    surface.center = position_vector(0.5, 0.0, 0.0);
    surface.A = position_vector(0.5, 1.0, 0.0);
    surface.B = position_vector(1.0, 0.0, 0.0);
    surface.L = 7.0;

    density2d(save_path, "density_slice", basis, P_matrix, surface, 1024);
}