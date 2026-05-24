#include "cube.hpp"
#include "visual.hpp"

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
    surface.B = position_vector(0.5, 0.0, 1.0);
    surface.L = 12.0;

    rhf_2d_homo_lumo(save_path, "homo_lumo1", basis, C_matrix, 10, surface, 1024);
}