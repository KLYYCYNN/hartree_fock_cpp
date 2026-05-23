#pragma once
#include "atoms.hpp"
#include <vector>
#include <string>
#include <algorithm>
#include <iterator>
#include <cmath>
#include <numbers>
#include <iostream>
#include <stdexcept>
#include <filesystem>


inline
int double_factorial(int n){
    if (n <= 0) return 1;

    int result = 1;

    for (int k = n; k > 0; k -= 2){
        result *= k;
    }

    return result;
}

//normalization of gaussian functions with angular momentum terms
inline
double cart_norm(double alpha,
                 int lx,
                 int ly,
                 int lz){

    int L = lx + ly + lz;

    double numerator =
        std::pow(4.0 * alpha, L);

    double denominator =
        double_factorial(2 * lx - 1) *
        double_factorial(2 * ly - 1) *
        double_factorial(2 * lz - 1);

    double prefactor =
        std::pow(2.0 * alpha / M_PI, 0.75);

    return prefactor *
           std::sqrt(numerator / denominator);
}


inline
std::vector<std::vector<int>> angmom_partition(int l) {
    std::vector<std::vector<int>> combinations;
    if (l < 0) return combinations; // Return empty if l is negative

    for (int a = 0; a <= l; ++a) {
        for (int b = 0; b <= l - a; ++b) {
            int c = l - a - b;
            combinations.push_back({a, b, c});
        }
    }
    return combinations;
}


inline
std::vector<basis_function> build_shell(const std::vector<double>& exponents,
                                          const std::vector<double>& coeffs,
                                          const vector3& centre, int l){
    //check the input data dimensions
    

    
    std::vector<std::vector<int>> angmom = angmom_partition(l);
    std::vector<basis_function> shell;
    std::vector<primitive_gaussian> primitives;

    for (int i = 0; i < exponents.size(); ++ i){
        primitive_gaussian prim_i;
        prim_i.alpha = exponents[i];
        prim_i.coeff = coeffs[i];
        primitives.emplace_back(prim_i);
    }


    for (int i = 0; i < angmom.size(); ++ i){

        basis_function func_i;
        std::vector<int> angmom_i = angmom[i];
        std::vector<primitive_gaussian> primitives_i(primitives.size());

        func_i.lx = angmom_i[0];
        func_i.ly = angmom_i[1];
        func_i.lz = angmom_i[2];
        func_i.center = centre;

        for (int j = 0; j < primitives.size(); ++ j){

            primitive_gaussian primitive_j = primitives[j];
            primitive_j.norm = cart_norm(primitive_j.alpha,
                                         angmom_i[0],
                                         angmom_i[1],
                                         angmom_i[2]);

            primitives_i[j] = primitive_j;
        }

        func_i.prims = primitives_i;
        shell.emplace_back(func_i);
    }
    return shell;
}


inline
std::vector<std::pair<int, int>> section_idx(const std::vector<int>& vec) {
    std::vector<std::pair<int, int>> bounds;
    if (vec.empty()) return bounds;

    int start_idx = 0;

    for (int i = 1; i < vec.size(); ++i) {
        if (vec[i] != vec[start_idx]) {
            bounds.push_back({start_idx, i - 1});
            start_idx = i; 
        }
    }
    bounds.push_back({start_idx, static_cast<int>(vec.size() - 1)});

    return bounds;
}


inline
void validate_file_exists(const std::string& file_path) {
    //Check if the path exists AND points to an actual file (not a folder)
    if (!std::filesystem::exists(file_path) || 
        !std::filesystem::is_regular_file(file_path)) {
        throw std::runtime_error(
        "Critical Error: File does not exist or is invalid at path: "
         + file_path);
    }
}

inline
void validate_folder_exists(const std::string& folder_path) {
    // Check if the path exists AND is specifically a directory/folder
    if (!std::filesystem::exists(folder_path) || 
        !std::filesystem::is_directory(folder_path)) {
        throw std::runtime_error
        ("Critical Error: Folder does not exist or is invalid at path: " 
        + folder_path);
    }
}

inline
std::pair<std::vector<basis_function>, basis_info>
basis_construction(const std::vector<std::pair<std::string, vector3>>& config,
                   std::string basis_set, std::string param_path)

{
    std::vector<atom> molecule = configuration(config);
    std::vector<basis_function> basis;
    basis_info information;
    
    std::vector<int> index_list;
    std::vector<int> shell_list;
    std::vector<int> lx_list;
    std::vector<int> ly_list;
    std::vector<int> lz_list;
    std::vector<double> cx_list;
    std::vector<double> cy_list;
    std::vector<double> cz_list;
    std::vector<std::string> l_list;
    std::vector<std::string> atom_list;

    std::vector<std::string> angmom_symbols = {"S", "P", "D", "F", "G", "H"};

    for (int i = 0; i < molecule.size(); ++ i){

        std::string atom_name = Zname(molecule[i].Z);
        std::string fileName = param_path + "/" + basis_set + "/" +
        atom_name + ".csv";
        
        validate_file_exists(fileName);
        rapidcsv::Document parameters(fileName);

        std::vector<double> exponents = parameters.GetColumn<double>("exponent");
        std::vector<double> coefficients = parameters.GetColumn<double>("coefficient");
        std::vector<int> shell_id = parameters.GetColumn<int>("shell_id");
        std::vector<int> primitive_id = parameters.GetColumn<int>("primitive_id");
        std::vector<std::string> angmom_levels = parameters.GetColumn<std::string>("L");

        std::vector<std::pair<int, int>> slice_idx = section_idx(shell_id);
        

        for (int j = 0; j < slice_idx.size(); ++j){
            std::vector<double> shell_exp = 
            std::vector<double>(exponents.begin() + slice_idx[j].first, 
                                exponents.begin() + slice_idx[j].second + 1);

            std::vector<double> shell_coeffs = 
            std::vector<double>(coefficients.begin() + slice_idx[j].first, 
                                coefficients.begin() + slice_idx[j].second + 1);

            std::vector<int> shell_prim_id = 
            std::vector<int>(primitive_id.begin() + slice_idx[j].first, 
                             primitive_id.begin() + slice_idx[j].second + 1);
            
            int n_prims = *std::max_element(shell_prim_id.begin(),
                                            shell_prim_id.end()) + 1;


            if (n_prims != shell_exp.size()){
                throw std::runtime_error("wrong primitive number");
            }

            int current_shell = shell_id[slice_idx[j].first];
            std::string current_l = angmom_levels[slice_idx[j].first];

            if (current_shell != j){
                throw std::runtime_error
                ("building shells in the wrong order");
            }
            

            int L = getIndex(angmom_symbols, current_l);
            std::vector<basis_function> functions =
            build_shell(shell_exp, shell_coeffs, molecule[i].R, L);

            basis.insert(basis.end(), functions.begin(), functions.end());

            for (int k = 0; k < functions.size(); ++ k){

                index_list.emplace_back(index_list.size());
                shell_list.emplace_back(current_shell);
                lx_list.emplace_back(functions[k].lx);
                ly_list.emplace_back(functions[k].ly);
                lz_list.emplace_back(functions[k].lz);
                cx_list.emplace_back(functions[k].center.x);
                cy_list.emplace_back(functions[k].center.y);
                cz_list.emplace_back(functions[k].center.z);
                l_list.emplace_back(current_l);
                atom_list.emplace_back(atom_name);
            }
        }
    }

    information.index = index_list;
    information.atoms = atom_list;
    information.shells = shell_list;
    information.types = l_list;
    information.cx = cx_list;
    information.cy = cy_list;
    information.cz = cz_list;
    information.lx = lx_list;
    information.ly = ly_list;
    information.lz = lz_list;

    return {basis, information};
}

inline
basis_function build_function(const std::vector<double>& exp,
                              const std::vector<double>& coeffs,
                              int l, int m, int n, vector3 center)
{
    if (exp.size() != coeffs.size()){
        throw std::runtime_error
        ("different numbers of coefficients and expoenets");
    }

    basis_function function;
    std::vector<primitive_gaussian> primitives;

    for (int i = 0; i < exp.size(); ++ i){

        primitive_gaussian prim_i;
        prim_i.coeff = coeffs[i];
        prim_i.alpha = exp[i];
        prim_i.norm = cart_norm(exp[i], l, m, n);

        primitives.emplace_back(prim_i);

    }

    function.center = center;
    function.lx = l;
    function.ly = m;
    function.lz = n;
    function.prims = primitives;

    return function;

}

inline
std::vector<basis_function> basis_recon(std::string file_path,
                                        std::string param_path,
                                        std::string basis_set)
{
    validate_file_exists(file_path);
    validate_folder_exists(param_path);

    rapidcsv::Document info(file_path);

    std::vector<std::string> 
    atom_list = info.GetColumn<std::string>("atom");

    std::vector<int> 
    shell_ids = info.GetColumn<int> ("shell_id");

    std::vector<int>
    lx_list = info.GetColumn<int>("lx");
    std::vector<int>
    ly_list = info.GetColumn<int>("ly");
    std::vector<int>
    lz_list = info.GetColumn<int>("lz");

    std::vector<double>
    x = info.GetColumn<double>("x");
    std::vector<double>
    y = info.GetColumn<double>("y");
    std::vector<double>
    z = info.GetColumn<double>("z");

    std::vector<basis_function> basis;

    for (int i = 0; i < x.size(); ++ i){

        std::string fileName = param_path + "/" + basis_set + "/"
                               + atom_list[i] + ".csv";
        
        validate_file_exists(fileName);
        rapidcsv::Document params(fileName);

        std::vector<int> shells
        = params.GetColumn<int>("shell_id");

        std::vector<std::pair<int, int>> 
        split_indices = section_idx(shells);

        std::pair<int, int> idx = split_indices[shell_ids[i]];

        if (shells[idx.first] != shell_ids[i]){
            throw std::runtime_error
            ("shell mismatch encountered in basis reconstruction");
        }

        std::vector<double> 
        atom_coeffs = params.GetColumn<double>("coefficient");
        std::vector<double>
        atom_exps = params.GetColumn<double>("exponent");

        std::vector<double> coeffs
        = std::vector<double>(atom_coeffs.begin() + idx.first,
                              atom_coeffs.begin() + idx.second + 1);
        std::vector<double> exps
        = std::vector<double>(atom_exps.begin() + idx.first,
                              atom_exps.begin() + idx.second + 1);

        vector3 C = position_vector(x[i], y[i], z[i]);

        basis_function func = build_function(exps, coeffs, 
                              lx_list[i], ly_list[i], lz_list[i], C);

        basis.emplace_back(func);
    }
    return basis;
}



