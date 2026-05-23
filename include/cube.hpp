#include <cmath>
#include <functional>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include "indexing.hpp"
#include "basis.hpp"
#include "integrals.hpp"
#include "visual.hpp"
#include "save.hpp"

double function_eval(const basis_function& function,
                     const vector3& position)
{

    vector3 A = function.center;
    vector3 r = position;
    double x = r.x;
    double y = r.y;
    double z = r.z;

    int l = function.lx;
    int m = function.ly;
    int n = function.lz;

    std::vector<primitive_gaussian> 
    primitives = function.prims;
    double mod2 = distance2(r, A);

    double val = 0.0;
    for (int i = 0; i < primitives.size(); ++i){

        double N = primitives[i].norm;
        double alpha = primitives[i].alpha;
        double c = primitives[i].coeff;

        val += N * c * std::pow(x - A.x, l) *
               std::pow(y - A.y, m) *
               std::pow(z - A.z, n) *
               std::exp(-alpha * mod2);
    }
    return val;

}

std::vector<double> electron_density(const std::vector<basis_function>& basis,
                                     const std::vector<double>& density_matrix,
                                     const std::vector<vector3>& r)
{
    int K = basis.size();
    int n = K * (K + 1) / 2;
    std::vector<double> rho_r(r.size());

    for (int i = 0; i < r.size(); ++i){

        std::vector<double> phi(K);
        for (int j = 0; j < K; ++ j){

            phi[j] = function_eval(basis[j], r[i]);

        }

        double rho = 0;

        for (int j = 0; j < n; ++ j){

            std::pair<int, int> idx = pair_to_indices(j);

            double val = density_matrix[index2d(idx.first, idx.second, K)] *
                        phi[idx.first] * phi[idx.second];

            if (idx.first == idx.second){
                rho += val;

            } else{
                rho += 2 * val;
            }
        }

        rho_r[i] = rho;

        if (i % 5000 == 0){

            double progress = (i + 1.0) / r.size();
            double pctg = std::floor(progress * 1e4) / 1e2;

            std::cout << "\rEvaluating wavefunction, progress: "
                      << pctg << "%          " << std::flush;

        }


    }

    return rho_r;
}


std::pair<int, int> homo_lumo_indices(
    const std::vector<double>& orbital_energies,
    const std::vector<double>& occupations
) {
    int K = orbital_energies.size();

    int homo = -1;
    int lumo = -1;

    for (int i = 0; i < K; ++i) {
        if (occupations[i] > 1e-8) {
            if (homo == -1 || orbital_energies[i] > orbital_energies[homo])
                homo = i;
        } else {
            if (lumo == -1 || orbital_energies[i] < orbital_energies[lumo])
                lumo = i;
        }
    }

    if (homo == -1 || lumo == -1)
        throw std::runtime_error("Could not find HOMO/LUMO");

    return {homo, lumo};
}


std::pair<std::vector<double>, std::vector<double>> 
homo_lumo_wf(const std::vector<basis_function>& basis,
             const std::vector<double>& coeff_mat,
             const std::pair<int, int> indices,
             const std::vector<vector3>& r)
{

    int K = basis.size();
    std::vector<double> homo_r(r.size());
    std::vector<double> lumo_r(r.size());

    for (int i = 0; i < r.size(); ++ i){
        std::vector<double> phi(K);

        for (int j = 0; j < K; ++ j){
            phi[j] = function_eval(basis[j], r[i]);
        }

        double homo_wf = 0.0;
        double lumo_wf = 0.0;

        int homo_idx = indices.first;
        int lumo_idx = indices.second;

        for (int j = 0; j < K; ++ j){

            homo_wf += coeff_mat[index2d(j, homo_idx, K)] * phi[j];
            lumo_wf += coeff_mat[index2d(j, lumo_idx, K)] * phi[j];

        }
        
        homo_r[i] = homo_wf;
        lumo_r[i] = lumo_wf;

        if (i % 5000 == 0){

            double progress = (i + 1.0) / r.size();
            double pctg = std::floor(progress * 1e4) / 1e2;

            std::cout << "\rEvaluating wavefunction, progress: "
                      << pctg << "%          " << std::flush;

        }
    }

    return {homo_r, lumo_r};
}


std::vector<vector3> box_points(const vector3& center,
                               const double& L,
                               const int res)
{

    if (res < 2){
        throw std::runtime_error("resolution must be at least 2");
    }

    std::vector<double> x = 
    linspace(center.x - L / 2, center.x + L / 2, res);

    std::vector<double> y = 
    linspace(center.y - L / 2, center.y + L / 2, res);

    std::vector<double> z = 
    linspace(center.z - L / 2, center.z + L / 2, res);

    std::vector<vector3> positions(res * res * res);

    for (int i = 0; i < res; ++ i){
        for (int j = 0; j < res; ++j){
            for (int k = 0; k < res; ++k){

                positions[index3d(i, j, k, res)] = 
                position_vector(x[i], y[j], z[k]);

            }
        }
    }

    return positions;

}


cube_data density3d(const std::vector<basis_function>& basis,
                    const std::vector<double>& density_matrix,
                    const vector3& center,
                    const double L,
                    const int resolution)
{

    std::vector<vector3> positions = box_points(center, L, resolution);
    
    cube_data output;
    output.field =
    electron_density(basis, density_matrix, positions);

    output.origin = position_vector(center.x - L / 2,
                    center.y - L / 2, center.z - L / 2);
    output.dx = L / (resolution - 1);
    output.res = resolution;

    return output;
}



std::pair<cube_data, cube_data>
rhf_3d_homo_lumo(const std::vector<basis_function>& basis,
                 const std::vector<double>& coeff_mat,
                 const int n_electron,
                 const vector3& center,
                 const double L,
                 const int res)
{
    int K = basis.size();

    if (coeff_mat.size() != K * K){
        throw std::runtime_error
        ("Coefficient matrix dimension doesn't match basis size.");
    }



    if (n_electron % 2 != 0){
        throw std::runtime_error("Odd electron number for RHF.");
    }


    std::vector<vector3> positions = box_points(center, L, res);

    std::pair<int, int> indices = {n_electron / 2 - 1, n_electron / 2};

    std::pair<std::vector<double>, std::vector<double>>
    homo_lumo = homo_lumo_wf(basis, coeff_mat, indices, positions);

    cube_data homo, lumo;
    homo.field = homo_lumo.first;
    lumo.field = homo_lumo.second;

    vector3 origin = position_vector(center.x - L / 2,
                     center.y - L / 2, center.z - L / 2);
            
    homo.origin = origin;
    lumo.origin = origin;

    double dx = L / (res - 1);
    homo.dx = dx;
    lumo.dx = dx;

    homo.res = res;
    lumo.res = res;

    return {homo, lumo};
}



std::pair<cube_data, cube_data>
uhf_3d_homo_lumo(const std::vector<basis_function>& basis,
                 const std::vector<double>& coeff_mat,
                 const std::vector<double>& mo_energies,
                 const std::vector<double>& occupations,
                 const vector3& center,
                 const double& L,
                 const int& res)
{

    int K = basis.size();

    if (coeff_mat.size() != K * K){
        throw std::runtime_error
        ("Coefficient matrix dimension doesn't match basis size.");
    }

    if (mo_energies.size() != K || occupations.size() != K){
        throw std::runtime_error("Bad HOMO/LUMO occupation input");
        }

    if (res < 2){
        throw std::runtime_error("resolution must be at least 2");
    }

    std::pair<int, int>
    indices = homo_lumo_indices(mo_energies, occupations);

    std::vector<vector3> positions = box_points(center, L, res);

    std::pair<std::vector<double>, std::vector<double>>
    homo_lumo = homo_lumo_wf(basis, coeff_mat, indices, positions);

    cube_data homo, lumo;
    homo.field = homo_lumo.first;
    lumo.field = homo_lumo.second;

    vector3 origin = position_vector(center.x - L / 2,
                     center.y - L / 2, center.z - L / 2);
            
    homo.origin = origin;
    lumo.origin = origin;

    double dx = L / (res - 1);
    homo.dx = dx;
    lumo.dx = dx;

    homo.res = res;
    lumo.res = res;

    return {homo, lumo};
}



void write_cube_file(const std::string& filename,
                     const cube_data& cube,
                     const std::vector<atom>& atoms,
                     const std::string& comment = 
                     "Generated by HF code")
{
    int N = cube.res;
    int n_total = N * N * N;

    if (cube.field.size() != static_cast<size_t>(n_total)) {
        throw std::runtime_error
        ("cube field size does not match resolution");
    }

    std::ofstream out(filename);

    if (!out) {
        throw std::runtime_error
        ("Could not open cube file for writing");
    }

    out << comment << "\n";
    out << "Electron density / molecular orbital cube\n";

    // Number of atoms and grid origin
    out << atoms.size() << " "
        << cube.origin.x << " "
        << cube.origin.y << " "
        << cube.origin.z << "\n";

    // Grid vectors
    out << N << " " << cube.dx << " 0.0 0.0\n";
    out << N << " 0.0 " << cube.dx << " 0.0\n";
    out << N << " 0.0 0.0 " << cube.dx << "\n";

    // Atom list: Z, charge, x, y, z
    for (const auto& a : atoms) {
        out << a.Z << " "
            << static_cast<double>(a.Z) << " "
            << a.R.x << " "
            << a.R.y << " "
            << a.R.z << "\n";
    }

    // Grid values, z index changes fastest
    int count = 0;

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {

                count += 1;
                int idx = index3d(i, j, k, N);

                out << std::scientific << std::setprecision(8)
                    << cube.field[idx] << " ";

                if (count % 6 == 0){
                    out << "\n";
                }
                
                if (count % 25000 == 0){
                    double progress = std::floor(static_cast<double>(count)
                     * 1e4  / static_cast<double>(n_total)) / 1e2;

                     std::cout << "\rWriting cube file: " << progress
                     << "%         " << std::flush;
                }
            }
        }
    }

    out << "\n";
}



void save_matrix_csv(const std::string& file_name,
                     const std::vector<double>& matrix,
                     int rows, int cols)
{
    if (matrix.size() != rows * cols){
        throw std::runtime_error(
        "matrix size incompatible with dimensions");
    }

    std::ofstream file(file_name);

    if (!file.is_open()){
        throw std::runtime_error(
        "failed to open csv output file");
    }

    for (int i = 0; i < rows; ++i){

        for (int j = 0; j < cols; ++j){

            file << matrix[i * cols + j];

            if (j != cols - 1){
                file << ",";
            }
        }

        file << "\n";
    }

    file.close();
}



void density2d(const std::string& path,
               const std::string& folder_name,
               const std::vector<basis_function>& basis, 
               const std::vector<double>& density_matrix,
               square plane, int res)
{

    vector3 C = plane.center;
    double L = plane.L;
    //obtain two orthogonal unit vectors
    vector3 e1 = v3normalize(v3sub(plane.A, C));

    vector3 e2n = v3sub(plane.B, C);
    vector3 proj21 = v3scale(e1, v3dot(e2n, e1));
    vector3 e2 = v3normalize(v3sub(e2n, proj21));
    //starting point
    vector3 corner = v3sub(C, v3add(v3scale(e1, L / 2.0), 
                                    v3scale(e2, L / 2.0)));

    std::vector<vector3> positions(res * res);

    for (int i = 0; i < res; ++ i){
        for (int j = 0; j < res; ++ j){

            double step_i = i * L / (res - 1);
            double step_j = j * L / (res - 1);

            positions[index2d(i, j, res)] =  
            v3add(corner, v3add(v3scale(e2, step_i), 
                          v3scale(e1, step_j)));

            }
        }
    
    std::vector<double> rho_r = electron_density(basis, 
                                                 density_matrix, 
                                                 positions);

    std::string folder = path + "/" + folder_name;
    create_folder(folder);
    
    std::string csv_name = folder + "/density2d.csv";
    save_matrix_csv(csv_name, rho_r, res, res);

    std::string fileName = folder + "/square.json";

    nlohmann::ordered_json data;
    data["side_length"] = L;
    data["resolution"] = std::to_string(res) + " X " +
                         std::to_string(res);

    std::ofstream file(fileName);
    if (file.is_open()){
        file << data.dump(4);
        file.close();
    } else{
        std::cout << "metadata writing failed." << std::endl;
    }

    std::vector<double> hside = {e2.x, e2.y, e2.z};
    std::vector<double> vside = {e1.x, e1.y, e1.z};
    std::vector<double> centre = {C.x, C.y, C.z};

    std::vector<std::string> 
    headers = {"hside_direc", "vside_direc", "center"};

    save_csv({hside, vside, centre}, headers, folder + "/square.csv");
}




void rhf_2d_homo_lumo(const std::string& path,
                      const std::string& folder_name,
                      const std::vector<basis_function>& basis, 
                      const std::vector<double>& coeff_mat,
                      int n_electrons, square plane, int res)
{
    if (n_electrons % 2 != 0){
        throw std::runtime_error
        ("Odd number of electrons in RHF");
    }

    vector3 C = plane.center;
    double L = plane.L;
    //obtain two orthogonal unit vectors
    vector3 e1 = v3normalize(v3sub(plane.A, C));

    vector3 e2n = v3sub(plane.B, C);
    vector3 proj21 = v3scale(e1, v3dot(e2n, e1));
    vector3 e2 = v3normalize(v3sub(e2n, proj21));
    //starting point
    vector3 corner = v3sub(C, v3add(v3scale(e1, L / 2.0), 
                                    v3scale(e2, L / 2.0)));

    std::vector<vector3> positions(res * res);

    for (int i = 0; i < res; ++ i){
        for (int j = 0; j < res; ++ j){

            double step_i = i * L / (res - 1);
            double step_j = j * L / (res - 1);

            positions[index2d(i, j, res)] =  
            v3add(corner, v3add(v3scale(e2, step_i), 
                          v3scale(e1, step_j)));

            }
        }
    
    std::pair<int, int> indices = {n_electrons / 2 - 1,
                                   n_electrons / 2};

    std::pair<std::vector<double>, std::vector<double>>
    homo_lumo = homo_lumo_wf(basis, coeff_mat, indices, positions);

    std::string folder = path + "/" + folder_name;
    create_folder(folder);
    
    save_matrix_csv(folder + "/homo2d.csv", homo_lumo.first, res, res);
    save_matrix_csv(folder + "/lumo2d.csv", homo_lumo.second, res, res);

    std::string fileName = folder + "/square.json";

    nlohmann::ordered_json data;
    data["side_length"] = L;
    data["resolution"] = std::to_string(res) + " X " +
                         std::to_string(res);

    std::ofstream file(fileName);
    if (file.is_open()){
        file << data.dump(4);
        file.close();
    } else{
        std::cout << "metadata writing failed." << std::endl;
    }

    std::vector<double> hside = {e2.x, e2.y, e2.z};
    std::vector<double> vside = {e1.x, e1.y, e1.z};
    std::vector<double> centre = {C.x, C.y, C.z};

    std::vector<std::string> 
    headers = {"hside_direc", "vside_direc", "center"};

    save_csv({hside, vside, centre}, headers, folder + "/square.csv");

}


void uhf_2d_homo_lumo(const std::string& path,
                      const std::string& folder_name,
                      const std::vector<basis_function>& basis, 
                      const std::vector<double>& coeff_mat,
                      const std::vector<double>& mo_energies,
                      const std::vector<double>& occupations,
                      square plane, int res)
{


    vector3 C = plane.center;
    double L = plane.L;
    //obtain two orthogonal unit vectors
    vector3 e1 = v3normalize(v3sub(plane.A, C));

    vector3 e2n = v3sub(plane.B, C);
    vector3 proj21 = v3scale(e1, v3dot(e2n, e1));
    vector3 e2 = v3normalize(v3sub(e2n, proj21));
    //starting point
    vector3 corner = v3sub(C, v3add(v3scale(e1, L / 2.0), 
                                    v3scale(e2, L / 2.0)));

    std::vector<vector3> positions(res * res);

    for (int i = 0; i < res; ++ i){
        for (int j = 0; j < res; ++ j){

            double step_i = i * L / (res - 1);
            double step_j = j * L / (res - 1);

            positions[index2d(i, j, res)] =  
            v3add(corner, v3add(v3scale(e2, step_i), 
                          v3scale(e1, step_j)));

            }
        }
    
    std::pair<int, int> indices = homo_lumo_indices(mo_energies, occupations);

    std::pair<std::vector<double>, std::vector<double>>
    homo_lumo = homo_lumo_wf(basis, coeff_mat, indices, positions);

    std::string folder = path + "/" + folder_name;
    create_folder(folder);
    
    save_matrix_csv(folder + "/homo2d.csv", homo_lumo.first, res, res);
    save_matrix_csv(folder + "/lumo2d.csv", homo_lumo.second, res, res);

    nlohmann::ordered_json data;
    data["side_length"] = L;
    data["resolution"] = std::to_string(res) + " X " +
                         std::to_string(res);

    std::string fileName = folder + "/square/json";

    std::ofstream file(fileName);
    if (file.is_open()){
        file << data.dump(4);
        file.close();
    } else{
        std::cout << "metadata writing failed." << std::endl;
    }

    std::vector<double> hside = {e2.x, e2.y, e2.z};
    std::vector<double> vside = {e1.x, e1.y, e1.z};
    std::vector<double> centre = {C.x, C.y, C.z};

    std::vector<std::string> 
    headers = {"hside_direc", "vside_direc", "center"};

    save_csv({hside, vside, centre}, headers, folder + "/square.csv");

}


cube_data density3d_gpu(const std::vector<basis_function>& basis,
                        const std::vector<double>& density_matrix,
                        const vector3& center,
                        const double L,
                        const int resolution)
{

    std::vector<vector3> positions = box_points(center, L, resolution);
    
    cube_data output;
    output.field = density_gpu(basis, density_matrix, positions);

    output.origin = position_vector(center.x - L / 2,
                    center.y - L / 2, center.z - L / 2);
    output.dx = L / (resolution - 1);
    output.res = resolution;

    return output;
}
