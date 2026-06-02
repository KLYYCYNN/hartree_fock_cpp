#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <functional>
#include "visual.hpp"
#include "save.hpp"
#include "indexing.hpp"

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


std::vector<double>
orbital_wf(const std::vector<basis_function>& basis,
           const std::vector<double>& coeff_mat,
           const int idx,
           const std::vector<vector3>& r)
{

    int K = basis.size();
    std::vector<double> psi(r.size());

    for (int i = 0; i < r.size(); ++ i){
        std::vector<double> phi(K);

        for (int j = 0; j < K; ++ j){
            phi[j] = function_eval(basis[j], r[i]);
        }

        double wf = 0.0;

        for (int j = 0; j < K; ++ j){

            wf += coeff_mat[index2d(j, idx, K)] * phi[j];

        }
        
        psi[i] = wf;

        if (i % 5000 == 0){

            double progress = (i + 1.0) / r.size();
            double pctg = std::floor(progress * 1e4) / 1e2;

            std::cout << "\rEvaluating wavefunction, progress: "
                      << pctg << "%          " << std::flush;

        }
    }

    return psi;
}


std::vector<vector3> box_points(const vector3 center,
                                const double L,
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


std::vector<vector3> square_points(const vector3 center,
                                   const double L,
                                   const int res,
                                   std::string dir){

    std::vector<std::string> direcs = {"xy", "yz", "xz"};
    std::vector<std::pair<int, int>> axis = {{0, 1}, {1, 2}, {0, 2}};
    std::vector<double> pos = {center.x, center.y, center.z};

    if (dir != direcs[0] && dir != direcs[1] && dir != direcs[2]){
        throw std::runtime_error(R"(valid inputs are xy, yz, and xz)");
    }

    int direc_idx = getIndex(direcs, dir);
    std::pair<int, int> axes = axis[direc_idx];
    int p_axis = 3 - axes.first - axes.second;

    double inc = L / (res - 1.0);

    std::vector<double> corner(3);
    corner[axes.first] = pos[axes.first] - L / 2.0;
    corner[axes.second] = pos[axes.second] - L / 2.0;
    corner[p_axis] = pos[p_axis];

    std::vector<vector3> points;

    for (int i = 0; i < res; ++i){
        for (int j = 0; j < res; ++j){

            vector3 point_vec;
            std::vector<double> point(3);

            point[axes.first] = corner[axes.first] +
                                inc * static_cast<double>(i);

            point[axes.second] = corner[axes.second] +
                                 inc * static_cast<double>(j);

            point[p_axis] = pos[p_axis];

            point_vec = position_vector(point[0], point[1], point[2]);
            points.emplace_back(point_vec);
        }
    }

    return points;
}


std::vector<double> density_gpu(const std::vector<basis_function>& basis,
                                const std::vector<double>& density_matrix,
                                const std::vector<vector3>& r);

std::vector<double> orbital_gpu(const std::vector<basis_function>& basis,
                                const std::vector<double>& coeff_mat,
                                const int index,
                                const std::vector<vector3>& r);



cube_data density3d(const std::vector<basis_function>& basis,
                    const std::vector<double>& density_matrix,
                    const vector3 center,
                    const double L,
                    const int resolution,
                    std::string renderer = "CPU")
{

    size_t K = basis.size();
    if (K * K != density_matrix.size()){
        throw std::runtime_error(R"(Mismatch between density matrix and
                                    basis sizes)");
    }

    if (renderer != "CPU" && renderer != "GPU"){
        throw std::runtime_error("Unsupported rendering hardware");
    }

    std::vector<vector3> positions = box_points(center, L, resolution);
    std::vector<double> Psi2;

    if (renderer == "CPU"){
        Psi2 = electron_density(basis, density_matrix, positions);
    } else{
        Psi2 = density_gpu(basis, density_matrix, positions);
    }

    cube_data output;
    output.field = Psi2;

    output.origin = position_vector(center.x - L / 2,
                    center.y - L / 2, center.z - L / 2);
    output.dx = L / (resolution - 1);
    output.res = resolution;

    return output;
}



void check_index(int idx, int K){

    if (idx < 0 || idx >= K){
        throw std::runtime_error("orbital index out of range");
    }

}


cube_data
homo_lumo_3d(const std::vector<basis_function>& basis,
             const std::pair
             <std::vector<double>, std::vector<double>>& coeff_mats,
             const std::pair<int, int> occupation,
             const vector3 center,
             const double L,
             const int res,
             const bool homo_or_lumo = true,
             const bool alpha_or_beta = true,
             const std::string renderer = "CPU")
{

    int K = basis.size();

    std::vector<double> coeff_alpha = coeff_mats.first;
    std::vector<double> coeff_beta = coeff_mats.second;

    if (coeff_alpha.size() != coeff_beta.size()){
        throw std::runtime_error
        ("Coefficient matrices dimension mismatch");
    }

    if (coeff_alpha.size() != K * K){
        throw std::runtime_error
        ("Coefficient matrix dimension doesn't match basis size.");
    }

    if (res < 2){
        throw std::runtime_error("resolution must be at least 2");
    }

    if (renderer != "CPU" && renderer != "GPU"){
        throw std::runtime_error("unsupported rendering hardware");
    }

    std::function<std::vector<double>(
    const std::vector<basis_function>&,
    const std::vector<double>&,
    int,
    const std::vector<vector3>&)> orbital_eval;

    if (renderer == "CPU"){
        orbital_eval = orbital_wf;
    } else{
        orbital_eval = orbital_gpu;
    }

    std::vector<vector3> positions = box_points(center, L, res);

    std::vector<double> psi;
    int orbital_index;
    
    if (homo_or_lumo){
        if (alpha_or_beta){

            orbital_index = occupation.first - 1;
            check_index(orbital_index, K);
            psi = orbital_eval(basis, coeff_alpha, 
                               orbital_index, positions);

        } else{

            orbital_index = occupation.second - 1;
            check_index(orbital_index, K);
            psi = orbital_eval(basis, coeff_beta, 
                               orbital_index, positions);
        }
    } else{
        if (alpha_or_beta){

            orbital_index = occupation.first;
            check_index(orbital_index, K);
            psi = orbital_eval(basis, coeff_alpha, 
                               orbital_index , positions);

        } else{

            orbital_index = occupation.second;
            check_index(orbital_index, K);
            psi = orbital_eval(basis, coeff_beta, 
                               orbital_index , positions);

        }
    }

    cube_data psi_r;
    psi_r.field = psi;

    vector3 origin = position_vector(center.x - L / 2,
                     center.y - L / 2, center.z - L / 2);
    
    psi_r.origin = origin;
    psi_r.dx = L / (res - 1);
    psi_r.res = res;

    return psi_r;
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


std::vector<double> density2d(const std::vector<basis_function>& basis,
                              const std::vector<double>& density_matrix,
                              const vector3& center,
                              const double L,
                              const std::string direc,
                              const int resolution,
                              std::string renderer = "CPU")
{

    size_t K = basis.size();
    if (K * K != density_matrix.size()){
        throw std::runtime_error(R"(Mismatch between density matrix and
                                    basis sizes)");
    }

    if (renderer != "CPU" && renderer != "GPU"){
        throw std::runtime_error("Unsupported rendering hardware");
    }

    std::vector<vector3> positions = 
                         square_points(center, L, resolution, direc);

    std::vector<double> Psi2;

    if (renderer == "CPU"){
        Psi2 = electron_density(basis, density_matrix, positions);
    } else{
        Psi2 = density_gpu(basis, density_matrix, positions);
    }

    return Psi2;
}


std::vector<double>
homo_lumo_2d(const std::vector<basis_function>& basis,
             const std::pair
             <std::vector<double>, std::vector<double>>& coeff_mats,
             const std::pair<int, int> occupation,
             const vector3 center,
             const double L,
             const std::string direc,
             const int res,
             const bool homo_or_lumo = true,
             const bool alpha_or_beta = true,
             const std::string renderer = "CPU")
{

    int K = basis.size();

    std::vector<double> coeff_alpha = coeff_mats.first;
    std::vector<double> coeff_beta = coeff_mats.second;

    if (coeff_alpha.size() != coeff_beta.size()){
        throw std::runtime_error
        ("Coefficient matrices dimension mismatch");
    }

    if (coeff_alpha.size() != K * K){
        throw std::runtime_error
        ("Coefficient matrix dimension doesn't match basis size.");
    }

    if (res < 2){
        throw std::runtime_error("resolution must be at least 2");
    }

    if (renderer != "CPU" && renderer != "GPU"){
        throw std::runtime_error("unsupported rendering hardware");
    }

    std::function<std::vector<double>(
    const std::vector<basis_function>&,
    const std::vector<double>&,
    int,
    const std::vector<vector3>&)> orbital_eval;

    if (renderer == "CPU"){
        orbital_eval = orbital_wf;
    } else{
        orbital_eval = orbital_gpu;
    }

    std::vector<vector3> positions = square_points(center, L, res, direc);

    std::vector<double> psi;
    int orbital_index;
    
    if (homo_or_lumo){
        if (alpha_or_beta){

            orbital_index = occupation.first - 1;
            check_index(orbital_index, K);
            psi = orbital_eval(basis, coeff_alpha, 
                               orbital_index, positions);

        } else{

            orbital_index = occupation.second - 1;
            check_index(orbital_index, K);
            psi = orbital_eval(basis, coeff_beta, 
                               orbital_index, positions);
        }
    } else{
        if (alpha_or_beta){

            orbital_index = occupation.first;
            check_index(orbital_index, K);
            psi = orbital_eval(basis, coeff_alpha, 
                               orbital_index , positions);

        } else{

            orbital_index = occupation.second;
            check_index(orbital_index, K);
            psi = orbital_eval(basis, coeff_beta, 
                               orbital_index , positions);

        }
    }

    return psi;
}


inline
void check_orbital_index(const int idx, const int K)
{
    if (idx < 0 || idx >= K) {
        throw std::runtime_error("orbital index out of range");
    }
}


inline
std::vector<double>
orbital_2d(const std::vector<basis_function>& basis,
           const std::vector<double>& coeff_mat,
           const int orbital_index,
           const vector3 center,
           const double L,
           const std::string direc,
           const int res,
           const std::string renderer = "CPU")
{
    int K = static_cast<int>(basis.size());

    if (coeff_mat.size() != static_cast<size_t>(K) * K) {
        throw std::runtime_error(
            "Coefficient matrix dimension doesn't match basis size."
        );
    }

    if (res < 2) {
        throw std::runtime_error("resolution must be at least 2");
    }

    if (renderer != "CPU" && renderer != "GPU") {
        throw std::runtime_error("unsupported rendering hardware");
    }

    check_orbital_index(orbital_index, K);

    std::function<std::vector<double>(
        const std::vector<basis_function>&,
        const std::vector<double>&,
        int,
        const std::vector<vector3>&
    )> orbital_eval;

    if (renderer == "CPU") {
        orbital_eval = orbital_wf;
    } else {
        orbital_eval = orbital_gpu;
    }

    std::vector<vector3> positions =
        square_points(center, L, res, direc);

    std::vector<double> psi =
        orbital_eval(basis, coeff_mat, orbital_index, positions);

    return psi;
}


inline
cube_data
orbital_3d(const std::vector<basis_function>& basis,
           const std::vector<double>& coeff_mat,
           const int orbital_index,
           const vector3 center,
           const double L,
           const int res,
           const std::string renderer = "CPU")
{
    int K = static_cast<int>(basis.size());

    if (coeff_mat.size() != static_cast<size_t>(K) * K) {
        throw std::runtime_error(
            "Coefficient matrix dimension doesn't match basis size."
        );
    }

    if (res < 2) {
        throw std::runtime_error("resolution must be at least 2");
    }

    if (renderer != "CPU" && renderer != "GPU") {
        throw std::runtime_error("unsupported rendering hardware");
    }

    check_orbital_index(orbital_index, K);

    std::function<std::vector<double>(
        const std::vector<basis_function>&,
        const std::vector<double>&,
        int,
        const std::vector<vector3>&
    )> orbital_eval;

    if (renderer == "CPU") {
        orbital_eval = orbital_wf;
    } else {
        orbital_eval = orbital_gpu;
    }

    std::vector<vector3> positions =
        box_points(center, L, res);

    std::vector<double> psi =
        orbital_eval(basis, coeff_mat, orbital_index, positions);

    cube_data output;

    output.field = psi;

    output.origin = position_vector(
        center.x - L / 2.0,
        center.y - L / 2.0,
        center.z - L / 2.0
    );

    output.dx = L / static_cast<double>(res - 1);
    output.res = res;

    return output;
}
