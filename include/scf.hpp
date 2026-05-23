#pragma once
#include <vector>
#include <Eigen/Dense>
#include <iostream>
#include <string>
#include <chrono>
#include <format>
#include "integrals.hpp"
#include "indexing.hpp"
#include "linalg.hpp"
#include "eri_gpu.hpp"

std::vector<double> Hcore(const std::vector<atom>& molecule,
                          const std::vector<basis_function>& basis){

    std::vector<double> T = kinetic_matrix(basis);
    std::vector<double> V = attraction_matrix(basis, molecule);

    std::vector<double> core_ham(basis.size() * basis.size());
    for (int i = 0; i < basis.size() * basis.size(); ++ i){

        core_ham[i] = T[i] + V[i];

    }

    return core_ham;
}

// compute the total electronic energy 
double energy(const std::vector<double>& density,
              const std::vector<double>& fock,
              const std::vector<double>& hcore){
    double E = 0;
    int nbasis = (int) std::sqrt(hcore.size());

    for (int i = 0; i < nbasis * nbasis; ++ i){
        E += density[i] * (hcore[i] + fock[i]);
    }

    return 0.5 * E;
}


std::vector<std::pair<int, int>> 
unique_pairs(const std::vector<int>& vec) {
    std::vector<std::pair<int, int>> combinations;
    
    if (vec.size() < 2) return combinations;

    // Outer loop goes from first element to second-to-last
    for (size_t i = 0; i < vec.size() - 1; ++i) {
        // Inner loop starts from the element after i
        for (size_t j = i + 1; j < vec.size(); ++j) {
            combinations.push_back({vec[i], vec[j]});
        }
    }

    return combinations;
}

//nuclear repulsion energy
double repulsion_energy(const std::vector<atom>& molecule){
    std::vector<int> indices;
    for (int i=0; i<molecule.size(); ++i){
        indices.emplace_back(i);
    }

    std::vector<std::pair<int, int>> pairs = unique_pairs(indices);
    double E = 0.0;
    for (int i=0; i<pairs.size(); ++i){
        E += molecule[pairs[i].first].Z * molecule[pairs[i].second].Z / 
        distance(molecule[pairs[i].first].R, molecule[pairs[i].second].R);
    }

    return E;
}


std::string
formatTimeC20(std::chrono::high_resolution_clock::time_point tp) {
    // Cast high_resolution_clock to system_clock to get calendar time
    auto sys_tp = std::chrono::clock_cast<std::chrono::system_clock>(tp);
    
    // %R = 24-hour time (HH:MM), %d/%m/%y = dd/mm/yy
    return std::format("{:%R %d/%m/%y}", sys_tp);
}


scf_data
scf_cycle(const std::vector<std::pair<std::string, vector3>>& config,
          const std::vector<double>& P0,
          std::string basis_set,
          std::string param_path,
          int charge = 0,
          std::string eri_device = "CPU", 
          int max_it = 10000, 
          double epsilon = 1e-6,
          double damping = 0.3,
          double screening_threshold = 1e-8,
          double hybrid_split = 0.24)
{
    //start timing and initialize variables
    auto t0 = std::chrono::high_resolution_clock::now();                                                            
    std::cout << "\rstarting SCF cycle" << std::flush;
    std::cout << "" <<std::endl;

    if (eri_device != "CPU" && eri_device != "Hybrid" &&
        eri_device != "GPU"){

        throw std::runtime_error("unsupported ERI computing platform");
    }

    double total_energy = 0.0;
    std::vector<double> density_matrix = P0;
    std::vector<double> coeff_matrix;
    std::vector<double> mo_energies;
    bool success = false;
    int n_it = 0;

    //convert input configuration to basis set
    std::vector<atom> molecule(config.size());
    int n_electrons = -charge;

    for (int i=0; i<config.size(); ++i){

        molecule[i].Z = nameZ(config[i].first);
        molecule[i].R = config[i].second;
        n_electrons += molecule[i].Z;

    }

    std::pair<std::vector<basis_function>, basis_info> Basis = 
    basis_construction(config, basis_set, param_path);

    std::vector<basis_function> basis = Basis.first;
    int nbasis = basis.size();
    if (P0.size() != nbasis * nbasis){
        throw std::runtime_error
        ("initial density matrix dimension inconsistent with basis size");
    }

    std::vector<double> H_core = Hcore(molecule, basis);


    //compute other constants before the iterating process starts; 
    //nuclear repulsion energy, ERI tensor, overlap matrix and associated
    //orthogonalization transformations.

    double E_nuc = repulsion_energy(molecule);
    std::vector<double> S = overlap_matrix(basis);
    Eigen::MatrixXd X = Xmatrix(S);

    std::cout << "constant matrices computed" << std::endl;

    auto t1 = std::chrono::high_resolution_clock::now();
    double prep_time = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "start calculating ERI tensor elements   "
                  "time taken:  " << prep_time << " s" << std::endl;

    std::vector<double> ERI;

    if (eri_device == "CPU"){
        ERI = repulsion_tensor(basis, screening_threshold);
    }

    if (eri_device == "Hybrid"){
        ERI = ERI_tensor(basis, hybrid_split, screening_threshold);
    }

    if (eri_device == "GPU"){
        ERI = ERI_GPU(basis, screening_threshold);
    }


    auto t2 = std::chrono::high_resolution_clock::now();
    double eri_time = std::chrono::duration<double>(t2 - t1).count();
    std::cout << "" <<std::endl;
    std::cout << "ERI calculation complete, time taken:  " << 
                 eri_time << " s" << std::endl;

    for (int i=0; i<max_it; ++i){

        n_it += 1;

        //contract the ERI tensor with the density matrix to get a matrix.
        std::vector<double> G = repulsion_matrix(density_matrix, ERI);  
        std::vector<double> fock_matrix(G.size());

        //build the fock matrix
        for (int j = 0; j < nbasis * nbasis; ++j){
            fock_matrix[j] = H_core[j] + G[j];
        }

        //solve the eigenvalue problem and compute a new density matrix
        std::pair<std::vector<double>, std::vector<double>> 
        eigen_solution = EigProblem(fock_matrix, X);

        coeff_matrix = eigen_solution.first;
        mo_energies = eigen_solution.second;
        std::vector<double> P_new = update_Pmatrix(coeff_matrix,
                                                   n_electrons);
        
        //compute the squared difference with the old density matrix
        double P_diff = 0.0;
        for (int j=0; j<P_new.size(); ++j){

            P_diff += (density_matrix[j] - P_new[j]) *
                      (density_matrix[j] - P_new[j]);
        }
        
        // check the RMSD, if it's small enough then the density matrix 
        // has converged and the process is terminated.
        if (std::sqrt(P_diff/P_new.size()) < epsilon){

            success = true;
            density_matrix = P_new;
            total_energy = energy(density_matrix, fock_matrix, H_core)
                           + E_nuc;

            std::cout << "" <<std::endl;
            std::cout << "Convergence criterion met after " << n_it << 
                          " iterations, energy value: " << 
                          total_energy << " Hartrees." << std::endl;
            break;
        }

        //keeping track of the time taken
        auto ti = std::chrono::high_resolution_clock::now();
        double loop_time = std::chrono::duration<double>(ti - t2).count();
        std::cout << "\rCycle " << n_it << ":" 
        << loop_time << " s       " << std::flush;

        // if the density matrix hasn't converged,
        // update it with damping and start a new iteration.
        for (int j = 0; j < density_matrix.size(); ++j){

            density_matrix[j] = damping * P_new[j] +
                                (1.0 - damping) * density_matrix[j];
        }


        if (i == max_it - 1){

            std::cout << "" <<std::endl;
            std::cout << "Failed to achieve convergence, "
                      << "RMSD of density matrix elements "
                      << std::sqrt(P_diff/P_new.size()) << std::endl;

        }
    }

    auto t3 = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(t3 - t0).count();

    scf_data output;
    output.E = total_energy;
    output.K = nbasis;
    output.basis = Basis.second;

    output.density = density_matrix;
    output.coefficient = coeff_matrix;
    output.mo_energy = mo_energies;

    output.converged = success;
    output.iterations = n_it;

    output.duration = total_time;
    output.start = formatTimeC20(t0);
    output.end = formatTimeC20(t3);

    output.charge = charge;
    output.system = config;
    
    output.hardware = eri_device;
    output.BasisSet = basis_set;
    
    std::cout << "SCF cycle completed, total time taken:  " <<
                 total_time << " s" << std::endl;

    return output; 
}

