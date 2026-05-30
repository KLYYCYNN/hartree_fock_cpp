#pragma once
#include <vector>
#include <Eigen/Dense>
#include <iostream>
#include <string>
#include <chrono>
#include <format>
#include "integrals.hpp"
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


std::vector<double> update_damping(const std::vector<double>& new_matrix,
                                   const std::vector<double>& old_matrix,
                                   double damping)
{

    int n1 = static_cast<int>(new_matrix.size());
    int n2 = static_cast<int>(old_matrix.size());

    if (n1 != n2){
        throw std::runtime_error(R"(DimensionError: dimension inconsistency
                            encountered when updating density matrix)");
    }

    std::vector<double> updated_matrix(n1);

    for (int i = 0; i < n1; ++i){

        updated_matrix[i] = damping * new_matrix[i]
                            + (1 - damping) * old_matrix[i];

    }

    return updated_matrix;
}


std::string 
formatTimeC20(std::chrono::high_resolution_clock::time_point tp) {
    // Cast high_resolution_clock to system_clock to get calendar time
    auto sys_tp = 
    std::chrono::clock_cast<std::chrono::system_clock>(tp);
    
    // Drop the sub-seconds by casting to whole seconds
    auto whole_secs = 
    std::chrono::time_point_cast<std::chrono::seconds>(sys_tp);
    
    // %T = 24-hour time with integer seconds (HH:MM:SS), %d/%m/%y = dd/mm/yy
    return std::format("{:%T %d/%m/%y}", whole_secs);
}


rhf_data
rhf_solver(const std::vector<std::pair<std::string, vector3>>& config,
           std::string basis_set,
           std::string param_path,
           int charge = 0,
           std::string eri_device = "CPU", 
           bool set_initial_density = false,
           std::vector<double> P0 = {},
           int max_it = 10000, 
           double epsilon = 1e-6,
           double damping = 0.3,
           double screening_threshold = 1e-8)
{
    //start timing and initialize variables
    auto t0 = std::chrono::high_resolution_clock::now();                                                            
    std::cout << "\rstarting SCF cycle" << std::flush;
    std::cout << "" <<std::endl;

    if (eri_device != "CPU" && eri_device != "GPU"){
        throw std::runtime_error("unsupported ERI computing platform");
    }


    double total_energy = 0.0;
    std::vector<double> density_matrix;
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

    // if there is no input initial density matrix
    // use a zero matrix as initial guess
    if (set_initial_density){

        if (P0.size() != nbasis * nbasis){
            throw std::runtime_error
            (R"(initial density matrix dimension 
             inconsistent with basis size)");
        }

        density_matrix = P0;
    } else{
        std::vector<double> zero_vector(nbasis * nbasis, 0.0);
        density_matrix = zero_vector;
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

    rhf_data output;
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


uhf_data
uhf_solver(const std::vector<std::pair<std::string, vector3>>& config,
          std::string basis_set,
          std::string param_path,
          std::pair<int, int> spin_occ,
          int charge = 0,
          std::string eri_device = "CPU", 
          bool set_initial_density = false,
          std::vector<double> Pa0 = {},
          std::vector<double> Pb0 = {},
          int max_it = 10000, 
          double epsilon = 1e-6,
          double damping = 0.3,
          double screening_threshold = 1e-8)
{
    //start timing and initialize variables
    auto t0 = std::chrono::high_resolution_clock::now();                                                            
    std::cout << "\rstarting SCF cycle" << std::flush;
    std::cout << "" <<std::endl;

    if (eri_device != "CPU" && eri_device != "GPU"){
        throw std::runtime_error("unsupported ERI computing platform");
    }

    double E_tot = 0.0;
    double E_ele = 0.0;
    std::vector<double> Pa;
    std::vector<double> Pb;
    std::vector<double> Ca;
    std::vector<double> Cb;
    std::vector<double> Ea;
    std::vector<double> Eb;
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

    if (n_electrons != spin_occ.first + spin_occ.second){
        throw std::runtime_error(R"(inconsistency in electron number)");
    }

    std::pair<std::vector<basis_function>, basis_info> Basis = 
    basis_construction(config, basis_set, param_path);

    std::vector<basis_function> basis = Basis.first;
    int nbasis = basis.size();

    // if there is no input initial density matrix
    // use a zero matrix as initial guess
    if (set_initial_density){

        if (Pa0.size() != nbasis * nbasis
            || Pb0.size() != nbasis * nbasis){
            throw std::runtime_error
            (R"(initial density matrix dimension 
             inconsistent with basis size)");
        }

        Pa = Pa0;
        Pb = Pb0;

    } else{
        std::vector<double> zero_vector(nbasis * nbasis, 0.0);
        Pa = zero_vector;
        Pb = zero_vector;
    }


    // compute other constants before the iterating process starts; 
    // nuclear repulsion energy, ERI tensor, overlap matrix and associated
    // orthogonalization transformations.
    std::vector<double> H_core = Hcore(molecule, basis);
    double E_nuc = repulsion_energy(molecule);
    std::vector<double> S = overlap_matrix(basis);
    Eigen::MatrixXd X = Xmatrix(S);


    auto t1 = std::chrono::high_resolution_clock::now();
    double prep_time = std::chrono::duration<double>(t1 - t0).count();
    std::cout << "start calculating ERI tensor elements   "
                  "time taken:  " << prep_time << " s" << std::endl;

    std::vector<double> ERI;

    if (eri_device == "CPU"){
        ERI = repulsion_tensor(basis, screening_threshold);
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

        // compute coulomb and exchange matrices
        std::vector<double> J = coulomb_matrix(Pa, Pb, ERI);
        std::vector<double> Ka = exchange_matrix(Pa, ERI);
        std::vector<double> Kb = exchange_matrix(Pb, ERI);

        // build the fock matrices
        std::vector<double> Fa = fock_matrix_uhf(H_core, J, Ka);
        std::vector<double> Fb = fock_matrix_uhf(H_core, J, Kb);

        // solve the eigenvalue problem and compute a new density matrix
        std::pair<std::vector<double>, std::vector<double>> 
        eig_a = EigProblem(Fa, X);
        std::pair<std::vector<double>, std::vector<double>> 
        eig_b = EigProblem(Fb, X);

        Ca = eig_a.first;
        Cb = eig_b.first;
        Ea = eig_a.second;
        Eb = eig_b.second;

        std::vector<double> Pa_new = update_density_uhf(Ca, spin_occ.first);
        std::vector<double> Pb_new = update_density_uhf(Cb, spin_occ.second);
        
        // compute the squared difference with the old density matrix
        double diff_a = RMSD(Pa, Pa_new);
        double diff_b = RMSD(Pb, Pb_new);
        
        // check the RMSD, if it's small enough then the density matrix 
        // has converged and the process is terminated.
        if (diff_a < epsilon && diff_b < epsilon){

            success = true;
            Pa = Pa_new;
            Pb = Pb_new;
            E_ele = energy(Pa, Fa, H_core) 
                    + energy(Pb, Fb, H_core);

            E_tot = E_ele + E_nuc;
            std::cout << "" <<std::endl;
            std::cout << "Convergence criterion met after " << n_it << 
                          " iterations, energy value: " << 
                          E_tot << " Hartrees." << std::endl;
            break;
        }


        if (i == max_it - 1){

            std::cout << "" <<std::endl;
            std::cout << "Failed to achieve convergence, "
                      << "RMSD of density matrix elements: "
                      << diff_a << ", " << diff_b << std::endl;

        }

        // if the density matrix hasn't converged,
        // update it with damping and start a new iteration.
        Pa = update_damping(Pa_new, Pa, damping);
        Pb = update_damping(Pb_new, Pb, damping);

        //keeping track of the time taken
        auto ti = std::chrono::high_resolution_clock::now();
        double loop_time = std::chrono::duration<double>(ti - t2).count();
        std::cout << "\rCycle " << n_it << ":" 
        << loop_time << " s       " << std::flush;

    }

    auto t3 = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(t3 - t0).count();

    uhf_data output;
    output.K = nbasis;
    output.basis = Basis.second;

    output.E_ele = E_ele;
    output.E_nuc = E_nuc;
    output.E_tot = E_tot;

    output.Pa = Pa;
    output.Pb = Pb;
    output.Ca = Ca;
    output.Cb = Cb;
    output.Ea = Ea;
    output.Eb = Eb;

    output.converged = success;
    output.iterations = n_it;

    output.duration = total_time;
    output.start = formatTimeC20(t0);
    output.end = formatTimeC20(t3);

    output.charge = charge;
    output.system = config;
    output.Na = spin_occ.first;
    output.Nb = spin_occ.second;
    
    output.hardware = eri_device;
    output.BasisSet = basis_set;
    
    std::cout << "SCF cycle completed, total time taken:  " <<
                 total_time << " s" << std::endl;

    return output; 
}
