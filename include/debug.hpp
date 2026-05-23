#pragma once
#include <string>
#include <algorithm>
#include <iterator>
#include <cmath>
#include <numbers>
#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <stdexcept>
#include "types.hpp"
#include "rapidcsv.h"
#include "atoms.hpp"
#include "integrals.hpp"
#include "indexing.hpp"

std::pair<std::vector<basis_function>, std::vector<std::string>> debug_sto3g_basis(std::vector<atom> atoms){    //input a list of atoms, containing positions and Z numbers

    std::vector<basis_function> basis_functions;
    int n_atoms = atoms.size();
    rapidcsv::Document basis_number("params/n_basis.csv"); //This file tells the program how many s and p type basis functions there are for each kind of atom
    std::vector<int> s_list = basis_number.GetColumn<int>("s");
    std::vector<int> p_list = basis_number.GetColumn<int>("p");

    std::vector<std::vector<int>> p_angmom = {{0,0,1}, {1,0,0}, {0,1,0}};
    std::vector<std::string> axis = {"z", "x", "y"};
    std::vector<std::string> basis_order;

    for (int i=0; i<n_atoms; ++i){           //iterate over each atom
        atom atom_i = atoms[i];
        std::string atom_type = Zname(atom_i.Z);  //get the name of an atom in the list
        std::string fileName = "params/" + atom_type + ".csv";   
        rapidcsv::Document parameters(fileName);       // read the basis set parameter of this atom

        std::vector<double> contraction_coefficients = parameters.GetColumn<double>("contraction_coefficients");
        std::vector<double> exponents = parameters.GetColumn<double>("exponents");

        int s = s_list[atom_i.Z-1];
        int p = p_list[atom_i.Z-1];
    
        for (int j=0; j<s; ++j){
            basis_function basis_j;
            basis_j.prims = {};

            for (int k=0; k<3; ++k){     // loop three times to contruct the basis function from 3 primitive gaussians
                primitive_gaussian prim_k;
                double d = contraction_coefficients[3*j+k];
                double alpha = exponents[3*j+k];
                double N = pow( 2.0*alpha/M_PI, 0.75 );
                // get the exponents and contraction coefficients from the file, and calculate the normalization
                prim_k.alpha = alpha;
                prim_k.coeff = d;
                prim_k.norm = N;
                basis_j.prims.emplace_back(prim_k); //defined the parameters of this primitive gaussian, and add it to the basis function
            }
        
            basis_j.center = atom_i.R;
            basis_j.lx = 0;
            basis_j.ly = 0;
            basis_j.lz = 0;

           basis_functions.emplace_back(basis_j); //set other parameters of the basis function and add it to the set of basis functions.
           std::string basis_type = atom_type + "_" + std::to_string(j+1) + "s";
           basis_order.emplace_back(basis_type);
        }


        int pshells = (int) p/3;
        for (int j=0; j<pshells; ++j){
            

            for (int k=0; k<3; ++k){ //3 for px, py, pz
                basis_function basis_k;
                basis_k.prims = {};
            
                for (int l=0; l<3; ++l){ //3 for 3 gaussians per function
                    primitive_gaussian prim_l;
                    double d = contraction_coefficients[3*s+3*j+l];
                    double alpha = exponents[3*s+3*j+l];
                    double N = 2 * std::sqrt(alpha) * pow( 2.0*alpha/M_PI, 0.75 );
                    // get the exponents and contraction coefficients from the file, and calculate the normalization
                    prim_l.alpha = alpha;
                    prim_l.coeff = d;
                    prim_l.norm = N;
                    basis_k.prims.emplace_back(prim_l); //defined the parameters of this primitive gaussian, and add it to the basis function

                }

                std::vector<int> angmom = p_angmom[k];
                basis_k.lx = angmom[0];
                basis_k.ly = angmom[1];
                basis_k.lz = angmom[2];
                basis_k.axis = k;
                basis_k.l = 1;

                basis_functions.emplace_back(basis_k);

                std::string basis_type = atom_type + "_" + std::to_string(j+1) + "p" + axis[k];
                basis_order.emplace_back(basis_type);
            }
        }


    }

    return {basis_functions, basis_order};

}

int p(int i) {
    if (i == 2) return 3; // px
    if (i == 3) return 4; // py
    if (i == 4) return 2; // pz
    return i;             // Li 1s, Li 2s, H 1s
}

void diagnose_eri_mapping(
    const std::vector<double>& ERI1,
    const std::vector<double>& ERI2,
    const std::vector<int>& map,
    int K,
    double tol = 1e-10
) {
    if ((int)map.size() != K) {
        throw std::runtime_error("map.size() != K");
    }

    if ((int)ERI1.size() != K*K*K*K || (int)ERI2.size() != K*K*K*K) {
        throw std::runtime_error("ERI tensor size is not K^4");
    }

    double max_abs_err = 0.0;
    double max_rel_err = 0.0;

    int wi = 0, wj = 0, wk = 0, wl = 0;

    int n_bad = 0;
    int n_total = K*K*K*K;

    for (int i = 0; i < K; ++i)
    for (int j = 0; j < K; ++j)
    for (int k = 0; k < K; ++k)
    for (int l = 0; l < K; ++l) {
        int ip = map[i];
        int jp = map[j];
        int kp = map[k];
        int lp = map[l];

        double a = ERI1[index4d(i, j, k, l, K)];
        double b = ERI2[index4d(ip, jp, kp, lp, K)];

        double abs_err = std::abs(a - b);
        double denom = std::max({std::abs(a), std::abs(b), 1e-14});
        double rel_err = abs_err / denom;

        if (abs_err > tol) {
            ++n_bad;
        }

        if (abs_err > max_abs_err) {
            max_abs_err = abs_err;
            max_rel_err = rel_err;
            wi = i; wj = j; wk = k; wl = l;
        }
    }

    std::cout << std::setprecision(12);
    std::cout << "\n=== ERI mapping diagnosis ===\n";
    std::cout << "K = " << K << "\n";
    std::cout << "Total elements checked = " << n_total << "\n";
    std::cout << "Elements above tol = " << n_bad << "\n";
    std::cout << "Tolerance = " << tol << "\n";
    std::cout << "Max abs error = " << max_abs_err << "\n";
    std::cout << "Max rel error = " << max_rel_err << "\n";

    std::cout << "\nWorst element:\n";
    std::cout << "ERI1[" << wi << "," << wj << "," << wk << "," << wl << "]\n";

    int wip = map[wi];
    int wjp = map[wj];
    int wkp = map[wk];
    int wlp = map[wl];

    std::cout << "ERI2[" << wip << "," << wjp << "," << wkp << "," << wlp << "]\n";

    double a = ERI1[index4d(wi, wj, wk, wl, K)];
    double b = ERI2[index4d(wip, wjp, wkp, wlp, K)];

    std::cout << "ERI1 value = " << a << "\n";
    std::cout << "ERI2 value = " << b << "\n";
    std::cout << "difference = " << a - b << "\n";

    if (max_abs_err < tol) {
        std::cout << "\nResult: PASS. ERI tensors match under this mapping.\n";
    } else {
        std::cout << "\nResult: FAIL. ERI tensors do not match under this mapping.\n";
    }
}


void check_eri_symmetry(
    const std::vector<double>& ERI,
    int K,
    double tol = 1e-10
) {
    if ((int)ERI.size() != K * K * K * K) {
        throw std::runtime_error("ERI tensor size is not K^4.");
    }

    double max_err_nu_mu = 0.0;
    double max_err_sig_lam = 0.0;
    double max_err_pair_swap = 0.0;
    double max_err_all = 0.0;

    int bad_nu_mu = 0;
    int bad_sig_lam = 0;
    int bad_pair_swap = 0;

    int wi = 0, wj = 0, wk = 0, wl = 0;
    std::string worst_type;

    for (int mu = 0; mu < K; ++mu)
    for (int nu = 0; nu < K; ++nu)
    for (int lambda = 0; lambda < K; ++lambda)
    for (int sigma = 0; sigma < K; ++sigma) {

        double eri = ERI[index4d(mu, nu, lambda, sigma, K)];

        double eri_nu_mu =
            ERI[index4d(nu, mu, lambda, sigma, K)];

        double eri_sig_lam =
            ERI[index4d(mu, nu, sigma, lambda, K)];

        double eri_pair_swap =
            ERI[index4d(lambda, sigma, mu, nu, K)];

        double err_nu_mu = std::abs(eri - eri_nu_mu);
        double err_sig_lam = std::abs(eri - eri_sig_lam);
        double err_pair_swap = std::abs(eri - eri_pair_swap);

        if (err_nu_mu > tol) ++bad_nu_mu;
        if (err_sig_lam > tol) ++bad_sig_lam;
        if (err_pair_swap > tol) ++bad_pair_swap;

        if (err_nu_mu > max_err_nu_mu) {
            max_err_nu_mu = err_nu_mu;
        }

        if (err_sig_lam > max_err_sig_lam) {
            max_err_sig_lam = err_sig_lam;
        }

        if (err_pair_swap > max_err_pair_swap) {
            max_err_pair_swap = err_pair_swap;
        }

        if (err_nu_mu > max_err_all) {
            max_err_all = err_nu_mu;
            wi = mu; wj = nu; wk = lambda; wl = sigma;
            worst_type = "(mu nu | lambda sigma) vs (nu mu | lambda sigma)";
        }

        if (err_sig_lam > max_err_all) {
            max_err_all = err_sig_lam;
            wi = mu; wj = nu; wk = lambda; wl = sigma;
            worst_type = "(mu nu | lambda sigma) vs (mu nu | sigma lambda)";
        }

        if (err_pair_swap > max_err_all) {
            max_err_all = err_pair_swap;
            wi = mu; wj = nu; wk = lambda; wl = sigma;
            worst_type = "(mu nu | lambda sigma) vs (lambda sigma | mu nu)";
        }
    }

    std::cout << std::setprecision(14);

    std::cout << "\n=== ERI symmetry check ===\n";
    std::cout << "Basis size K = " << K << "\n";
    std::cout << "Total ERI elements checked = " << K*K*K*K << "\n";
    std::cout << "Tolerance = " << tol << "\n\n";

    std::cout << "Symmetry 1: (mu nu | lambda sigma) = (nu mu | lambda sigma)\n";
    std::cout << "  Max error = " << max_err_nu_mu << "\n";
    std::cout << "  Bad elements = " << bad_nu_mu << "\n\n";

    std::cout << "Symmetry 2: (mu nu | lambda sigma) = (mu nu | sigma lambda)\n";
    std::cout << "  Max error = " << max_err_sig_lam << "\n";
    std::cout << "  Bad elements = " << bad_sig_lam << "\n\n";

    std::cout << "Symmetry 3: (mu nu | lambda sigma) = (lambda sigma | mu nu)\n";
    std::cout << "  Max error = " << max_err_pair_swap << "\n";
    std::cout << "  Bad elements = " << bad_pair_swap << "\n\n";

    std::cout << "Worst overall error = " << max_err_all << "\n";
    std::cout << "Worst type: " << worst_type << "\n";
    std::cout << "At indices: "
              << "(" << wi << " " << wj << " | "
              << wk << " " << wl << ")\n";

    double eri = ERI[index4d(wi, wj, wk, wl, K)];

    std::cout << "Reference value = " << eri << "\n";

    std::cout << "Compared values:\n";
    std::cout << "  (" << wj << " " << wi << " | " << wk << " " << wl << ") = "
              << ERI[index4d(wj, wi, wk, wl, K)] << "\n";

    std::cout << "  (" << wi << " " << wj << " | " << wl << " " << wk << ") = "
              << ERI[index4d(wi, wj, wl, wk, K)] << "\n";

    std::cout << "  (" << wk << " " << wl << " | " << wi << " " << wj << ") = "
              << ERI[index4d(wk, wl, wi, wj, K)] << "\n";

    if (max_err_all < tol) {
        std::cout << "\nResult: PASS. ERI tensor symmetry looks good.\n";
    } else {
        std::cout << "\nResult: FAIL. ERI tensor symmetry is broken.\n";
    }
}

double boys_debug_quad(int m, double T)
{
    if (m < 0) {
        throw std::runtime_error("boys: negative m");
    }

    if (T < 0.0 || !std::isfinite(T)) {
        throw std::runtime_error("boys: invalid T");
    }

    // Safe for small T.
    // For s/p ERIs, m <= 4, so this is cheap.
    if (T < 1e-2) {
        double sum = 0.0;
        double term = 1.0;

        for (int k = 0; k < 100; ++k) {
            if (k > 0) {
                term *= -T / k;
            }

            double add = term / (2.0 * m + 2.0 * k + 1.0);
            sum += add;

            if (std::abs(add) < 1e-16 * std::max(1.0, std::abs(sum))) {
                break;
            }
        }

        return sum;
    }

    // For larger T, upward recursion is fine for m <= 4.
    double F = 0.5 * std::sqrt(M_PI / T) * std::erf(std::sqrt(T));

    for (int k = 0; k < m; ++k) {
        F = ((2.0 * k + 1.0) * F - std::exp(-T)) / (2.0 * T);
    }

    return F;
}


void basis_check(const std::vector<basis_function>& basis){
    std::cout << "\n================ BASIS SET ================\n";

    for (size_t i = 0; i < basis.size(); ++i){
        const basis_function& bf = basis[i];

        std::cout << "\nBasis function " << i << "\n";

        std::cout << "Center = ("
                  << bf.center.x << ", "
                  << bf.center.y << ", "
                  << bf.center.z << ")\n";

        std::cout << "Angular momentum = ("
                  << bf.lx << ", "
                  << bf.ly << ", "
                  << bf.lz << ")\n";

        if (bf.l == 0)
            std::cout << "Type = S\n";

        if (bf.l == 1){
            if (bf.axis == 0) std::cout << "Type = Px\n";
            if (bf.axis == 1) std::cout << "Type = Py\n";
            if (bf.axis == 2) std::cout << "Type = Pz\n";
        }

        std::cout << "Number of primitives = "
                  << bf.prims.size() << "\n";

        std::cout << "\nPrimitives:\n";

        for (size_t j = 0; j < bf.prims.size(); ++j){
            const primitive_gaussian& prim = bf.prims[j];

            std::cout
                << "  " << j
                << "  alpha = " << prim.alpha
                << "  coeff = " << prim.coeff
                << "  norm = "  << prim.norm
                << "\n";
        }
    }

    std::cout << "\n===========================================\n";
}