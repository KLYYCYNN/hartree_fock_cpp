#pragma once
#include <vector>
#include "gpu_types.hpp"

std::vector<int> test_gpu_basis_read(const std::vector<BasisFunctionGPU>& basis);

std::vector<double> test_gpu_boys();

std::vector<double> hermite_E_test(int i, int j,
                                   double alpha, double beta,
                                   double A, double B);

double coulomb_R_test_gpu(int t, int u, int v, int n,
                          double p, double q,
                          double Px, double Py, double Pz,
                          double Qx, double Qy, double Qz);

std::vector<double> ERI_tensor(
    const std::vector<basis_function>& cpu_basis,
    double split_fraction = 0.5,
    double threshold = 1e-9
);

std::vector<double> ERI_GPU(
    const std::vector<basis_function>& cpu_basis,
    double threshold
);