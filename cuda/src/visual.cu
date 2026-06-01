#include <cuda_runtime.h>
#include <stdexcept>
#include "evaluation.cuh"


void cuda_check(cudaError_t err, const char* msg)
{
    if (err != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(err) << std::endl;
        std::exit(1);
    }
}




std::vector<double> density_gpu(const std::vector<basis_function>& basis,
                                const std::vector<double>& density_matrix,
                                const std::vector<vector3>& r)
{

    int K = basis.size();
    int n_points = r.size();

    std::vector<BasisFunctionGPU> gpu_basis = convert_basis_to_gpu(basis);

    BasisFunctionGPU* d_basis = nullptr;
    vector3* d_r = nullptr;
    double* d_rho_r = nullptr;
    double* d_Pmat = nullptr;

    cuda_check(cudaMalloc(&d_basis, K * sizeof(BasisFunctionGPU)), "Malloc Basis");
    cuda_check(cudaMalloc(&d_r, n_points * sizeof(vector3)), "Malloc coordinate grid");
    cuda_check(cudaMalloc(&d_rho_r, n_points * sizeof(double)), "Malloc output values");
    cuda_check(cudaMalloc(&d_Pmat, K * K * sizeof(double)), "Malloc density matrix");

    cuda_check(cudaMemcpy(d_basis, gpu_basis.data(), K * sizeof(BasisFunctionGPU),
                          cudaMemcpyHostToDevice), "copy basis host to device");
    cuda_check(cudaMemcpy(d_r, r.data(), n_points * sizeof(vector3),
                          cudaMemcpyHostToDevice), "copy positions host to device");
    cuda_check(cudaMemcpy(d_Pmat, density_matrix.data(), K * K * sizeof(double),
                          cudaMemcpyHostToDevice), "copy density matrix host to device");

    int block = 256;
    int grid = (n_points + block - 1) / block;

    electron_density_gpu<<<grid, block>>>(d_basis, K, d_Pmat, n_points, d_r, d_rho_r);
    cuda_check(cudaGetLastError(), "Kernel Launch");
    cuda_check(cudaDeviceSynchronize(), "Kernel Sync");

    std::vector<double> rho_r(n_points);
    cuda_check(cudaMemcpy(rho_r.data(), d_rho_r, n_points * sizeof(double), 
                          cudaMemcpyDeviceToHost), "copy values device to host");

    cuda_check(cudaFree(d_basis), "free device basis");
    cuda_check(cudaFree(d_r), "free device positions");
    cuda_check(cudaFree(d_rho_r), "free device values");
    cuda_check(cudaFree(d_Pmat), "free device matrix");

    return rho_r;
}


std::vector<double> orbital_gpu(const std::vector<basis_function>& basis,
                                const std::vector<double>& coeff_mat,
                                const int index,
                                const std::vector<vector3>& r)
{

    int K = basis.size();
    int n_points = r.size();

    if (coeff_mat.size() != K * K){
        throw std::runtime_error(R"(Mismatch between coefficient matrix and 
                                  basis sizes)");
    }

    if (index < 0 || index >= K) {
    throw std::runtime_error("orbital index out of range");
    }

    std::vector<BasisFunctionGPU> gpu_basis = convert_basis_to_gpu(basis);

    BasisFunctionGPU* d_basis = nullptr;
    vector3* d_r = nullptr;
    double* d_psi_r = nullptr;
    double* d_Cmat = nullptr;
    
    cuda_check(cudaMalloc(&d_basis, K * sizeof(BasisFunctionGPU)), "Malloc Basis");
    cuda_check(cudaMalloc(&d_r, n_points * sizeof(vector3)), "Malloc coordinate grid");
    cuda_check(cudaMalloc(&d_psi_r, n_points * sizeof(double)), "Malloc output values");
    cuda_check(cudaMalloc(&d_Cmat, K * K * sizeof(double)), "Malloc coefficient matrix");

    cuda_check(cudaMemcpy(d_basis, gpu_basis.data(), K * sizeof(BasisFunctionGPU),
                          cudaMemcpyHostToDevice), "copy basis host to device");
    cuda_check(cudaMemcpy(d_r, r.data(), n_points * sizeof(vector3),
                          cudaMemcpyHostToDevice), "copy positions host to device");
    cuda_check(cudaMemcpy(d_Cmat, coeff_mat.data(), K * K * sizeof(double),
                          cudaMemcpyHostToDevice), "copy coefficient matrix host to device");

    int block = 256;
    int grid = (n_points + block - 1) / block;

    orbital_kernel<<<grid, block>>>(d_basis, d_Cmat, K, index, n_points, d_r, d_psi_r);
    cuda_check(cudaGetLastError(), "Kernel Launch");
    cuda_check(cudaDeviceSynchronize(), "Kernel Sync");

    std::vector<double> psi_r(n_points);
    cuda_check(cudaMemcpy(psi_r.data(), d_psi_r, n_points * sizeof(double), 
                          cudaMemcpyDeviceToHost), "copy values device to host");

    cuda_check(cudaFree(d_basis), "free device basis");
    cuda_check(cudaFree(d_r), "free device positions");
    cuda_check(cudaFree(d_psi_r), "free device values");
    cuda_check(cudaFree(d_Cmat), "free device matrix");

    return psi_r;

}