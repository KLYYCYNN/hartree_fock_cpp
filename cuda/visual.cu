#include <cuda_runtime.h>
#include <stdexcept>
#include "gpu_types.hpp"


void cuda_check(cudaError_t err, const char* msg)
{
    if (err != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(err) << std::endl;
        std::exit(1);
    }
}


__device__
void gpu_int_pow(double base, int index, double& pow){

    pow = 1.0;
    for (int i = 0; i < index; ++ i){

        pow *= base;

    }

}


__device__
void function_eval_gpu(const BasisFunctionGPU& function,
                       const vector3& position,
                       double& val)
{
    double x = position.x;
    double y = position.y;
    double z = position.z;

    int l = function.lx;
    int m = function.ly;
    int n = function.lz;

    double d2 = (function.x - x) *
                (function.x - x) +
                (function.y - y) *
                (function.y - y) +
                (function.z - z) *
                (function.z - z);


    val = 0.0;

    for (int i = 0; i < function.nprims; ++i){

        double N = function.prims[i].norm;
        double c = function.prims[i].coeff;
        double a = function.prims[i].alpha;

        double powx, powy, powz;
        gpu_int_pow(x - function.x, l, powx);
        gpu_int_pow(y - function.y, m, powy);
        gpu_int_pow(z - function.z, n, powz);

        val += N * c * powx * powy * powz *
               exp(-a * d2);

    }
}

__device__
void pair_to_indices(int pid, int& a, int& b) {
    a = int((sqrt(8.0 * pid + 1.0) - 1.0) / 2.0);
    b = pid - a * (a + 1) / 2;
}


__global__
void electron_density_gpu(const BasisFunctionGPU* basis,
                          int K,
                          const double* density_matrix,
                          int n_points,
                          const vector3* r,
                          double* rho_r)

{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_points) return;

    rho_r[idx] = 0.0;
    int n = K * (K + 1) / 2;

    for (int i = 0; i < n; ++ i){

        int idx1, idx2;
        pair_to_indices(i, idx1, idx2);

        double phi1, phi2;
        function_eval_gpu(basis[idx1], r[idx], phi1);
        function_eval_gpu(basis[idx2], r[idx], phi2);

        double val = density_matrix[idx1 * K + idx2] *
                     phi1 * phi2;

        if (idx1 == idx2){

            rho_r[idx] += val;

        } else{

            rho_r[idx] += 2.0 * val;

        }
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
    cuda_check(cudaMalloc(&d_r, n_points * sizeof(vector3)), "Malloc Positions");
    cuda_check(cudaMalloc(&d_rho_r, n_points * sizeof(double)), "Malloc function values");
    cuda_check(cudaMalloc(&d_Pmat, K * K * sizeof(double)), "Malloc function values");

    cuda_check(cudaMemcpy(d_basis, gpu_basis.data(), K * sizeof(BasisFunctionGPU),
                          cudaMemcpyHostToDevice), "copy basis host to device");
    cuda_check(cudaMemcpy(d_r, r.data(), n_points * sizeof(vector3),
                          cudaMemcpyHostToDevice), "copy positions host to device");
    cuda_check(cudaMemcpy(d_Pmat, density_matrix.data(), K * K * sizeof(double),
                          cudaMemcpyHostToDevice), "copy positions host to device");

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
    (cudaFree(d_rho_r), "free device values");
    (cudaFree(d_Pmat), "free device matrix");

    return rho_r;
}
