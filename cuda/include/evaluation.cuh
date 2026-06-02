#include <cuda_runtime.h>
#include "indexing.cuh"
#include "gpu_types.hpp"

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


__global__
void orbital_kernel(const BasisFunctionGPU* basis,
                 const double* coeff_mat,
                 int K,
                 int column,
                 int n_points,
                 const vector3* r,
                 double* psi_r)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_points) return;

    psi_r[idx] = 0.0;

    for (int i = 0; i < K; ++i){

        double phi;
        function_eval_gpu(basis[i], r[idx], phi);
        psi_r[idx] += phi * coeff_mat[K * i + column];

    }

}