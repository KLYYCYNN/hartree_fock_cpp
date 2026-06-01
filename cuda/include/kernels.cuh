#pragma once
#include "recursion.cuh"
#include "indexing.cuh"
#include "gpu_types.hpp"

__device__ __host__
void primitive_eri_gpu(const PrimitiveGPU& p1, const PrimitiveGPU& p2,
                       const PrimitiveGPU& p3, const PrimitiveGPU& p4,
                       const BasisFunctionGPU& b1, const BasisFunctionGPU& b2,
                       const BasisFunctionGPU& b3, const BasisFunctionGPU& b4,
                       double& eri)
{

    double alpha = p1.alpha;
    double beta = p2.alpha;
    double gamma = p3.alpha;
    double delta = p4.alpha;

    double p = alpha + beta;
    double q = gamma + delta;

    double Px = (alpha * b1.x + beta * b2.x) / p;
    double Py = (alpha * b1.y + beta * b2.y) / p;
    double Pz = (alpha * b1.z + beta * b2.z) / p;

    double Qx = (gamma * b3.x + delta * b4.x) / q;
    double Qy = (gamma * b3.y + delta * b4.y) / q;
    double Qz = (gamma * b3.z + delta * b4.z) / q;

    int t = b1.lx + b2.lx;
    int u = b1.ly + b2.ly;
    int v = b1.lz + b2.lz;

    int tau = b3.lx + b4.lx;
    int nu = b3.ly + b4.ly;
    int phi = b3.lz + b4.lz;

    double E12x[MAX_E], E12y[MAX_E], E12z[MAX_E], 
           E34x[MAX_E], E34y[MAX_E], E34z[MAX_E];

    hermite_E_gpu(b1.lx, b2.lx, alpha, beta, b1.x, b2.x, E12x);
    hermite_E_gpu(b1.ly, b2.ly, alpha, beta, b1.y, b2.y, E12y);
    hermite_E_gpu(b1.lz, b2.lz, alpha, beta, b1.z, b2.z, E12z);

    hermite_E_gpu(b3.lx, b4.lx, gamma, delta, b3.x, b4.x, E34x);
    hermite_E_gpu(b3.ly, b4.ly, gamma, delta, b3.y, b4.y, E34y);
    hermite_E_gpu(b3.lz, b4.lz, gamma, delta, b3.z, b4.z, E34z);

    eri = 0.0;

    for (int i1 = 0; i1 <= t; ++i1){
        for (int i2 = 0; i2 <= u; ++i2){
            for (int i3 = 0; i3 <= v; ++i3){
                for (int i4 = 0; i4 <= tau; ++i4){
                    for (int i5 = 0; i5 <= nu; ++i5){
                        for (int i6 = 0; i6 <= phi; ++i6){

                            double sign = ((i4 + i5 + i6) % 2 == 0) 
                                          ? 1.0 : -1.0;

                            double Coulomb_R;
                            coulomb_R_gpu(i1 + i4, i2 + i5, i3 + i6, 
                                          0, p, q,
                                          Px, Py, Pz,
                                          Qx, Qy, Qz,
                                          Coulomb_R);
                            
                            eri += sign * E12x[i1] * E12y[i2] * E12z[i3]
                                   * E34x[i4] * E34y[i5] * E34z[i6]
                                   * Coulomb_R;

                        }
                    }
                }
            }
        }
    }

    eri *= 2.0 * pow(PI_GPU, 2.5) / (p * q * sqrt(p + q));

}


__global__
void eri_kernel(const BasisFunctionGPU* basis,
                const double* bounds,
                const int K,
                const double threshold,
                double* unique_eri)
{
    extern __shared__ double sh[];

    size_t Ksize = static_cast<size_t>(K);
    size_t n_pairs = Ksize * (Ksize + 1) / 2;
    size_t n_quartets = n_pairs * (n_pairs + 1) / 2;

    for (size_t qid = static_cast<size_t>(blockIdx.x);
         qid < n_quartets;
         qid += static_cast<size_t>(gridDim.x)) {

        int p1id, p2id;
        quartet_to_pairs(qid, p1id, p2id);

        if (bounds[p1id] * bounds[p2id] < threshold) {
            if (threadIdx.x == 0) {
                unique_eri[qid] = 0.0;
            }
            __syncthreads();
            continue;
        }

        int mu, nu, lam, sig;
        pair_to_indices(p1id, mu, nu);
        pair_to_indices(p2id, lam, sig);

        BasisFunctionGPU b1 = basis[mu];
        BasisFunctionGPU b2 = basis[nu];
        BasisFunctionGPU b3 = basis[lam];
        BasisFunctionGPU b4 = basis[sig];

        int Nprim = b1.nprims * b2.nprims * b3.nprims * b4.nprims;

        double local = 0.0;

        for (int q = threadIdx.x; q < Nprim; q += blockDim.x) {
            int l = q % b4.nprims;
            int tmp = q / b4.nprims;

            int k = tmp % b3.nprims;
            tmp /= b3.nprims;

            int j = tmp % b2.nprims;
            int i = tmp / b2.nprims;

            PrimitiveGPU p1 = b1.prims[i];
            PrimitiveGPU p2 = b2.prims[j];
            PrimitiveGPU p3 = b3.prims[k];
            PrimitiveGPU p4 = b4.prims[l];

            double prim_eri = 0.0;

            primitive_eri_gpu(
                p1, p2, p3, p4,
                b1, b2, b3, b4,
                prim_eri
            );

            local += p1.coeff * p2.coeff * p3.coeff * p4.coeff *
                     p1.norm  * p2.norm  * p3.norm  * p4.norm  *
                     prim_eri;
        }

        sh[threadIdx.x] = local;
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                sh[threadIdx.x] += sh[threadIdx.x + stride];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            unique_eri[qid] = sh[0];
        }

        __syncthreads();
    }
}