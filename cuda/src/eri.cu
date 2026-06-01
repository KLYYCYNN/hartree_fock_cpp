#include "gpu_types.hpp"
#include "integrals.hpp"
#include "kernels.cuh"
#include <cuda_runtime.h>
#include <stdexcept>


inline void cuda_check(cudaError_t err, const char* msg)
{
    if (err != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(err) << std::endl;
        std::exit(1);
    }
}



// This function computes ERI by letting cpu and gpu do the same thing

std::vector<double> ERI_GPU(
    const std::vector<basis_function>& basis,
    double threshold
) {

    std::vector<BasisFunctionGPU> h_basis = convert_basis_to_gpu(basis);

    int K = basis.size();
    size_t n_pairs = K * (K + 1) / 2;
    size_t n_quartets = n_pairs * (n_pairs + 1) / 2;

    // compute bounds on cpu
    std::vector<double> h_bounds = schwarz_bounds(basis);

    // create pointers for basis set, bounds and eri in VRAM
    BasisFunctionGPU* d_basis = nullptr;
    double* d_bounds = nullptr;
    double* d_eri_unique = nullptr;

    // allocate VRAM for these three things
    cuda_check(cudaMalloc(&d_basis, 
                          K * sizeof(BasisFunctionGPU)),
                          "cudaMalloc d_basis");

    cuda_check(cudaMalloc(&d_bounds, 
                          n_pairs * sizeof(double)),
                          "cudaMalloc d_bounds");

    cuda_check(cudaMalloc(&d_eri_unique, 
                          n_quartets * sizeof(double)),
                          "cudaMalloc eri_unique");


    // transfer basis and bounds from DRAM to VRAM
    cuda_check(cudaMemcpy(d_basis, h_basis.data(),
                          K * sizeof(BasisFunctionGPU),
                          cudaMemcpyHostToDevice),
                          "cudaMemcpy basis H2D");

    cuda_check(cudaMemcpy(d_bounds, h_bounds.data(), 
                          n_pairs * sizeof(double),
                          cudaMemcpyHostToDevice), 
                          "cudaMemcpy bounds H2D");

    // set up number of blocks, grids, and the size of shared memory
    int block = 256;
    int grid = std::min<size_t>(65535, n_quartets);
    size_t shared_memory = block * sizeof(double);

    // launch the kernel to compute all unique eri on GPU
    eri_kernel<<<grid, block, shared_memory>>>(
        d_basis, d_bounds, K, threshold, d_eri_unique
    );

    cuda_check(cudaGetLastError(), "eri kernel launch");
    cuda_check(cudaDeviceSynchronize(), "eri kernel sync");

    //copy the unique eri elements from VRAM to DRAM
    std::vector<double> h_eri_unique(n_quartets);
    cuda_check(cudaMemcpy(h_eri_unique.data(), d_eri_unique, 
                          n_quartets * sizeof(double),
                          cudaMemcpyDeviceToHost), 
                          "cudaMemcpy eri D2H");

    cudaFree(d_eri_unique);
    cudaFree(d_bounds);
    cudaFree(d_basis);

    return h_eri_unique;
}




