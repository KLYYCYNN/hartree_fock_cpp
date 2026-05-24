#include "gpu_types.hpp"
#include "eri_cpu.hpp"
#include <cuda_runtime.h>
#include <stdexcept>
#include <thread>
#include <algorithm>

#define CUDA_CHECK(call)                                      \
    do {                                                      \
        cudaError_t err = call;                               \
        if (err != cudaSuccess) {                             \
            throw std::runtime_error(cudaGetErrorString(err));\
        }                                                     \
    } while (0)


#define CUDA_CHECK_MSG(call, msg)                                      \
    do {                                                               \
        cudaError_t err = call;                                        \
        if (err != cudaSuccess) {                                      \
            std::cerr << msg << ": " << cudaGetErrorString(err) << "\n";\
            throw std::runtime_error(cudaGetErrorString(err));         \
        }                                                              \
    } while (0)

__global__
void test_basis_kernel(const BasisFunctionGPU* basis,
                       int* output,
                       int K)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= K) return;

    const BasisFunctionGPU bf = basis[idx];

    output[idx] = bf.lx + bf.ly + bf.lz;
}


std::vector<int> test_gpu_basis_read(const std::vector<BasisFunctionGPU>& basis)
{
    int K = static_cast<int>(basis.size());

    BasisFunctionGPU* d_basis = nullptr;
    int* d_output = nullptr;

    //allocate VRAM for arrays of basis functions and integers
    CUDA_CHECK(cudaMalloc(&d_basis, K * sizeof(BasisFunctionGPU))); 
    CUDA_CHECK(cudaMalloc(&d_output, K * sizeof(int)));

    //transfer basis data from DRAM to VRAM
    CUDA_CHECK(cudaMemcpy(d_basis,
                          basis.data(),
                          K * sizeof(BasisFunctionGPU),
                          cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (K + threads - 1) / threads;

    //launch kernel, it will take information from d_basis array and 
    //fill the d_output array with the computed output.
    test_basis_kernel<<<blocks, threads>>>(d_basis, d_output, K);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    //create an object to collect the data in DRAM
    std::vector<int> output(K);

    //transfer the compute output from VRAM back to DRAM
    CUDA_CHECK(cudaMemcpy(output.data(),
                          d_output,
                          K * sizeof(int),
                          cudaMemcpyDeviceToHost));
    //free VRAM
    CUDA_CHECK(cudaFree(d_basis));
    CUDA_CHECK(cudaFree(d_output));

    return output;
}


//-------------------BOYS FUNCITON--------------------------

__device__
void boys_gpu(int m, double T, double& F)
{
    if (T < 1e-8) {
        F = 1.0 / (2.0 * m + 1.0);
        return;
    }

    F = 0.5 * sqrt(M_PI / T) * erf(sqrt(T));

    for (int k = 0; k < m; ++k) {
        F = ((2.0 * k + 1.0) * F - exp(-T)) / (2.0 * T);
    }
}


__global__
void boys_test_kernel(double* output)
{
    boys_gpu(0, 1.0, output[0]);
    boys_gpu(1, 1.0, output[1]);
    boys_gpu(2, 1.0, output[2]);
}

//-------------------------------------------------------------

std::vector<double> test_gpu_boys(){

    std::vector<double> output(3);

    double* d_output = nullptr;

    CUDA_CHECK(cudaMalloc(&d_output, 3 * sizeof(double)));

    int threads = 256;
    int blocks = 1;

    boys_test_kernel<<<blocks, threads>>>(d_output);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(output.data(),
                          d_output,
                          3 * sizeof(double),
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_output));

    return output;
}


//--------------------HERMITE RECURSION-------------------------

__device__
void hermite_E_gpu(int i, int j, int t,
                   double alpha, double beta,
                   double A, double B,
                   double& out)
{
    double p = alpha + beta;
    double q = alpha * beta / p;
    double AB = A - B;

    // invalid Hermite index
    if (t < 0 || t > i + j) {
        out = 0.0;
        return;
    }

    // base case
    if (i == 0 && j == 0 && t == 0) {
        out = exp(-q * AB * AB);
        return;
    }

    // if no angular momentum left but t is nonzero
    if (i == 0 && j == 0) {
        out = 0.0;
        return;
    }

    double term1 = 0.0;
    double term2 = 0.0;
    double term3 = 0.0;

    if (i > 0) {
        double E1, E2, E3;

        hermite_E_gpu(i - 1, j, t - 1,
                             alpha, beta, A, B, E1);

        hermite_E_gpu(i - 1, j, t,
                             alpha, beta, A, B, E2);

        hermite_E_gpu(i - 1, j, t + 1,
                             alpha, beta, A, B, E3);

        term1 = E1 / (2.0 * p);
        term2 = -(beta / p) * AB * E2;
        term3 = (t + 1) * E3;

        out = term1 + term2 + term3;
        return;
    }

    if (j > 0) {
        double E1, E2, E3;

        hermite_E_gpu(i, j - 1, t - 1,
                             alpha, beta, A, B, E1);

        hermite_E_gpu(i, j - 1, t,
                             alpha, beta, A, B, E2);

        hermite_E_gpu(i, j - 1, t + 1,
                             alpha, beta, A, B, E3);

        term1 = E1 / (2.0 * p);
        term2 = +(alpha / p) * AB * E2;
        term3 = (t + 1) * E3;

        out = term1 + term2 + term3;
        return;
    }
}

//-----------------------------------------------------------------------------

__global__
void hermite_test_kernel(int i, int j, int t,
                         double alpha, double beta,
                         double A, double B,
                         double* E_array)
{
    hermite_E_gpu(i, j, t, alpha, beta, A, B, E_array[0]);
}



double hermite_E_test(int i, int j, int t,
                      double alpha, double beta,
                      double A, double B)
{
    double output = 0.0;
    double* d_output = nullptr;

    CUDA_CHECK(cudaMalloc(&d_output, sizeof(double)));

    hermite_test_kernel<<<1, 1>>>(i, j, t, alpha, beta, A, B, d_output);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(&output,
                          d_output,
                          sizeof(double),
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_output));

    return output;
}


//------------------------COULOMB R RECURSION------------------------------------------------

constexpr int MAX_R = 5; // enough for up to f/g in many cases

__device__
int r3idx(int a, int b, int c)
{
    return (a * MAX_R + b) * MAX_R + c;
}

__device__
void coulomb_R_gpu(int ta, int ub, int vc, int n,
                   double p, double q,
                        double Px, double Py, double Pz,
                        double Qx, double Qy, double Qz,
                        double& out)
{
    if (ta >= MAX_R || ub >= MAX_R || vc >= MAX_R) {
        out = 0.0;
        return;
    }

    double rho = p * q / (p + q);

    double X = Px - Qx;
    double Y = Py - Qy;
    double Z = Pz - Qz;

    double T = rho * (X*X + Y*Y + Z*Z);

    int max_m = ta + ub + vc + n;

    double R_next[MAX_R * MAX_R * MAX_R];
    double R_curr[MAX_R * MAX_R * MAX_R];

    for (int i = 0; i < MAX_R * MAX_R * MAX_R; ++i) {
        R_next[i] = 0.0;
        R_curr[i] = 0.0;
    }

    // highest Boys order layer
    double F;
    boys_gpu(max_m, T, F);
    R_next[r3idx(0,0,0)] = pow(-2.0 * rho, max_m) * F;

    // descend m = max_m - 1 down to n
    for (int m = max_m - 1; m >= n; --m) {

        for (int i = 0; i < MAX_R * MAX_R * MAX_R; ++i)
            R_curr[i] = 0.0;

        double Fm;
        boys_gpu(m, T, Fm);
        R_curr[r3idx(0,0,0)] = pow(-2.0 * rho, m) * Fm;

        for (int a = 0; a <= ta; ++a) {
            for (int b = 0; b <= ub; ++b) {
                for (int c = 0; c <= vc; ++c) {

                    if (a == 0 && b == 0 && c == 0) continue;

                    double val = 0.0;

                    if (a > 0) {
                        val = X * R_next[r3idx(a-1,b,c)];

                        if (a >= 2) {
                            val += (a - 1) * R_next[r3idx(a-2,b,c)];
                        }
                    }
                    else if (b > 0) {
                        val = Y * R_next[r3idx(a,b-1,c)];

                        if (b >= 2) {
                            val += (b - 1) * R_next[r3idx(a,b-2,c)];
                        }
                    }
                    else if (c > 0) {
                        val = Z * R_next[r3idx(a,b,c-1)];

                        if (c >= 2) {
                            val += (c - 1) * R_next[r3idx(a,b,c-2)];
                        }
                    }

                    R_curr[r3idx(a,b,c)] = val;
                }
            }
        }

        for (int i = 0; i < MAX_R * MAX_R * MAX_R; ++i)
            R_next[i] = R_curr[i];
    }

    out = R_next[r3idx(ta, ub, vc)];
}

//---------------------------------------------------------------------------------------

__global__
void coulomb_R_test_kernel(int t, int u, int v, int n,
                           double p, double q,
                           double Px, double Py, double Pz,
                           double Qx, double Qy, double Qz,
                           double* output)
{
    coulomb_R_gpu(t, u, v, n,
                  p, q,
                  Px, Py, Pz,
                  Qx, Qy, Qz, output[0]);
}


double coulomb_R_test_gpu(int t, int u, int v, int n,
                          double p, double q,
                          double Px, double Py, double Pz,
                          double Qx, double Qy, double Qz)
{
    double output = 0.0;
    double* d_output = nullptr;

    CUDA_CHECK(cudaMalloc(&d_output, sizeof(double)));

    coulomb_R_test_kernel<<<1, 1>>>(t, u, v, n,
                                    p, q,
                                    Px, Py, Pz,
                                    Qx, Qy, Qz,
                                    d_output);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(&output,
                          d_output,
                          sizeof(double),
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_output));

    return output;
}


//-------------------------PRIMITIVE TWO-ELECTRON INTEGRAL-------------------------------------

__device__
void PrimitiveRepulsionGPU(const PrimitiveGPU& g1,
                           const PrimitiveGPU& g2,
                           const PrimitiveGPU& g3,
                           const PrimitiveGPU& g4,
                           const BasisFunctionGPU b1,
                           const BasisFunctionGPU b2,
                           const BasisFunctionGPU b3,
                           const BasisFunctionGPU b4,
                           double& val)
{
    int l1 = b1.lx;
    int l2 = b2.lx;
    int l3 = b3.lx;
    int l4 = b4.lx;
    int m1 = b1.ly;
    int m2 = b2.ly;
    int m3 = b3.ly;
    int m4 = b4.ly;
    int n1 = b1.lz;
    int n2 = b2.lz;
    int n3 = b3.lz;
    int n4 = b4.lz;

    double p = g1.alpha + g2.alpha;
    double q = g3.alpha + g4.alpha;

    double Px = (g1.alpha * b1.x + g2.alpha * b2.x) / p;
    double Py = (g1.alpha * b1.y + g2.alpha * b2.y) / p;
    double Pz = (g1.alpha * b1.z + g2.alpha * b2.z) / p;
    double Qx = (g3.alpha * b3.x + g4.alpha * b4.x) / q;
    double Qy = (g3.alpha * b3.y + g4.alpha * b4.y) / q;
    double Qz = (g3.alpha * b3.z + g4.alpha * b4.z) / q;

    val = 0.0;

    for (int t = 0; t <= l1 + l2; ++ t){
        for (int u = 0; u <= m1 + m2; ++ u){
            for (int v = 0; v <= n1 + n2; ++ v){
                for (int tau = 0; tau <= l3 + l4; ++ tau){
                    for (int nu = 0; nu <= m3 + m4; ++ nu){
                        for (int phi = 0; phi <= n3 + n4; ++ phi){

                            double sign = ((tau + nu + phi) % 2 == 0) ? 1.0 : -1.0;
                            double E1, E2, E3, E4, E5, E6, R;

                            hermite_E_gpu(l1, l2, t, g1.alpha, g2.alpha, b1.x, b2.x, E1);
                            hermite_E_gpu(m1, m2, u, g1.alpha, g2.alpha, b1.y, b2.y, E2);
                            hermite_E_gpu(n1, n2, v, g1.alpha, g2.alpha, b1.z, b2.z, E3);
                            hermite_E_gpu(l3, l4, tau, g3.alpha, g4.alpha, b3.x, b4.x, E4);
                            hermite_E_gpu(m3, m4, nu, g3.alpha, g4.alpha, b3.y, b4.y, E5);
                            hermite_E_gpu(n3, n4, phi, g3.alpha, g4.alpha, b3.z, b4.z, E6);

                            coulomb_R_gpu(t + tau, u + nu, v + phi, 0, 
                                          p, q, Px, Py, Pz, Qx, Qy, Qz, R);

                            val += sign * E1 * E2 * E3 * E4 * E5 * E6 * R;
                
                        }
                    }
                }
            }
        }
    }
    
    val *= 2 * pow(M_PI, 2.5) * pow(p + q, -0.5) / (p * q);
}


//-----------------------BASIS LEVEL TWO-ELECTRON INTEGRAL-----------------------------

__device__
void BasisRepulsionGPU(const BasisFunctionGPU& b1,
                       const BasisFunctionGPU& b2,
                       const BasisFunctionGPU& b3,
                       const BasisFunctionGPU& b4,
                       double& eri)
{
    eri = 0.0;

    for (int i = 0; i < b1.nprims; ++ i){
        for (int j = 0; j < b2.nprims; ++ j){
            for (int k = 0; k < b3.nprims; ++ k){
                for (int l = 0; l < b4.nprims; ++ l){

                    PrimitiveGPU g1 = b1.prims[i];
                    PrimitiveGPU g2 = b2.prims[j];
                    PrimitiveGPU g3 = b3.prims[k];
                    PrimitiveGPU g4 = b4.prims[l];

                    double p_eri;
                    PrimitiveRepulsionGPU(g1, g2, g3, g4, 
                                          b1, b2, b3, b4, 
                                          p_eri);

                    eri += g1.coeff * g2.coeff * g3.coeff * g4.coeff *
                           g1.norm * g2.norm * g3.norm * g4.norm * p_eri;

                }
            }
        }
    }
}

//I need to write some comments to help make sure I don't forget the math
//of ERI symmetry and Schwarz screening.

//consider a triangle matrix Mij
//only elemets with indices j <= i matters
//because exhanging the order of basis function within one of the pairs
//doesn't change the value of the ERI element of the quartet.

//(0,0)
//(1,0) (1,1)
//(2,0) (2,1) (2,2)
//(3,0) (3,1) (3,2) (3,3)
//(4,0) (4,1) (4,2) (4,3) (4,4)
//(5,0) (5,1) (5,2) (5,3) (5,4) (5,5)
//....................................
//(K-1,0)..............................(K-1,K-1)

//so the number of unique pairs is K(K+1)/2
//flattening this triangular object, the element
//(m, n) has 1d index p = m(m+1)/2 + n if m, n starts from 0
// and p can be inverted to find m and n

//It's the same idea when dealing with the quartets, I can
//also think of a triangle matrix, Aij, j <= i,
//but now each matrix element is a combinations of two pairs, or a quartet.
//Exchanging the order of two pairs in a quartet also doesn't change the
//ERI element value, so it's still just a triangular matrix.
//for this triangular matrix, the row and column numbers i, j correpond to 
//the flattened indices of the pairs in the last triangular matrix. 
//So I only need to compute the integral of the unique quartets, 
//which has a number of J(J+1)/2 where J = K(K+1)/2 and K is the number
//of basis functions. 

//But actually I can skip some more if I know some of them are
//going to be vanishingly small. Using the Cauchy-Schwarz inequality,
//<f, g> <= sqrt(<f, f><g, g>) = ||f|| ||g||, I can establish an
//upper bound for all ERI elements by just doing a O(K^2) computation
//Here f and g are pairs, consider f = ab where a and b are basis functions.
//Then for all unique pairs ab, compute ||ab||. So if the product of the
//norms of the two pairs is lower than a threshold, I can just skip this
//quartet and set it's ERI element to zero. This is such beautiful math!



//convert 1d flattened triangular indices to 2d (row, col)
__device__
void pair_to_indices(int pid, int& a, int& b)
{
    a = int((sqrt(8.0 * pid + 1.0) - 1.0) * 0.5);

    int tri = a * (a + 1) / 2;

    if (tri > pid) {
        --a;
        tri = a * (a + 1) / 2;
    }

    while ((a + 1) * (a + 2) / 2 <= pid) {
        ++a;
        tri = a * (a + 1) / 2;
    }

    b = pid - tri;
}


__device__
int rank4idx(int i, int j, int k, int l, int K){
    return ((((i * K) + j) * K) + k) * K + l;
}

//----------------UPPER BOUNDS OF ERI ELEMENTS------------------------

__global__
void eri_bounds(int K, double* bounds, BasisFunctionGPU* basis)
{
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    int n_pairs = K * (K + 1) / 2;
    if (pid >= n_pairs) return;

    int mu, nu;
    pair_to_indices(pid, mu, nu);

    if (mu < 0 || mu >= K || nu < 0 || nu >= K) {
        bounds[pid] = -1.0;
        return;
    }

    if (basis[mu].nprims < 0 || basis[mu].nprims > MAX_PRIMS ||
        basis[nu].nprims < 0 || basis[nu].nprims > MAX_PRIMS) {
        bounds[pid] = -2.0;
        return;
    }

    double eri = 0.0;
    BasisRepulsionGPU(basis[mu], basis[nu],
                      basis[mu], basis[nu], eri);

    bounds[pid] = sqrt(fmax(eri, 0.0));
}
//----------------------KERNEL TO COMPUTE ERI--------------------------
//--------------------GPU SIDE OF HYBRID APPROACH--------------------


__global__
void eri_kernel(const BasisFunctionGPU* basis,
                const double* bounds,
                double* unique_eri,
                int K,
                double threshold,
                int start_qid,
                int n_compute)
{
    int local_qid = blockIdx.x * blockDim.x + threadIdx.x;
    if (local_qid >= n_compute) return;

    int qid = start_qid + local_qid;

    int p, q;
    pair_to_indices(qid, p, q);

    if (bounds[p] * bounds[q] < threshold) { //schwarz screening
        unique_eri[local_qid] = 0.0;
        return;
    }

    int mu, nu, lambda, sigma;
    pair_to_indices(p, mu, nu);
    pair_to_indices(q, lambda, sigma);

    BasisRepulsionGPU(basis[mu], basis[nu],
    basis[lambda], basis[sigma], unique_eri[local_qid]);

}



inline void cuda_check(cudaError_t err, const char* msg)
{
    if (err != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(err) << std::endl;
        std::exit(1);
    }
}




// This function computes ERI by letting cpu and gpu do the same thing

std::vector<double> ERI_tensor(
    const std::vector<basis_function>& cpu_basis,
    double split_fraction,
    double threshold
) {
    std::vector<BasisFunctionGPU> h_basis = convert_basis_to_gpu(cpu_basis);

    split_fraction = std::clamp(split_fraction, 0.0, 1.0);

    int K = cpu_basis.size();
    int n_pairs = K * (K + 1) / 2;
    int n_quartets = n_pairs * (n_pairs + 1) / 2;

    int split_qid = int(std::ceil(split_fraction * n_quartets));

    int cpu_count = split_qid;
    int gpu_start = split_qid;
    int gpu_count = n_quartets - split_qid;

    std::vector<double> cpu_unique(cpu_count, 0.0);
    std::vector<double> gpu_unique(gpu_count, 0.0);

    std::vector<double> cpu_bounds = schwarz_bounds(cpu_basis);

    BasisFunctionGPU* d_basis = nullptr;
    double* d_bounds = nullptr;
    double* d_gpu_unique = nullptr;

    cuda_check(cudaMalloc(&d_basis, K * sizeof(BasisFunctionGPU)),
               "cudaMalloc d_basis");

    cuda_check(cudaMalloc(&d_bounds, n_pairs * sizeof(double)),
               "cudaMalloc d_bounds");

    cuda_check(cudaMemcpy(d_basis,
                          h_basis.data(),
                          K * sizeof(BasisFunctionGPU),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy basis H2D");

    int block = 256;

    int grid_bounds = (n_pairs + block - 1) / block;
    eri_bounds<<<grid_bounds, block>>>(K, d_bounds, d_basis);
    cuda_check(cudaGetLastError(), "eri_bounds launch");
    cuda_check(cudaDeviceSynchronize(), "eri_bounds sync");

    if (gpu_count > 0) {
        cuda_check(cudaMalloc(&d_gpu_unique, gpu_count * sizeof(double)),
                   "cudaMalloc d_gpu_unique");

        cuda_check(cudaMemset(d_gpu_unique, 0, gpu_count * sizeof(double)),
                   "cudaMemset d_gpu_unique");
    }

    std::thread cpu_worker([&]() {
        ERI_CPU(
            cpu_basis,
            cpu_bounds,
            cpu_unique,
            cpu_count,
            threshold
        );
    });

    if (gpu_count > 0) {
        int grid_eri = (gpu_count + block - 1) / block;

        eri_kernel<<<grid_eri, block>>>(
            d_basis,
            d_bounds,
            d_gpu_unique,
            K,
            threshold,
            gpu_start,
            gpu_count
        );

        cuda_check(cudaGetLastError(), "eri_kernel launch");
    }

    cpu_worker.join();

    if (gpu_count > 0) {
        cuda_check(cudaDeviceSynchronize(), "eri_kernel sync");

        cuda_check(cudaMemcpy(gpu_unique.data(),
                              d_gpu_unique,
                              gpu_count * sizeof(double),
                              cudaMemcpyDeviceToHost),
                   "cudaMemcpy gpu_unique D2H");
    }

    cudaFree(d_basis);
    cudaFree(d_bounds);
    if (d_gpu_unique) cudaFree(d_gpu_unique);

    cpu_unique.insert(cpu_unique.end(), gpu_unique.begin(), gpu_unique.end());

    return fill_eri_symmetry(cpu_unique, K);
}


//------------------KERNEL AND FUNCTION TO COMPUTE ERI PURELY ON GPU----------------------

__global__
void eri_gpu_kernel(const BasisFunctionGPU* basis,
                       const double* bounds,
                       double* ERI,
                       int K,
                       double threshold)
{
    int qid = blockIdx.x * blockDim.x + threadIdx.x;

    int n_pairs = K * (K + 1) / 2;
    int n_quartets = n_pairs * (n_pairs + 1) / 2;

    if (qid >= n_quartets) return;

    //qid is the flattened quartet index
    //p, q are the flattened pair indices
    int p, q;
    pair_to_indices(qid, p, q);  // p >= q

    //here p and q are flattened indices of each pair in the triangle
    //because each of p and q are the row and column number of
    //a unique quartet, so the row and column numbers are the flattened 
    //1d index of the two pairs in the quartet.

    if (bounds[p] * bounds[q] < threshold) return; //Schwarz screening

    //mu, nu, lambda, sigma are the original 2d indices of the two pairs.
    int mu, nu, lambda, sigma;
    pair_to_indices(p, mu, nu);
    pair_to_indices(q, lambda, sigma);

    //compute the integral on the surviving quartets only.
    double val;
    BasisRepulsionGPU(basis[mu], basis[nu],
    basis[lambda], basis[sigma], val);

    ERI[rank4idx(mu, nu, lambda, sigma, K)] = val;
    ERI[rank4idx(nu, mu, lambda, sigma, K)] = val;
    ERI[rank4idx(mu, nu, sigma, lambda, K)] = val;
    ERI[rank4idx(nu, mu, sigma, lambda, K)] = val;
    ERI[rank4idx(sigma, lambda, mu, nu, K)] = val;
    ERI[rank4idx(lambda, sigma, mu, nu, K)] = val;
    ERI[rank4idx(sigma, lambda, nu, mu, K)] = val;
    ERI[rank4idx(lambda, sigma, nu, mu, K)] = val;
}


std::vector<double> ERI_GPU(
    const std::vector<basis_function>& cpu_basis,
    double threshold
) {

    std::vector<BasisFunctionGPU> h_basis = convert_basis_to_gpu(cpu_basis); 
    int K = h_basis.size();
    int n_pairs = K * (K + 1) / 2;
    int n_quartets = n_pairs * (n_pairs + 1) / 2;
    int eri_size = K * K * K * K;

    BasisFunctionGPU* d_basis = nullptr;
    double* d_bounds = nullptr;
    double* d_ERI = nullptr;

    cuda_check(cudaMalloc(&d_basis, K * sizeof(BasisFunctionGPU)),
               "cudaMalloc d_basis");

    cuda_check(cudaMalloc(&d_bounds, n_pairs * sizeof(double)),
               "cudaMalloc d_bounds");

    cuda_check(cudaMalloc(&d_ERI, eri_size * sizeof(double)),
               "cudaMalloc d_ERI");

    cuda_check(cudaMemcpy(d_basis,
                          h_basis.data(),
                          K * sizeof(BasisFunctionGPU),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy basis H2D");

    cuda_check(cudaMemset(d_ERI, 0, eri_size * sizeof(double)),
               "cudaMemset d_ERI");

    int block = 256;

    int grid_bounds = (n_pairs + block - 1) / block;
    eri_bounds<<<grid_bounds, block>>>(K, d_bounds, d_basis);
    cuda_check(cudaGetLastError(), "eri_bounds launch");
    cuda_check(cudaDeviceSynchronize(), "eri_bounds sync");

    int grid_quartets = (n_quartets + block - 1) / block;

    eri_gpu_kernel<<<grid_quartets, block>>>(
        d_basis,
        d_bounds,
        d_ERI,
        K,
        threshold
    );
    cuda_check(cudaGetLastError(), "eri_unique_kernel launch");
    cuda_check(cudaDeviceSynchronize(), "eri_unique_kernel sync");


    std::vector<double> h_ERI(eri_size);

    cuda_check(cudaMemcpy(h_ERI.data(),
                          d_ERI,
                          eri_size * sizeof(double),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy ERI D2H");

    cudaFree(d_basis);
    cudaFree(d_bounds);
    cudaFree(d_ERI);

    return h_ERI;
}


