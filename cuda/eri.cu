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
double boys_gpu(int m, double T)
{
    if (T < 1e-8) {
        return 1.0 / (2.0 * m + 1.0);
    }

    double F = 0.5 * sqrt(M_PI / T) * erf(sqrt(T));

    for (int k = 0; k < m; ++k) {
        F = ((2.0 * k + 1.0) * F - exp(-T)) / (2.0 * T);
    }

    return F;
}


__global__
void boys_test_kernel(double* output)
{
    output[0] = boys_gpu(0, 1.0);
    output[1] = boys_gpu(1, 1.0);
    output[2] = boys_gpu(2, 1.0);
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

constexpr int MAX_E = 7;


//--------------------HERMITE RECURSION-------------------------

__device__
void hermite_E_gpu(int i, int j,
                   double alpha, double beta,
                   double A, double B,
                   double Eout[MAX_E])
{
    double E[3][3][MAX_E];

    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
            for (int t = 0; t < MAX_E; ++t)
                E[a][b][t] = 0.0;

    double p = alpha + beta;
    double q = alpha * beta / p;
    double AB = A - B;

    E[0][0][0] = exp(-q * AB * AB);

    for (int ia = 1; ia <= i; ++ia) {
        for (int tt = 0; tt <= ia; ++tt) {
            double term1 = (tt > 0) ? E[ia - 1][0][tt - 1] / (2.0 * p) : 0.0;
            double term2 = -(beta / p) * AB * E[ia - 1][0][tt];
            double term3 = (tt + 1 < MAX_E) ? (tt + 1) * E[ia - 1][0][tt + 1] : 0.0;

            E[ia][0][tt] = term1 + term2 + term3;
        }
    }

    for (int ia = 0; ia <= i; ++ia) {
        for (int jb = 1; jb <= j; ++jb) {
            for (int tt = 0; tt <= ia + jb; ++tt) {
                double term1 = (tt > 0) ? E[ia][jb - 1][tt - 1] / (2.0 * p) : 0.0;
                double term2 = +(alpha / p) * AB * E[ia][jb - 1][tt];
                double term3 = (tt + 1 < MAX_E) ? (tt + 1) * E[ia][jb - 1][tt + 1] : 0.0;

                E[ia][jb][tt] = term1 + term2 + term3;
            }
        }
    }

    for (int tt = 0; tt < MAX_E; ++tt)
        Eout[tt] = 0.0;

    for (int tt = 0; tt <= i + j; ++tt)
        Eout[tt] = E[i][j][tt];
}

//-----------------------------------------------------------------------------

__global__
void hermite_test_kernel(int i, int j,
                         double alpha, double beta,
                         double A, double B,
                         double* E_array)
{
    hermite_E_gpu(i, j, alpha, beta, A, B, E_array);
}


std::vector<double> hermite_E_test(int i, int j,
                                   double alpha, double beta,
                                   double A, double B)
{
    int n = i + j + 1;

    std::vector<double> output(n);

    double* d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_output, n * sizeof(double)));

    hermite_test_kernel<<<1, 1>>>(i, j, alpha, beta, A, B, d_output);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(output.data(),
                          d_output,
                          n * sizeof(double),
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_output));

    return output;
}


//------------------------COULOMB R RECURSION------------------------------------------------

constexpr int MAX_R = 9;  // enough for up to d-functions: t+u+v <= 8

__device__
int ridx(int a, int b, int c, int m)
{
    return (((a * MAX_R + b) * MAX_R + c) * MAX_R + m);
}


__device__
double coulomb_R_gpu(int ta, int ub, int vc, int n,
                     double p, double q,
                     double Px, double Py, double Pz,
                     double Qx, double Qy, double Qz)
{
    double rho = p * q / (p + q);

    double X = Px - Qx;
    double Y = Py - Qy;
    double Z = Pz - Qz;

    double T = rho * (X*X + Y*Y + Z*Z);

    double R[MAX_R * MAX_R * MAX_R * MAX_R];

    for (int i = 0; i < MAX_R * MAX_R * MAX_R * MAX_R; ++i)
        R[i] = 0.0;

    int max_m = ta + ub + vc + n;

    for (int m = 0; m <= max_m; ++m) {
        R[ridx(0,0,0,m)] = pow(-2.0 * rho, m) * boys_gpu(m, T);
    }

    for (int m = max_m - 1; m >= n; --m) {
        for (int a = 0; a <= ta; ++a) {
            for (int b = 0; b <= ub; ++b) {
                for (int c = 0; c <= vc; ++c) {

                    if (a == 0 && b == 0 && c == 0) continue;

                    double value = 0.0;

                    if (a > 0) {
                        value = (a - 1) * ((a >= 2) ? R[ridx(a-2,b,c,m+1)] : 0.0)
                              + X * R[ridx(a-1,b,c,m+1)];
                    }
                    else if (b > 0) {
                        value = (b - 1) * ((b >= 2) ? R[ridx(a,b-2,c,m+1)] : 0.0)
                              + Y * R[ridx(a,b-1,c,m+1)];
                    }
                    else if (c > 0) {
                        value = (c - 1) * ((c >= 2) ? R[ridx(a,b,c-2,m+1)] : 0.0)
                              + Z * R[ridx(a,b,c-1,m+1)];
                    }

                    R[ridx(a,b,c,m)] = value;
                }
            }
        }
    }

    return R[ridx(ta, ub, vc, n)];
}

//---------------------------------------------------------------------------------------

__global__
void coulomb_R_test_kernel(int t, int u, int v, int n,
                           double p, double q,
                           double Px, double Py, double Pz,
                           double Qx, double Qy, double Qz,
                           double* output)
{
    output[0] = coulomb_R_gpu(t, u, v, n,
                              p, q,
                              Px, Py, Pz,
                              Qx, Qy, Qz);
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
double PrimitiveRepulsionGPU(const PrimitiveGPU& g1,
                             const PrimitiveGPU& g2,
                             const PrimitiveGPU& g3,
                             const PrimitiveGPU& g4,
                             const BasisFunctionGPU b1,
                             const BasisFunctionGPU b2,
                             const BasisFunctionGPU b3,
                             const BasisFunctionGPU b4)
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


    double E1[MAX_E], E2[MAX_E], E3[MAX_E];
    double E4[MAX_E], E5[MAX_E], E6[MAX_E];

    hermite_E_gpu(l1, l2, g1.alpha, g2.alpha, b1.x, b2.x, E1);
    hermite_E_gpu(m1, m2, g1.alpha, g2.alpha, b1.y, b2.y, E2);
    hermite_E_gpu(n1, n2, g1.alpha, g2.alpha, b1.z, b2.z, E3);
    hermite_E_gpu(l3, l4, g3.alpha, g4.alpha, b3.x, b4.x, E4);
    hermite_E_gpu(m3, m4, g3.alpha, g4.alpha, b3.y, b4.y, E5);
    hermite_E_gpu(n3, n4, g3.alpha, g4.alpha, b3.z, b4.z, E6);

    double eri = 0.0;

    for (int t = 0; t <= l1 + l2; ++ t){
        for (int u = 0; u <= m1 + m2; ++ u){
            for (int v = 0; v <= n1 + n2; ++ v){
                for (int tau = 0; tau <= l3 + l4; ++ tau){
                    for (int nu = 0; nu <= m3 + m4; ++ nu){
                        for (int phi = 0; phi <= n3 + n4; ++ phi){

                            double sign = ((tau + nu + phi) % 2 == 0) ? 1.0 : -1.0;

                            eri += sign *
                                   E1[t] * E2[u] * E3[v] *
                                   E4[tau] * E5[nu] * E6[phi] *
                                   coulomb_R_gpu(t + tau, u + nu, v + phi,
                                                 0, p, q,
                                                 Px, Py, Pz,
                                                 Qx, Qy, Qz);

                        }
                    }
                }
            }
        }
    }
    return 2 * pow(M_PI, 2.5) * pow(p + q, -0.5) * eri / (p * q);
}


//-----------------------BASIS LEVEL TWO-ELECTRON INTEGRAL-----------------------------

__device__
double BasisRepulsionGPU(const BasisFunctionGPU& b1,
                         const BasisFunctionGPU& b2,
                         const BasisFunctionGPU& b3,
                         const BasisFunctionGPU& b4)
{
    double eri = 0.0;

    for (int i = 0; i < b1.nprims; ++ i){
        for (int j = 0; j < b2.nprims; ++ j){
            for (int k = 0; k < b3.nprims; ++ k){
                for (int l = 0; l < b4.nprims; ++ l){

                    PrimitiveGPU g1 = b1.prims[i];
                    PrimitiveGPU g2 = b2.prims[j];
                    PrimitiveGPU g3 = b3.prims[k];
                    PrimitiveGPU g4 = b4.prims[l];

                    eri += g1.coeff * g2.coeff * g3.coeff * g4.coeff *
                           g1.norm * g2.norm * g3.norm * g4.norm * 
                           PrimitiveRepulsionGPU(g1, g2, g3, g4, b1, b2, b3, b4);

                }
            }
        }
    }
    return eri;
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
void pair_to_indices(int pid, int& a, int& b) {
    a = int((sqrt(8.0 * pid + 1.0) - 1.0) / 2.0);
    b = pid - a * (a + 1) / 2;
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

    // invert triangular index:
    // find mu such that mu(mu+1)/2 <= pid < (mu+1)(mu+2)/2
    int mu, nu;
    pair_to_indices(pid, mu, nu);

    // compute (mu nu | mu nu)
    double eri = BasisRepulsionGPU(basis[mu], basis[nu],
                                   basis[mu], basis[nu]);

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

    if (bounds[p] * bounds[q] < threshold) {
        unique_eri[local_qid] = 0.0;
        return;
    }

    int mu, nu, lambda, sigma;
    pair_to_indices(p, mu, nu);
    pair_to_indices(q, lambda, sigma);

    unique_eri[local_qid] = BasisRepulsionGPU(
        basis[mu], basis[nu],
        basis[lambda], basis[sigma]
    );
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
    double val = BasisRepulsionGPU(basis[mu], basis[nu],
                                   basis[lambda], basis[sigma]);

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


