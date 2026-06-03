#pragma once
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <stdexcept>
#include <cmath>
#include <deque>
#include "indexing.hpp"


Eigen::MatrixXd vec2mat(const std::vector<double>& vec, int rows, int cols) {
    // 1. Safety check for dimensions
    if (vec.size() != static_cast<size_t>(rows * cols)) {
        throw std::runtime_error("Vector size does not match provided dimensions.");
    }

    // 2. Map the vector data to a Row-Major view and return it
    // This automatically copies the mapped data into the returned MatrixXd
    return Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
        vec.data(), rows, cols
    );
}



std::vector<double> mat2vec(const Eigen::MatrixXd& mat) {
    // 1. Force evaluation into a RowMajor matrix layout
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> rowMajorMat = mat;

    // 2. Now .data() is valid and points to a row-by-row buffer
    return std::vector<double>(
        rowMajorMat.data(), 
        rowMajorMat.data() + rowMajorMat.size()
    );
}

std::vector<double> vec2vec(const Eigen::VectorXd& ev) {
    // Construct std::vector using the pointer to the start and end of Eigen data
    return std::vector<double>(ev.data(), ev.data() + ev.size());
}

double vector_dot(const std::vector<double>& r1,
                  const std::vector<double>& r2)
{

    if (r1.size() != r2.size()){
        throw std::runtime_error(R"(cannot take scalar product between
                                    two vectors of different sizes)");
    }

    double prod = 0.0;

    for (int i = 0; i < static_cast<int>(r1.size()); ++i){
        prod += r1[i] * r2[i];
    }

    return prod;
}


std::vector<double> matrix_add(const std::vector<double>& A, 
                               const std::vector<double>& B)
{
    int n1 = static_cast<int>(A.size());
    int n2 = static_cast<int>(B.size());

    if (n1 != n2){
        throw std::runtime_error("Cannot add vectors with difference sizes");
    }

    std::vector<double> sum(n1);

    for (int i = 0; i < n1; ++ i){
        sum[i] = A[i] + B[i];
    }

    return sum;
}


std::vector<double> matrix_scale(const std::vector<double>& A, 
                                 double lambda)
{

    int n = static_cast<int>(A.size());
    std::vector<double> B(n);

    for (int i = 0; i < n; ++ i){
        B[i] = lambda * A[i];
    }

    return B;

}


//contracts the rank-4 electron interaction tensor with density matrix,
//compute J and K values and construct the electron-electron interaction matrix

std::vector<double> repulsion_matrix(const std::vector<double>& density_matrix,
                                     const std::vector<double>& unique_eri){

    int n_basis = static_cast<int>
                  (std::sqrt(static_cast<int>(density_matrix.size())));

    std::vector<double> mat(density_matrix.size());

    for (int i=0; i<n_basis; ++i){
        for (int j=0; j<n_basis; ++j){

            mat[index2d(i, j, n_basis)] = 0.0;

            for (int k=0; k<n_basis; ++k){
                for (int l=0; l<n_basis; ++l){

                    double J = density_matrix[index2d(k, l, n_basis)] *
                               unique_eri[eri_index(i, j, k, l)];

                    double K = density_matrix[index2d(k, l, n_basis)] *
                               unique_eri[eri_index(i, k, j, l)];

                    mat[index2d(i, j, n_basis)] += J - 0.5*K;

                }
            }
        }
    }
    return mat;
}


Eigen::MatrixXd Xmatrix(const std::vector<double>& overlap_matrix){
//orthogonalization transformation
    Eigen::MatrixXd S = vec2mat(overlap_matrix,
                                (int) pow(overlap_matrix.size(), 0.5),
                                (int) pow(overlap_matrix.size(), 0.5));

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenS(S); 

    Eigen::MatrixXd U = eigenS.eigenvectors();
    Eigen::VectorXd s = eigenS.eigenvalues();
    Eigen::MatrixXd s_sqrt_repc =
    s.array().inverse().sqrt().matrix().asDiagonal();
    Eigen::MatrixXd X = U * s_sqrt_repc * U.transpose();

    return X;
}

//return the eigenvalue and eigenvectors of the fock matrix
std::pair<std::vector<double>, std::vector<double>>
EigProblem(const std::vector<double>& fock_matrix,
           const Eigen::MatrixXd X_matrix){

    Eigen::MatrixXd X = X_matrix;

    //transform the Fock matrix
    Eigen::MatrixXd F = X.transpose() * vec2mat(fock_matrix,
                        (int) sqrt(fock_matrix.size()),
                        (int) sqrt(fock_matrix.size())) * X;
    
    //eigen vector and eigenvalues of fock matrix
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> CE(F);                  
    Eigen::MatrixXd C_prime = CE.eigenvectors();
    Eigen::VectorXd e = CE.eigenvalues();

    std::pair<std::vector<double>, std::vector<double>>
    result = {mat2vec(X*C_prime), vec2vec(e)};

    return result;
}


//compute density matrix from the eigenvectors of fock matrix
std::vector<double> update_Pmatrix(const std::vector<double>& C,
                                   int n_electron){

    int n_basis = (int) std::sqrt(C.size());
    int n_occ = n_electron / 2;
    std::vector<double> P(C.size());

    for (int i=0; i<n_basis; ++i){
        for (int j=0; j<n_basis; ++j){
            for (int k=0; k<n_occ; ++k){

                P[index2d(i, j, n_basis)] += 2 *
                C[index2d(i, k, n_basis)] *
                C[index2d(j, k, n_basis)];
            }
        }
    }
    return P;
}


std::vector<double> update_density_uhf(const std::vector<double>& C,
                                       int n_occ){

    int n_basis = (int) std::sqrt(C.size());
    std::vector<double> P(C.size());

    for (int i=0; i<n_basis; ++i){
        for (int j=0; j<n_basis; ++j){
            for (int k=0; k<n_occ; ++k){

                P[index2d(i, j, n_basis)] +=
                C[index2d(i, k, n_basis)] *
                C[index2d(j, k, n_basis)];
            }
        }
    }
    return P;
}


std::vector<double> 
coulomb_matrix(const std::vector<double>& density1,
               const std::vector<double>& density2,
               const std::vector<double>& unique_eri)
{
    if (density1.size() != density2.size()){
        throw std::runtime_error("density matrices sizes mismatch");
    }

    size_t eri_size = unique_eri.size();
    int K2 = static_cast<int>(density1.size());
    int K = static_cast<int>(std::sqrt(K2));

    if (K * K != K2){
        throw std::runtime_error("density matrix is not square");
    }

    size_t n_pairs = K * (K + 1) / 2;

    if (n_pairs * (n_pairs + 1) / 2 != eri_size){
        throw std::runtime_error(R"(DimensionError: mismatch between
                               density matrix and ERI sizes)");
    }

    std::vector<double> J_matrix(K2, 0.0);

    for (int i = 0; i < K; ++i){
        for (int j = 0; j < K; ++j){
            for (int k = 0; k < K; ++k){
                for (int l = 0; l < K; ++l){

                    J_matrix[index2d(i, j, K)] += 
                    (density1[index2d(k, l, K)]
                     + density2[index2d(k, l, K)])
                    * unique_eri[eri_index(i, j, k, l)];

                }
            }
        }
    }

    return J_matrix;
}


std::vector<double> 
exchange_matrix(const std::vector<double>& density_matrix,
                const std::vector<double>& unique_eri)
{
    size_t eri_size = unique_eri.size();
    int K2 = static_cast<int>(density_matrix.size());
    int K = static_cast<int>(std::sqrt(K2));

    if (K * K != K2){
        throw std::runtime_error("density matrix is not square");
    }

    size_t n_pairs = K * (K + 1) / 2;

    if (n_pairs * (n_pairs + 1) / 2 != eri_size){
        throw std::runtime_error(R"(DimensionError: mismatch between
                               density matrix and ERI sizes)");
    }

    std::vector<double> K_matrix(K2, 0.0);

    for (int i = 0; i < K; ++i){
        for (int j = 0; j < K; ++j){
            for (int k = 0; k < K; ++k){
                for (int l = 0; l < K; ++l){

                    K_matrix[index2d(i, j, K)] += 
                    density_matrix[index2d(k, l, K)]
                    * unique_eri[eri_index(i, k, j, l)];

                }
            }
        }
    }

    return K_matrix;
}


double RMSD(const std::vector<double>& A, 
            const std::vector<double>& B)
{
    int n1 = static_cast<int>(A.size());
    int n2 = static_cast<int>(B.size());

    if (n1 != n2){
        throw std::runtime_error("vectors have different sizes");
    }

    double tot_diff2 = 0.0;

    for (int i = 0; i < n1; ++i){
        double diff = A[i] - B[i];        
        tot_diff2 += diff * diff;
    }

    return std::sqrt(tot_diff2 / n1);
}

double vector_norm(const std::vector<double>& vec){

    double norm = 0.0;

    for (double xi : vec){
        norm += xi * xi;
    }

    return std::sqrt(norm);
}


std::vector<double> fock_matrix_uhf(const std::vector<double>& core_ham,
                                    const std::vector<double>& coulomb,
                                    const std::vector<double>& exchange)
{
    int n1 = static_cast<int>(core_ham.size());
    int n2 = static_cast<int>(coulomb.size());
    int n3 = static_cast<int>(exchange.size());

    if (n1 != n2 || n1 != n3 || n2 != n3){
        throw std::runtime_error(R"(DimensionError: matrix dimension
                            inconsistency encountered when building
                            the fock matrix)");
    }

    std::vector<double> fock(n1);

    for (int i = 0; i < n1; ++i){
        fock[i] = core_ham[i] + coulomb[i] - exchange[i];
    }

    return fock;
}


std::vector<double> error_matrix(const std::vector<double>& density,
                                 const std::vector<double>& fock,
                                 const std::vector<double>& overlap)
{
    size_t K = static_cast<size_t>(std::sqrt(density.size()));
    int int_K = static_cast<int>(K);

    if (fock.size() != K * K || overlap.size() != K * K){
        throw std::runtime_error(R"(mismatch between matrix sizes)");
    }

    if (density.size() != K * K){
        throw std::runtime_error(R"(density matrix is not a square matrix)");
    }


    Eigen::MatrixXd S = vec2mat(overlap, int_K , int_K);
    Eigen::MatrixXd P = vec2mat(density, int_K , int_K);
    Eigen::MatrixXd F = vec2mat(fock, int_K , int_K);

    Eigen::MatrixXd R = F * P * S - S * P * F;

    return mat2vec(R);

}


void
update_matrix_history(const std::vector<double>& matrix,
                      std::deque<std::vector<double>>& history,
                      int max_history)
{
    int hist_size = static_cast<int>(history.size());

    if (hist_size > max_history){
        throw std::runtime_error("Stored too many matrices");
    }

    if (hist_size == max_history){

        history.pop_front();
        history.push_back(matrix);

    } else{
        history.push_back(matrix);
    }

}


std::vector<double>
aug_pulay_matrix(const std::deque<std::vector<double>>& error_vectors)
{
    int n = static_cast<int>(error_vectors.size());

    std::vector<double> aug_B((n + 1) * (n + 1), 0.0);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            aug_B[index2d(i, j, n + 1)] =
                vector_dot(error_vectors[i], error_vectors[j]);
        }
    }

    for (int i = 0; i < n; ++i) {
        aug_B[index2d(i, n, n + 1)] = -1.0;
        aug_B[index2d(n, i, n + 1)] =  1.0;
    }

    aug_B[index2d(n, n, n + 1)] = 0.0;

    return aug_B;
}


std::vector<double>
diis_coefficients(const std::vector<double>& aug_B, int n)
{
    // aug_B has size (n+1)*(n+1), row-major
    Eigen::MatrixXd A = vec2mat(aug_B, n + 1, n + 1);

    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(n + 1);
    rhs(n) = 1.0;   // because your last row is +1 +1 +1 ...

    Eigen::VectorXd x = A.fullPivLu().solve(rhs);

    std::vector<double> coeffs(n);
    for (int i = 0; i < n; ++i) {
        coeffs[i] = x(i);
    }

    return coeffs;
}


std::vector<double> 
vector_lin_comb(const std::deque<std::vector<double>>& vectors,
                const std::vector<double>& coeffs)
{
    if (vectors.empty()) {
        throw std::runtime_error("Cannot combine empty vector history.");
    }

    size_t n1 = vectors.size();
    size_t n2 = coeffs.size();

    if (n1 != n2){
        throw std::runtime_error
              (R"(Different number of vectors and coefficients)");
    }

    int n = static_cast<int>(n1);
    int vec_dim = static_cast<int>(vectors[0].size());
    size_t vec_size = vectors[0].size();
    std::vector<double> combo(vec_dim, 0.0);

    for (int i = 0; i < n; ++i){

        const std::vector<double>& vec_i = vectors[i];

        if (vec_i.size() != vec_size){
            throw std::runtime_error(R"(cannot combine vectors with 
                                     different sizes)");
            }

        for (int j = 0; j < vec_dim; ++j){
            combo[j] += coeffs[i] * vec_i[j];
        }

    }

    return combo;
}

std::vector<double> 
fock_diis(const std::deque<std::vector<double>>& fock_history,
          const std::deque<std::vector<double>>& error_vectors)
{

    size_t hist_size = fock_history.size();

    if (hist_size == 0) {
        throw std::runtime_error("Empty DIIS history.");
    }

    if (error_vectors.size() != hist_size){
        throw std::runtime_error(R"(Stored different numbers of
                                 Fock matrices and error vectors)");
    }


    std::vector<double> aug_B = aug_pulay_matrix(error_vectors);
    std::vector<double> coeffs = 
                diis_coefficients(aug_B, static_cast<int>(hist_size));

    return vector_lin_comb(fock_history, coeffs);
}