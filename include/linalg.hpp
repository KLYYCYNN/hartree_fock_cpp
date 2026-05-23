#pragma once
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <cmath>
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




//contracts the rank-4 electron interaction tensor with density matrix,
//compute J and K values and construct the electron-electron interaction matrix

std::vector<double> repulsion_matrix(const std::vector<double>& density_matrix,
                                     const std::vector<double>& ERI){

    int n_basis = (int) pow(density_matrix.size(), 0.5);
    std::vector<double> mat(density_matrix.size());

    for (int i=0; i<n_basis; ++i){
        for (int j=0; j<n_basis; ++j){

            mat[ index2d(i, j, n_basis) ] = 0.0;

            for (int k=0; k<n_basis; ++k){
                for (int l=0; l<n_basis; ++l){

                    double J = density_matrix[index2d(k, l, n_basis)] *
                               ERI[index4d(i, j, k, l, n_basis)];

                    double K = density_matrix[index2d(k, l, n_basis)] *
                               ERI[index4d(i, k, j, l, n_basis)];

                    mat[ index2d(i, j, n_basis) ] += J - 0.5*K;

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