#pragma once
#include <vector>

inline
int index2d(int i, int j, int n){
    return  i*n + j;
}

inline
int index3d(int mu, int nu, int lambda, int n){
    return (mu*n + nu)*n + lambda;
}

inline
int index4d(int mu, int nu, int lambda, int sigma, int K){
    return ((mu*K + nu)*K + lambda)*K + sigma;
}

inline
std::vector<int> inv_index4d(int index, int K) {
    int sigma = index % K;
    index /= K;
    
    int lambda = index % K;
    index /= K;
    
    int nu = index % K;
    int mu = index / K; // The remaining quotient is mu
    
    return {mu, nu, lambda, sigma};
}


