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
void inv_index4d(int index, int K, 
                 int& mu, int& nu, int& lambda, int& sigma) {

    sigma = index % K;
    index /= K;
    
    lambda = index % K;
    index /= K;
    
    nu = index % K;
    mu = index / K; 
}

