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

inline
size_t pair_index(int a, int b)
{
    if (a < b) std::swap(a, b);

    return static_cast<size_t>(a) * (a + 1) / 2 + b;
}


inline
size_t eri_index(int i, int j,
                 int k, int l)
{
    size_t IJ = pair_index(i, j);
    size_t KL = pair_index(k, l);

    if (IJ < KL) std::swap(IJ, KL);

    return IJ * (IJ + 1) / 2 + KL;
}


inline
std::pair<int, int> quartet_to_pairs(size_t qid)
{
    int pid1, pid2;
    pid1 = int((sqrt(8.0 * static_cast<double>(qid) + 1.0) - 1.0) * 0.5);

    size_t tri =
        static_cast<size_t>(pid1) * (pid1 + 1) / 2;

    if (tri > qid) {
        --pid1;
        tri = static_cast<size_t>(pid1) * (pid1 + 1) / 2;
    }

    while (static_cast<size_t>(pid1 + 1) * (pid1 + 2) / 2 <= qid) {
        ++pid1;
        tri = static_cast<size_t>(pid1) * (pid1 + 1) / 2;
    }

    pid2 = static_cast<int>(qid - tri);

    return {pid1, pid2};
}
