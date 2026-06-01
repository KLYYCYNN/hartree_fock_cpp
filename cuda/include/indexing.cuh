#pragma once
#include <cuda_runtime.h>

__device__ __host__
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


__device__ __host__
void quartet_to_pairs(size_t qid, int& pid1, int& pid2)
{
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
}


__device__ __host__
int rank4idx(int i, int j, int k, int l, int K){
    return ((((i * K) + j) * K) + k) * K + l;
}


__device__ __host__
int rank2idx(int i, int j, int K){
    return i * K + j;
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