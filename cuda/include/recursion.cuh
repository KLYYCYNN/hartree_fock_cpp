#pragma once
#include <cuda_runtime.h>


constexpr int MAX_L = 3;      // up to f
constexpr int MAX_E = 2 * MAX_L + 1; // 7
constexpr int MAX_R_ORDER = 4 * MAX_L;   // ffff max = 12
constexpr double PI_GPU = 3.141592653589793238462643383279502884;


__device__ __host__
void hermite_E_gpu(int i, int j,
                   double alpha, double beta,
                   double A, double B,
                   double Eout[MAX_E])
{

    if (i > MAX_L || j > MAX_L || i + j >= MAX_E) {
        for (int tt = 0; tt < MAX_E; ++tt) {
            Eout[tt] = 0.0;
        }
    return;
    }

    double E[MAX_L + 1][MAX_L + 1][MAX_E];

    for (int a = 0; a <= MAX_L; ++a)
        for (int b = 0; b <= MAX_L; ++b)
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




constexpr int N_R =
    (MAX_R_ORDER + 1) *
    (MAX_R_ORDER + 2) *
    (MAX_R_ORDER + 3) / 6;



__device__ __host__
inline bool valid_r_index(int a, int b, int c)
{
    return a >= 0 && b >= 0 && c >= 0 &&
           a + b + c <= MAX_R_ORDER;
}



__device__ __host__
inline int packed_r_index(int a, int b, int c)
{
    int s = a + b + c;

    int offset_s = s * (s + 1) * (s + 2) / 6;

    int offset_a = 0;
    for (int aa = 0; aa < a; ++aa) {
        offset_a += (s - aa + 1);
    }

    int offset_b = b;

    return offset_s + offset_a + offset_b;
}




__device__ __host__
void boys_gpu(int m, double T, double& F)
{

    if (T < 1e-10) {
        F = 1.0 / (2.0 * m + 1.0);
        return;
    }

    // small-T series:
    // F_m(T) = sum_k (-T)^k / (k! (2m + 2k + 1))
    if (T < 1e-4) {
        double term = 1.0;
        double sum = 0.0;

        for (int k = 0; k < 12; ++k) {
            sum += term / (2.0 * m + 2.0 * k + 1.0);
            term *= -T / static_cast<double>(k + 1);
        }

        F = sum;
        return;
    }


    // normal branch: start from F0 and recurse upward
    double sqrtT = sqrt(T);
    F = 0.5 * sqrt(PI_GPU / T) * erf(sqrtT);

    double expT = exp(-T);

    for (int k = 0; k < m; ++k) {
        F = ((2.0 * k + 1.0) * F - expT) / (2.0 * T);
    }
}




__device__ __host__
void coulomb_R_gpu(int tx, int ty, int tz, int n,
                   double p, double q,
                   double Px, double Py, double Pz,
                   double Qx, double Qy, double Qz,
                   double& out)
{
    int max_order = tx + ty + tz + n;

    if (max_order > MAX_R_ORDER ||
        !valid_r_index(tx, ty, tz)) {
        out = 0.0;
        return;
    }

    double rho = p * q / (p + q);

    double X = Px - Qx;
    double Y = Py - Qy;
    double Z = Pz - Qz;

    double T = rho * (X*X + Y*Y + Z*Z);

    double Rnext[N_R];
    double Rcurr[N_R];

    for (int i = 0; i < N_R; ++i) {
        Rnext[i] = 0.0;
        Rcurr[i] = 0.0;
    }

    double F;
    boys_gpu(max_order, T, F);

    Rnext[packed_r_index(0, 0, 0)] =
        pow(-2.0 * rho, max_order) * F;

    for (int m = max_order - 1; m >= n; --m) {

        for (int i = 0; i < N_R; ++i) {
            Rcurr[i] = 0.0;
        }

        double Fm;
        boys_gpu(m, T, Fm);

        Rcurr[packed_r_index(0, 0, 0)] =
            pow(-2.0 * rho, m) * Fm;

        for (int s = 1; s <= tx + ty + tz; ++s) {
            for (int a = 0; a <= s; ++a) {
                for (int b = 0; b <= s - a; ++b) {

                    int c = s - a - b;

                    if (a > tx || b > ty || c > tz) continue;

                    double val = 0.0;

                    if (a > 0) {
                        val = X * Rnext[packed_r_index(a - 1, b, c)];

                        if (a >= 2) {
                            val += (a - 1) *
                                   Rnext[packed_r_index(a - 2, b, c)];
                        }
                    }
                    else if (b > 0) {
                        val = Y * Rnext[packed_r_index(a, b - 1, c)];

                        if (b >= 2) {
                            val += (b - 1) *
                                   Rnext[packed_r_index(a, b - 2, c)];
                        }
                    }
                    else if (c > 0) {
                        val = Z * Rnext[packed_r_index(a, b, c - 1)];

                        if (c >= 2) {
                            val += (c - 1) *
                                   Rnext[packed_r_index(a, b, c - 2)];
                        }
                    }

                    Rcurr[packed_r_index(a, b, c)] = val;
                }
            }
        }

        for (int i = 0; i < N_R; ++i) {
            Rnext[i] = Rcurr[i];
        }
    }

    out = Rnext[packed_r_index(tx, ty, tz)];
}


