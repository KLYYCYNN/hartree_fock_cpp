#pragma once
#include "basis.hpp"
#include "indexing.hpp"
#include <cmath>
#include <vector>
#include <iostream>
#include <stdexcept>


//------------------------OVERLAP---------------------------------
//using recursive relation to compute 1d overlap involving
//arbitrary angular momentum values
inline
double overlap_1d(const primitive_gaussian& ga,
                  const primitive_gaussian& gb,
                  int l,
                  int m,
                  double A,
                  double B){

    // boundary condition
    if (l < 0 || m < 0)
        return 0.0;

    double alpha = ga.alpha;
    double beta  = gb.alpha;

    double p  = alpha + beta;
    double mu = alpha * beta / p;

    double P = (alpha * A + beta * B) / p;

    // base case
    if (l == 0 && m == 0){
        return std::sqrt(M_PI / p)
             * std::exp(-mu * (A - B) * (A - B));
    }

    // recursion on l
    if (l > 0){
        return (P - A) * overlap_1d(ga, gb, l - 1, m, A, B)
             + ((l - 1) / (2.0 * p))
               * overlap_1d(ga, gb, l - 2, m, A, B)
             + (m / (2.0 * p))
               * overlap_1d(ga, gb, l - 1, m - 1, A, B);
    }

    // recursion on m
    return (P - B) * overlap_1d(ga, gb, l, m - 1, A, B)
         + (l / (2.0 * p))
           * overlap_1d(ga, gb, l - 1, m - 1, A, B)
         + ((m - 1) / (2.0 * p))
           * overlap_1d(ga, gb, l, m - 2, A, B);
}

inline
double primitive_overlap(const primitive_gaussian& g1,
                         const primitive_gaussian& g2, 
                         const basis_function& b1,
                         const basis_function& b2){

    vector3 C1, C2;
    C1 = b1.center;
    C2 = b2.center;

    double normalization = g1.norm * g2.norm;

    double Sx = overlap_1d(g1, g2, b1.lx, b2.lx, C1.x, C2.x);
    double Sy = overlap_1d(g1, g2, b1.ly, b2.ly, C1.y, C2.y);
    double Sz = overlap_1d(g1, g2, b1.lz, b2.lz, C1.z, C2.z);

    return  normalization * Sx * Sy * Sz;
}


// computes the overlap between two basis functions
// by iterating over every pair of two primitive gaussians
inline
double basis_overlap(const basis_function& f1, const basis_function& f2){
    double s = 0.0;

    for (int i = 0; i < f1.prims.size(); ++ i){
        for (int j = 0; j < f2.prims.size(); ++ j){

            primitive_gaussian prim1 = f1.prims[i];
            primitive_gaussian prim2 = f2.prims[j];

            s += prim1.coeff * prim2.coeff *
            primitive_overlap(prim1, prim2, f1, f2);

        }
    }

    return s;
}


//computes the overlap matrix by iterating over every pair of two basis functions
inline
std::vector<double> overlap_matrix(const std::vector<basis_function>& basis){

    int K = basis.size();
    std::vector<double> S_matrix(K * K);

    for (int i = 0; i < (K + 1) * K / 2; ++ i){

        std::pair<int, int> idx = pair_to_indices(i);

        S_matrix[index2d(idx.first, idx.second, K)] =
        basis_overlap(basis[idx.first], basis[idx.second]);

        S_matrix[index2d(idx.second, idx.first, K)] =
        S_matrix[index2d(idx.first, idx.second, K)];
    
    }

    return S_matrix;
}


//------------------------KINETIC--------------------------------
inline
double kinetic_1d(const primitive_gaussian& ga,
                  const primitive_gaussian& gb,
                  int l,
                  int m,
                  double A,
                  double B){

    double beta = gb.alpha;

    double term1 = beta * (2 * m + 1) * 
    overlap_1d(ga, gb, l, m, A, B);

    double term2 = -2 * beta * beta * 
    overlap_1d(ga, gb, l, m + 2, A, B);

    double term3 = -0.5 * m * (m - 1) *
    overlap_1d(ga, gb, l, m - 2, A, B);

    return term1 + term2 + term3;
}

inline
double primitive_kinetic(const primitive_gaussian& g1,
                         const primitive_gaussian& g2, 
                         const basis_function& b1,
                         const basis_function& b2){

    vector3 C1, C2;
    C1 = b1.center;
    C2 = b2.center;

    int l_x = b1.lx;
    int l_y = b1.ly;
    int l_z = b1.lz;

    int m_x = b2.lx;
    int m_y = b2.ly;
    int m_z = b2.lz;

    double normalization = g1.norm * g2.norm;

    double term1 = kinetic_1d(g1, g2, l_x, m_x, C1.x, C2.x) *
                   overlap_1d(g1, g2, l_y, m_y, C1.y, C2.y) *
                   overlap_1d(g1, g2, l_z, m_z, C1.z, C2.z);
    
    double term2 = kinetic_1d(g1, g2, l_y, m_y, C1.y, C2.y) *
                   overlap_1d(g1, g2, l_x, m_x, C1.x, C2.x) *
                   overlap_1d(g1, g2, l_z, m_z, C1.z, C2.z);
    
    double term3 = kinetic_1d(g1, g2, l_z, m_z, C1.z, C2.z) *
                   overlap_1d(g1, g2, l_y, m_y, C1.y, C2.y) *
                   overlap_1d(g1, g2, l_x, m_x, C1.x, C2.x);
    
    return normalization * (term1 + term2 + term3);
}

inline
double basis_kinetic(const basis_function& f1, const basis_function& f2){
    double t = 0.0;

    for (int i = 0; i < f1.prims.size(); ++ i){
        for (int j = 0; j < f2.prims.size(); ++ j){

            primitive_gaussian prim1 = f1.prims[i];
            primitive_gaussian prim2 = f2.prims[j];

            t += prim1.coeff * prim2.coeff *
            primitive_kinetic(prim1, prim2, f1, f2);

        }
    }

    return t;
}


inline
std::vector<double> kinetic_matrix(const std::vector<basis_function>& basis){

    int K = basis.size();
    std::vector<double> T_matrix(K * K);

    for (int i = 0; i < K * (K + 1) / 2; ++ i){

        std::pair<int, int> idx = pair_to_indices(i);

        T_matrix[index2d(idx.first, idx.second, K)] =
        basis_kinetic(basis[idx.first], basis[idx.second]);

        T_matrix[index2d(idx.second, idx.first, K)] =
        T_matrix[index2d(idx.first, idx.second, K)];

    }

    return T_matrix;
}


//----------------NUCLEAR ATTRACTION------------------------------

inline
double boys(int m, double T)
{
    if (m < 0) {
        throw std::runtime_error("boys: negative m");
    }

    if (T < 0.0 || !std::isfinite(T)) {
        throw std::runtime_error("boys: invalid T");
    }

    // Use series for small/moderate-small T.
    // This is much safer for high m.
    if (T < 1e-4) {
        double term = 1.0;
        double sum = 0.0;

        for (int k = 0; k < 50; ++k) {
            double add = term / (2.0 * m + 2.0 * k + 1.0);
            sum += add;

            if (std::abs(add) < 1e-16 * std::max(1.0, std::abs(sum))) {
                break;
            }

            term *= -T / static_cast<double>(k + 1);
        }

        return sum;
    }

    constexpr double PI = 3.141592653589793238462643383279502884;

    double sqrtT = std::sqrt(T);
    double F = 0.5 * std::sqrt(PI / T) * std::erf(sqrtT);

    double expT = std::exp(-T);

    for (int k = 0; k < m; ++k) {
        F = ((2.0 * k + 1.0) * F - expT) / (2.0 * T);
    }

    return F;
}


inline
std::vector<double> hermite_E_array(int i, int j,
                                    double alpha, double beta,
                                    double A, double B)
{
    constexpr int MAX_L_CPU = 4;
    constexpr int MAX_E_CPU = 2 * MAX_L_CPU + 1;

    if (i > MAX_L_CPU || j > MAX_L_CPU || i + j >= MAX_E_CPU) {
        throw std::runtime_error("Hermite E angular momentum too large");
    }

    double E[MAX_L_CPU + 1][MAX_L_CPU + 1][MAX_E_CPU];

    for (int a = 0; a <= MAX_L_CPU; ++a)
        for (int b = 0; b <= MAX_L_CPU; ++b)
            for (int t = 0; t < MAX_E_CPU; ++t)
                E[a][b][t] = 0.0;

    double p = alpha + beta;
    double q = alpha * beta / p;
    double AB = A - B;

    E[0][0][0] = std::exp(-q * AB * AB);

    for (int ia = 1; ia <= i; ++ia) {
        for (int tt = 0; tt <= ia; ++tt) {
            double term1 = (tt > 0) ? E[ia - 1][0][tt - 1] / (2.0 * p) : 0.0;
            double term2 = -(beta / p) * AB * E[ia - 1][0][tt];
            double term3 = (tt + 1 < MAX_E_CPU) ? (tt + 1) * E[ia - 1][0][tt + 1] : 0.0;

            E[ia][0][tt] = term1 + term2 + term3;
        }
    }

    for (int ia = 0; ia <= i; ++ia) {
        for (int jb = 1; jb <= j; ++jb) {
            for (int tt = 0; tt <= ia + jb; ++tt) {
                double term1 = (tt > 0) ? E[ia][jb - 1][tt - 1] / (2.0 * p) : 0.0;
                double term2 = +(alpha / p) * AB * E[ia][jb - 1][tt];
                double term3 = (tt + 1 < MAX_E_CPU) ? (tt + 1) * E[ia][jb - 1][tt + 1] : 0.0;

                E[ia][jb][tt] = term1 + term2 + term3;
            }
        }
    }

    std::vector<double> out(i + j + 1);
    for (int tt = 0; tt <= i + j; ++tt) {
        out[tt] = E[i][j][tt];
    }

    return out;
}


inline constexpr int max_l = 3;                  // up to f
inline constexpr int max_R_order = 4 * max_l;    // ffff max = 12

inline
constexpr int n_R =
    (max_R_order + 1) *
    (max_R_order + 2) *
    (max_R_order + 3) / 6;



inline
bool valid_r_index_cpu(int a, int b, int c)
{
    return a >= 0 && b >= 0 && c >= 0 &&
           a + b + c <= max_R_order;
}

inline
int packed_r_index_cpu(int a, int b, int c)
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



inline
double coulomb_R_attraction(int tx, int ty, int tz, int n,
                                double p,
                                vector3 P, vector3 C)
{
    int max_order = tx + ty + tz + n;

    if (max_order > max_R_order ||
        !valid_r_index_cpu(tx, ty, tz)) {
        return 0.0;
    }

    double rho = p;

    double X = P.x - C.x;
    double Y = P.y - C.y;
    double Z = P.z - C.z;

    double T = p * (X*X + Y*Y + Z*Z);

    std::vector<double> Rnext(n_R, 0.0);
    std::vector<double> Rcurr(n_R, 0.0);

    Rnext[packed_r_index_cpu(0, 0, 0)] =
        std::pow(-2.0 * rho, max_order) * boys(max_order, T);

    for (int m = max_order - 1; m >= n; --m) {

        std::fill(Rcurr.begin(), Rcurr.end(), 0.0);

        Rcurr[packed_r_index_cpu(0, 0, 0)] =
            std::pow(-2.0 * rho, m) * boys(m, T);

        for (int s = 1; s <= tx + ty + tz; ++s) {
            for (int a = 0; a <= s; ++a) {
                for (int b = 0; b <= s - a; ++b) {

                    int c = s - a - b;

                    if (a > tx || b > ty || c > tz) continue;

                    double val = 0.0;

                    if (a > 0) {
                        val = X * Rnext[packed_r_index_cpu(a - 1, b, c)];

                        if (a >= 2) {
                            val += (a - 1) *
                                   Rnext[packed_r_index_cpu(a - 2, b, c)];
                        }
                    }
                    else if (b > 0) {
                        val = Y * Rnext[packed_r_index_cpu(a, b - 1, c)];

                        if (b >= 2) {
                            val += (b - 1) *
                                   Rnext[packed_r_index_cpu(a, b - 2, c)];
                        }
                    }
                    else if (c > 0) {
                        val = Z * Rnext[packed_r_index_cpu(a, b, c - 1)];

                        if (c >= 2) {
                            val += (c - 1) *
                                   Rnext[packed_r_index_cpu(a, b, c - 2)];
                        }
                    }

                    Rcurr[packed_r_index_cpu(a, b, c)] = val;
                }
            }
        }

        std::swap(Rnext, Rcurr);
    }

    return Rnext[packed_r_index_cpu(tx, ty, tz)];
}



inline
double primitive_attraction(const primitive_gaussian& g1,
                            const primitive_gaussian& g2, 
                            const basis_function& b1, 
                            const basis_function& b2,
                            vector3 R, int Z){
    
    vector3 C1, C2;
    C1 = b1.center;
    C2 = b2.center;

    int l_x = b1.lx;
    int l_y = b1.ly;
    int l_z = b1.lz;

    int m_x = b2.lx;
    int m_y = b2.ly;
    int m_z = b2.lz;

    double alpha = g1.alpha;
    double beta = g2.alpha;
    double p = alpha + beta;

    vector3 P;
    P.x = (alpha * C1.x + beta * C2.x) / p;
    P.y = (alpha * C1.y + beta * C2.y) / p;
    P.z = (alpha * C1.z + beta * C2.z) / p;

    std::vector<double> E1 = hermite_E_array(l_x, m_x, alpha, beta, C1.x, C2.x);
    std::vector<double> E2 = hermite_E_array(l_y, m_y, alpha, beta, C1.y, C2.y);
    std::vector<double> E3 = hermite_E_array(l_z, m_z, alpha, beta, C1.z, C2.z);

    double val = 0.0;

    for (int t = 0; t <= l_x + m_x; ++ t){
        for (int u = 0; u <= l_y + m_y; ++ u){
            for (int v = 0; v <= l_z + m_z; ++ v){

                val += E1[t] * E2[u] * E3[v] * 
                coulomb_R_attraction(t, u, v, 0, p, P, R);

            }
        }
    }
    return -(2 * M_PI / p) * g1.norm * g2.norm * Z * val;
}


inline
double basis_attraction(const basis_function& f1, const basis_function& f2,
                        const std::vector<atom>& atom_list){

    double v = 0.0;

    for (int i=0; i<f1.prims.size(); ++i){
        for (int j=0; j<f2.prims.size(); ++j){
            for (int k=0; k<atom_list.size(); ++k){

                primitive_gaussian prim1 = f1.prims[i];
                primitive_gaussian prim2 = f2.prims[j];

                v += prim1.coeff * prim2.coeff *
                primitive_attraction(prim1, prim2, f1, f2,
                                     atom_list[k].R, atom_list[k].Z);

            }
        }
    }

    return v;
}


inline
std::vector<double> attraction_matrix(const std::vector<basis_function>& basis,
                                      const std::vector<atom>& atom_list){

    int K = basis.size();
    std::vector<double> V_matrix(K * K);

    for (int i = 0; i < K * (K + 1) / 2 ; ++ i){

        std::pair<int, int> idx = pair_to_indices(i);

        V_matrix[index2d(idx.first, idx.second, K)] =
        basis_attraction(basis[idx.first], basis[idx.second], atom_list);
        
        V_matrix[index2d(idx.second, idx.first, K)] =
        V_matrix[index2d(idx.first, idx.second, K)];

    }

    return V_matrix;
}


//----------------------ELECTRON REPULSION----------------------------

inline
double coulomb_R(int tx, int ty, int tz, int n,
                     double p, double q,
                     vector3 P, vector3 Q)
{
    int max_order = tx + ty + tz + n;

    if (max_order > max_R_order ||
        !valid_r_index_cpu(tx, ty, tz)) {
        return 0.0;
    }

    double rho = p * q / (p + q);

    double X = P.x - Q.x;
    double Y = P.y - Q.y;
    double Z = P.z - Q.z;

    double T = rho * (X*X + Y*Y + Z*Z);

    std::vector<double> Rnext(n_R, 0.0);
    std::vector<double> Rcurr(n_R, 0.0);

    Rnext[packed_r_index_cpu(0, 0, 0)] =
        std::pow(-2.0 * rho, max_order) * boys(max_order, T);

    for (int m = max_order - 1; m >= n; --m) {

        std::fill(Rcurr.begin(), Rcurr.end(), 0.0);

        Rcurr[packed_r_index_cpu(0, 0, 0)] =
            std::pow(-2.0 * rho, m) * boys(m, T);

        for (int s = 1; s <= tx + ty + tz; ++s) {
            for (int a = 0; a <= s; ++a) {
                for (int b = 0; b <= s - a; ++b) {

                    int c = s - a - b;

                    if (a > tx || b > ty || c > tz) continue;

                    double val = 0.0;

                    if (a > 0) {
                        val = X * Rnext[packed_r_index_cpu(a - 1, b, c)];

                        if (a >= 2) {
                            val += (a - 1) *
                                   Rnext[packed_r_index_cpu(a - 2, b, c)];
                        }
                    }
                    else if (b > 0) {
                        val = Y * Rnext[packed_r_index_cpu(a, b - 1, c)];

                        if (b >= 2) {
                            val += (b - 1) *
                                   Rnext[packed_r_index_cpu(a, b - 2, c)];
                        }
                    }
                    else if (c > 0) {
                        val = Z * Rnext[packed_r_index_cpu(a, b, c - 1)];

                        if (c >= 2) {
                            val += (c - 1) *
                                   Rnext[packed_r_index_cpu(a, b, c - 2)];
                        }
                    }

                    Rcurr[packed_r_index_cpu(a, b, c)] = val;
                }
            }
        }

        std::swap(Rnext, Rcurr);
    }

    return Rnext[packed_r_index_cpu(tx, ty, tz)];
}





inline
double primitive_repulsion(const primitive_gaussian& a,
                     const primitive_gaussian& b,
                     const primitive_gaussian& c,
                     const primitive_gaussian& d,
                     const basis_function& f1,
                     const basis_function& f2,
                     const basis_function& f3,
                     const basis_function& f4)
{
    vector3 A = f1.center;
    vector3 B = f2.center;
    vector3 C = f3.center;
    vector3 D = f4.center;

    double alpha = a.alpha;
    double beta  = b.alpha;
    double gamma = c.alpha;
    double delta = d.alpha;

    // Gaussian pair quantities
    double p = alpha + beta;
    double q = gamma + delta;


    vector3 P;
    P.x = (alpha * A.x + beta * B.x) / p;
    P.y = (alpha * A.y + beta * B.y) / p;
    P.z = (alpha * A.z + beta * B.z) / p;

    vector3 Q;
    Q.x = (gamma * C.x + delta * D.x) / q;
    Q.y = (gamma * C.y + delta * D.y) / q;
    Q.z = (gamma * C.z + delta * D.z) / q;


    int t = f1.lx + f2.lx;
    int u = f1.ly + f2.ly;
    int v = f1.lz + f2.lz;
    int tau = f3.lx + f4.lx;
    int nu = f3.ly + f4.ly;
    int phi = f3.lz + f4.lz;

    std::vector<double> E1 = hermite_E_array(f1.lx, f2.lx, alpha, beta, A.x, B.x);
    std::vector<double> E2 = hermite_E_array(f1.ly, f2.ly, alpha, beta, A.y, B.y);
    std::vector<double> E3 = hermite_E_array(f1.lz, f2.lz, alpha, beta, A.z, B.z);
    std::vector<double> E4 = hermite_E_array(f3.lx, f4.lx, gamma, delta, C.x, D.x);
    std::vector<double> E5 = hermite_E_array(f3.ly, f4.ly, gamma, delta, C.y, D.y);
    std::vector<double> E6 = hermite_E_array(f3.lz, f4.lz, gamma, delta, C.z, D.z);


    double eri = 0.0;
    for (int t_i = 0; t_i<=t; ++t_i){
        for (int u_i = 0; u_i<=u; ++u_i){
            for (int v_i = 0; v_i<=v; ++v_i){
                for (int tau_i = 0; tau_i<=tau; ++tau_i){
                    for (int nu_i = 0; nu_i<=nu; ++nu_i){
                        for (int phi_i = 0; phi_i<=phi; ++phi_i){

                            double sign = ((tau_i + nu_i + phi_i) % 2 == 0) ? 1.0 : -1.0;

                            eri += sign *
                                   E1[t_i] * E2[u_i] * E3[v_i] *
                                   E4[tau_i] * E5[nu_i] * E6[phi_i] *
                                   coulomb_R(t_i + tau_i,
                                             u_i + nu_i,
                                             v_i + phi_i,
                                             0, p, q, P, Q);
                        }
                    }
                }
            }
        }
    }
    return 2 * pow(M_PI, 2.5) * pow(p + q, -0.5) * eri / (p * q);
}

inline
double basis_repulsion(const basis_function& f1,
                       const basis_function& f2,
                       const basis_function& f3,
                       const basis_function& f4){
                        
    double eri = 0.0;

    for (int i=0; i<f1.prims.size(); ++i){
        for (int j=0; j<f2.prims.size(); ++j){
            for (int k=0; k<f3.prims.size(); ++k){
                for (int l=0; l<f4.prims.size(); ++l){

                    primitive_gaussian prim1 = f1.prims[i];
                    primitive_gaussian prim2 = f2.prims[j];
                    primitive_gaussian prim3 = f3.prims[k];
                    primitive_gaussian prim4 = f4.prims[l];

                    eri += prim1.coeff*prim2.coeff*prim3.coeff*prim4.coeff * 
                    prim1.norm*prim2.norm*prim3.norm*prim4.norm *
                    primitive_repulsion(prim1, prim2, prim3, prim4, f1, f2, f3, f4);

                }
            }
        }
    }

    return eri;
}


inline
std::vector<double> schwarz_bounds(const std::vector<basis_function>& basis)
{
    int K = basis.size();
    int n_pairs = K * (K + 1) / 2;
    std::vector<double> bounds(n_pairs);

    for (int i = 0; i < n_pairs; ++ i){
        
        std::pair<int, int> idx = pair_to_indices(i);

        double bound = basis_repulsion(basis[idx.first],
                                    basis[idx.second],
                                    basis[idx.first],
                                    basis[idx.second]);

        bounds[i] = std::sqrt(std::max(bound, 0.0));
    }

    return bounds;
}

inline
std::vector<double> repulsion_tensor(const std::vector<basis_function>& basis, 
                                     double threshold)
{                               
    int K = basis.size();
    size_t n_pairs = K * (K + 1) / 2;
    size_t n_quartets = n_pairs * (n_pairs + 1) / 2;

    std::vector<double> bounds = schwarz_bounds(basis);
    std::vector<double> unique_eri(n_quartets, 0.0);

    for (size_t i = 0; i < n_quartets; ++ i){

        std::pair<int, int> q_idx = quartet_to_pairs(i);
        std::pair<int, int> p1_idx = pair_to_indices(q_idx.first);
        std::pair<int, int> p2_idx = pair_to_indices(q_idx.second);

        if (bounds[q_idx.first] * bounds[q_idx.second] < threshold){
            continue;
        } else{
        
            double val = basis_repulsion(basis[p1_idx.first], basis[p1_idx.second],
                                         basis[p2_idx.first], basis[p2_idx.second]);

            unique_eri[i] = val;
        }

        if (i % 5000 == 0 || i == n_quartets - 1) {
            double progress = 100.0 * double(i + 1) / double(n_quartets);
            std::cout << "\rERI progress: " << progress << "%      " << std::flush;
        }
    }
    return unique_eri;
}
