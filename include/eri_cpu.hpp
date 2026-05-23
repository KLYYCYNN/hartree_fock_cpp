#pragma once
#include <functional>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include "types.hpp"
#include "indexing.hpp"
#include "integrals.hpp"

// CPU side of the hybrid approach

void ERI_CPU(const std::vector<basis_function>& basis,
             const std::vector<double>& bounds,
             std::vector<double>& unique_eri,
             int end_qid,
             double threshold)
{
    int K = basis.size();

    for (int qid = 0; qid < end_qid; ++qid) {

        auto q_idx = pair_to_indices(qid);
        int p = q_idx.first;
        int q = q_idx.second;

        if (bounds[p] * bounds[q] < threshold) {
            unique_eri[qid] = 0.0;
            continue;
        }

        auto p1_idx = pair_to_indices(p);
        auto p2_idx = pair_to_indices(q);

        unique_eri[qid] = basis_repulsion(
            basis[p1_idx.first], basis[p1_idx.second],
            basis[p2_idx.first], basis[p2_idx.second]
        );

        if (qid % 5000 == 0 || qid == end_qid - 1) {
            double progress = 100.0 * double(qid + 1) / double(end_qid);
            std::cout << "\rCPU ERI progress: "
                      << progress << "%      " << std::flush;
        }
    }
}



//This function is used to fill the ERI tensor after all unique quartets are integrated.
std::vector<double> fill_eri_symmetry(const std::vector<double>& unique_quartets, int K)
{                               
    std::vector<double> ERI(K * K * K * K);

    for (int i = 0; i < unique_quartets.size(); ++ i){

        double val = unique_quartets[i];

        std::pair<int, int> q_idx = pair_to_indices(i);
        std::pair<int, int> p1_idx = pair_to_indices(q_idx.first);
        std::pair<int, int> p2_idx = pair_to_indices(q_idx.second);


        ERI[index4d(p1_idx.first, p1_idx.second,
                    p2_idx.first, p2_idx.second, K)] = val;

        ERI[index4d(p1_idx.second, p1_idx.first,
                    p2_idx.first, p2_idx.second, K)] = val;

        ERI[index4d(p1_idx.first, p1_idx.second,
                    p2_idx.second, p2_idx.first, K)] = val;

        ERI[index4d(p1_idx.second, p1_idx.first,
                    p2_idx.second, p2_idx.first, K)] = val;


        ERI[index4d(p2_idx.first, p2_idx.second,
                    p1_idx.first, p1_idx.second, K)] = val;

        ERI[index4d(p2_idx.second, p2_idx.first,
                    p1_idx.first, p1_idx.second, K)] = val;

        ERI[index4d(p2_idx.first, p2_idx.second,
                    p1_idx.second, p1_idx.first, K)] = val;

        ERI[index4d(p2_idx.second, p2_idx.first,
                    p1_idx.second, p1_idx.first, K)] = val;

        }

    return ERI;
}
