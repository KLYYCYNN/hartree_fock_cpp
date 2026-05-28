#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>
#include "scf.hpp"

inline void index4d_inv(int idx, int K,
                        int& i, int& j, int& k, int& l)
{
    l = idx % K;
    idx /= K;

    k = idx % K;
    idx /= K;

    j = idx % K;
    idx /= K;

    i = idx;
}

inline void print_basis_func_info(const basis_info& info, int idx)
{
    std::cout << "basis[" << idx << "] "
              << "atom=" << info.atoms[idx]
              << ", type=" << info.types[idx]
              << ", shell=" << info.shells[idx]
              << ", lmn=("
              << info.lx[idx] << ", "
              << info.ly[idx] << ", "
              << info.lz[idx] << ")"
              << ", center=("
              << info.cx[idx] << ", "
              << info.cy[idx] << ", "
              << info.cz[idx] << ")";
}

void print_top_eri_differences(const std::vector<double>& eri_cpu,
                               const std::vector<double>& eri_gpu,
                               const basis_info& info,
                               int top_n = 10)
{
    if (eri_cpu.size() != eri_gpu.size()) {
        throw std::runtime_error("CPU/GPU ERI tensors have different sizes.");
    }

    int K = info.index.size();

    if (eri_cpu.size() != static_cast<size_t>(K * K * K * K)) {
        throw std::runtime_error("ERI tensor size does not match basis size.");
    }

    struct DiffEntry {
        int idx;
        double cpu;
        double gpu;
        double abs_diff;
        double rel_diff;
    };

    std::vector<DiffEntry> diffs;
    diffs.reserve(eri_cpu.size());

    for (int idx = 0; idx < static_cast<int>(eri_cpu.size()); ++idx) {
        double cpu = eri_cpu[idx];
        double gpu = eri_gpu[idx];
        double abs_diff = std::abs(cpu - gpu);

        double denom = std::max(std::abs(cpu), 1e-14);
        double rel_diff = abs_diff / denom;

        diffs.push_back({idx, cpu, gpu, abs_diff, rel_diff});
    }

    std::partial_sort(
        diffs.begin(),
        diffs.begin() + std::min(top_n, static_cast<int>(diffs.size())),
        diffs.end(),
        [](const DiffEntry& a, const DiffEntry& b) {
            return a.abs_diff > b.abs_diff;
        }
    );

    std::cout << std::setprecision(16);

    int nprint = std::min(top_n, static_cast<int>(diffs.size()));

    for (int n = 0; n < nprint; ++n) {
        const auto& d = diffs[n];

        int i, j, k, l;
        index4d_inv(d.idx, K, i, j, k, l);

        std::cout << "\n==============================\n";
        std::cout << "Rank " << n + 1 << "\n";
        std::cout << "flat index = " << d.idx << "\n";
        std::cout << "quartet = ("
                  << i << ", " << j << ", "
                  << k << ", " << l << ")\n";

        std::cout << "CPU = " << d.cpu << "\n";
        std::cout << "GPU = " << d.gpu << "\n";
        std::cout << "abs diff = " << d.abs_diff << "\n";
        std::cout << "rel diff = " << d.rel_diff << "\n\n";

        std::cout << "mu:     ";
        print_basis_func_info(info, i);
        std::cout << "\n";

        std::cout << "nu:     ";
        print_basis_func_info(info, j);
        std::cout << "\n";

        std::cout << "lambda: ";
        print_basis_func_info(info, k);
        std::cout << "\n";

        std::cout << "sigma:  ";
        print_basis_func_info(info, l);
        std::cout << "\n";
    }
}

int main(){

    std::vector<std::pair<std::string, vector3>> 
    config = {{"Be", position_vector(0.0, 0.0, 0.0)},
              {"H", position_vector(2.52, 0.0, 0.0)},
              {"H", position_vector(-2.52, 0.0, 0.0)}};

    std::pair<std::vector<basis_function>, basis_info> 
    basis = basis_construction(config, "cc-pVTZ", "/home/jc6224/hf_cpp/basis");

    std::vector<double> eri_cpu = repulsion_tensor(basis.first, 1e-8);
    std::vector<double> eri_gpu = ERI_GPU(basis.first, 1e-8);


    print_top_eri_differences(eri_cpu, eri_gpu, basis.second, 50);

    return 0;
}