#pragma once
#include "types.hpp"
#include "basis.hpp"

constexpr int MAX_PRIMS = 10;

struct PrimitiveGPU {
    double alpha;
    double coeff;
    double norm;
};

struct BasisFunctionGPU {
    int lx, ly, lz;
    double x, y, z;
    int nprims;
    PrimitiveGPU prims[MAX_PRIMS];
};


inline
std::vector<BasisFunctionGPU>
convert_basis_to_gpu(const std::vector<basis_function>& basis){
    std::vector<BasisFunctionGPU> gpu_basis;
    gpu_basis.reserve(basis.size());

    for (const auto& bf : basis){
        assert(bf.prims.size() <= MAX_PRIMS);

        BasisFunctionGPU gbf;

        gbf.lx = bf.lx;
        gbf.ly = bf.ly;
        gbf.lz = bf.lz;

        gbf.x = bf.center.x;
        gbf.y = bf.center.y;
        gbf.z = bf.center.z;

        gbf.nprims = static_cast<int>(bf.prims.size());

        for (int i = 0; i < gbf.nprims; ++i){
            gbf.prims[i].alpha = bf.prims[i].alpha;
            gbf.prims[i].coeff = bf.prims[i].coeff;
            gbf.prims[i].norm  = bf.prims[i].norm;
        }

        gpu_basis.push_back(gbf);
    }

    return gpu_basis;
}