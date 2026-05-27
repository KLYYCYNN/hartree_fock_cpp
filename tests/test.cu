// Comparing primitive level ERI calculation between CPU and GPU

#include "integrals.hpp"
#include "kernels.cuh"
#include "gpu_types.hpp"

int main(){

    std::vector<std::pair<std::string, vector3>>
    config = {{"C", position_vector(0.0, 0.0, 0.0)}};

    std::string param_path = "/home/jc6224/hf_cpp/basis";

    std::vector<basis_function> basis_cpu = basis_construction(config, "cc-pVTZ", param_path).first;
    std::vector<BasisFunctionGPU> basis_gpu = convert_basis_to_gpu(basis_cpu);

    std::vector<int> function_indices = {18, 18, 18, 18};
    std::vector<int> prim_indices = {0, 0, 0, 0};

    basis_function b1, b2, b3, b4;
    BasisFunctionGPU b1g, b2g, b3g, b4g;
    primitive_gaussian p1, p2, p3, p4;
    PrimitiveGPU p1g, p2g, p3g, p4g;

    b1 = basis_cpu[function_indices[0]];
    b2 = basis_cpu[function_indices[1]];
    b3 = basis_cpu[function_indices[2]];
    b4 = basis_cpu[function_indices[3]];

    p1 = b1.prims[prim_indices[0]];
    p2 = b2.prims[prim_indices[1]];
    p3 = b3.prims[prim_indices[2]];
    p4 = b4.prims[prim_indices[3]];

    b1g = basis_gpu[function_indices[0]];
    b2g = basis_gpu[function_indices[1]];
    b3g = basis_gpu[function_indices[2]];
    b4g = basis_gpu[function_indices[3]];

    p1g = b1g.prims[prim_indices[0]];
    p2g = b2g.prims[prim_indices[1]];
    p3g = b3g.prims[prim_indices[2]];
    p4g = b4g.prims[prim_indices[3]];

    double eri_cpu = primitive_repulsion(p1, p2, p3, p4, b1, b2, b3, b4);
    double eri_gpu;
    primitive_eri_gpu(p1g, p2g, p3g, p4g, b1g, b2g, b3g, b4g, eri_gpu);

    std::cout << "CPU: " << eri_cpu << std::endl;
    std::cout << "GPU: " << eri_gpu << std::endl;

    return 0;
}