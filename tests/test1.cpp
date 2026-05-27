// testing ERI tensors computed on GPU against those computed on CPU

#include "integrals.hpp"
#include "eri_gpu.hpp"
#include "visual.hpp"

int main(){

    std::vector<std::pair<std::string, vector3>> 
    config = {{"Li", position_vector(0.0, 0.0, 0.0)},
              {"H", position_vector(3.014, 0.0, 0.0)}};

    std::vector<basis_function> basis = basis_construction(config, "cc-pVTZ", "/home/jc6224/hf_cpp/basis").first;

    std::vector<double> eri_cpu = repulsion_tensor(basis, 1e-8);
    std::vector<double> eri_gpu = ERI_GPU(basis, 1e-8);

    std::pair<int, int> idx = {23542, 23600};
    std::cout << "CPU:" << std::endl;
    printVector(slice_vector(eri_cpu, idx.first, idx.second));
    std::cout <<""<< std::endl;
    std::cout <<""<< std::endl;
    std::cout << "GPU:" << std::endl;
    printVector(slice_vector(eri_gpu, idx.first, idx.second));

    return 0;
}