#pragma once
#include <vector>
#include "gpu_types.hpp"

std::vector<double> ERI_GPU(
    const std::vector<basis_function>& basis,
    double threshold
);