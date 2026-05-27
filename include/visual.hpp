#pragma once
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <stdexcept>
#include "rapidcsv.h"
#include "types.hpp"


void printMatrix(const std::vector<std::vector<double>>& mat) {
    for (const auto& row : mat) {
        for (const auto& val : row)
            std::cout << val << "\t";   // tab-separated columns
        std::cout << "\n";
    }
}


void printVector(const std::vector<double>& v) {
    for (const auto& i : v) {
        std::cout << i << " ";
    }
    std::cout << std::endl; // New line after the loop
}


void saveVector(const std::string& filename, const std::vector<double>& vec) {
    std::ofstream outFile(filename);

    if (outFile.is_open()) {
        for (const auto& val : vec) {
            outFile << val << "\n";
        }
        outFile.close();
    } else {
        std::cerr << "Error: Could not open file " << filename << std::endl;
    }
}


std::vector<double> linspace(double start, double stop, int npoints){
    double step = (stop - start) / (npoints - 1);
    std::vector<double> arrray;
    for (int i = 0; i < npoints; ++i)
    {
        arrray.emplace_back(start + i*step);
    }

    return arrray;
}


template <typename T>
void printVector(const std::vector<T>& vec, const std::string& sep = " ") {
    if (vec.empty()) {
        std::cout << "\n";
        return;
    }

    // Print all elements except the last one to avoid a trailing separator
    for (size_t i = 0; i < vec.size() - 1; ++i) {
        std::cout << vec[i] << sep;
    }
    
    // Print the final element followed by a newline
    std::cout << vec.back() << "\n";
}


std::vector<double> density_gpu(const std::vector<basis_function>& basis,
                                const std::vector<double>& density_matrix,
                                const std::vector<vector3>& r);



std::vector<double> slice_vector(const std::vector<double>& vec, int start, int end) {
    // Safety check: ensure indices are valid and in order
    if (start < 0 || end > vec.size() || start > end) {
        throw std::runtime_error(R"(Error in slice_vector:
  - Invalid slice indices provided.)");
    }

    // Return the sliced sub-vector using iterator range [start, end)
    return std::vector<double>(vec.begin() + start, vec.begin() + end);
}                       