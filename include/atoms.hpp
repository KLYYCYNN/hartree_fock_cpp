#pragma once
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <iterator>
#include <string_view>
#include "types.hpp"
#include "rapidcsv.h"


inline const std::vector<std::string> elements = 
{"H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na",
 "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca", "Sc", "Ti",
 "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge",
 "As", "Se", "Br", "Kr"};


inline
std::string Zname(int Z){
    return elements[Z-1];
}

inline
int getIndex(const std::vector<std::string>& vec, const std::string& element) {
    auto it = std::find(vec.begin(), vec.end(), element);

    if (it != vec.end()) {
        return std::distance(vec.begin(), it);
    }

    return -1; 
}

inline
int nameZ(std::string name){
    int Z = getIndex(elements, name) + 1;
    return Z;
}

inline
std::vector<atom> configuration(const std::vector<std::pair<std::string, vector3>>& config){
    std::vector<atom> atom_list;
    for (int i=0; i<config.size(); ++i){
        atom atom_i;
        std::string atom_type = config[i].first;
        int proton_number = nameZ(atom_type);

        atom_i.Z = proton_number;
        atom_i.R = config[i].second;
        atom_list.emplace_back(atom_i);
    }

    return atom_list;
}

//recreate system parameters using geometry.csv
inline
std::vector<atom> recreate_system(std::string file_path)
{

    rapidcsv::Document info(file_path);

    std::vector<std::string> atom_list 
    = info.GetColumn<std::string>("atom");

    std::vector<double> x
    = info.GetColumn<double>("x");
    std::vector<double> y
    = info.GetColumn<double>("y");
    std::vector<double> z
    = info.GetColumn<double>("z");

    std::vector<atom> system;

    for (int i = 0; i < atom_list.size();  ++ i){
        atom atom_i;
        atom_i.Z = nameZ(atom_list[i]);
        atom_i.R = position_vector(x[i], y[i], z[i]);

        system.emplace_back(atom_i);
    }

    return system;
}

