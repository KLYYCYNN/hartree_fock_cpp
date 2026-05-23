#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <filesystem>
#include <nlohmann/json.hpp>
#include "basis.hpp"
#include "rapidcsv.h"


//save matrices to csv, each nested subvector is a column.
void save_csv(const std::vector<std::vector<double>>& data,
              std::vector<std::string> column_headers,
              std::string fileName) {

    assert (column_headers.size() == data.size() && 
            "Headers don't match the columns.");

    rapidcsv::Document doc;

    for (int i=0; i<data.size(); ++i){
        doc.SetColumn<double>(i, data[i]);
        doc.SetColumnName(i, column_headers[i]);
    }

    doc.Save(fileName);
}

std::string formatElementCounts(const std::vector<std::string>& elements){
    if (elements.empty()) return "";

    // Vector of pairs to track {Element, Count} in order of appearance
    std::vector<std::pair<std::string_view, int>> counts;

    for (std::string_view el : elements) {
        if (!counts.empty() && counts.back().first == el) {
            // Increment count if it matches the current running element
            counts.back().second++;
        } else {
            // Start a new element group
            counts.push_back({el, 1});
        }
    }

    // Construct the final string
    std::string result;
    for (size_t i = 0; i < counts.size(); ++i) {
        if (counts[i].second > 1) {
            result += std::to_string(counts[i].second);
        }
        result += counts[i].first;

        // Add a comma and space separator except for the last item
        if (i < counts.size() - 1) {
            result += ", ";
        }
    }

    return result;
}

void create_folder(const std::string& folder_path) {
    try {
        // create_directories creates parent folders automatically if missing
        if (std::filesystem::create_directories(folder_path)) {

        } else {
            // Returns false if the path already exists
            std::cout << "Folder already exists: " << folder_path << std::endl;
        }
    } 
    catch (const std::filesystem::filesystem_error& e) {
        // Catches permission errors or invalid character issues
        std::cerr << "Error creating directory: " << e.what() << std::endl;
    }
}


void save_basis(const basis_info& info, std::string folder)
{

    std::string path = folder + "/basis.csv";
    rapidcsv::Document file;

    file.SetColumn<int>(0, info.index);
    file.SetColumnName(0, "index");

    file.SetColumn<std::string>(1, info.atoms);
    file.SetColumnName(1, "atom");

    file.SetColumn<int>(2, info.shells);
    file.SetColumnName(2, "shell_id");

    file.SetColumn<std::string>(3, info.types);
    file.SetColumnName(3, "angmom");

    file.SetColumn<int>(4, info.lx);
    file.SetColumnName(4, "lx");

    file.SetColumn<int>(5, info.ly);
    file.SetColumnName(5, "ly");

    file.SetColumn<int>(6, info.lz);
    file.SetColumnName(6, "lz");

    file.SetColumn<double>(7, info.cx);
    file.SetColumnName(7, "x");

    file.SetColumn<double>(8, info.cy);
    file.SetColumnName(8, "y");

    file.SetColumn<double>(9, info.cz);
    file.SetColumnName(9, "z");

    file.Save(path);
}


void save_metadata(std::string folder,
                   scf_data data, 
                   std::string method, 
                   std::string unit)
{

    std::string path = folder + "/metadata.json";

    std::vector<std::pair<std::string, vector3>>
    config = data.system;
    std::vector<std::string> atom_list;

    for (int i = 0; i < config.size(); ++ i){
        atom_list.emplace_back(config[i].first);
    }

    nlohmann::ordered_json metadata;
    metadata["method"] = method;
    metadata["basis"] = data.BasisSet;
    metadata["n_basis"] = data.K;
    metadata["unit"] = unit;
    metadata["n_iterations"] = data.iterations;
    metadata["converged"] = data.converged;
    metadata["ERI_hardware"] = data.hardware;

    metadata["system"] = formatElementCounts(atom_list);
    metadata["charge_e"] = data.charge;
    metadata["energy_hartree"] = data.E;

    metadata["start_time"] = data.start;
    metadata["finish_time"] = data.end;
    metadata["duration_s"] = data.duration;

    std::ofstream file(path);
    if (file.is_open()){
        file << metadata.dump(4);
        file.close();
    } else{
        std::cout << "metadata writing failed." << std::endl;
    }

}


void save_geometry(std::vector<std::pair<std::string, vector3>> config,
                   std::string folder)
{
    std::vector<std::string> atom_list;
    std::vector<double> x1;
    std::vector<double> x2;
    std::vector<double> x3;

    for (int i = 0; i < config.size(); ++ i){

        atom_list.emplace_back(config[i].first);
        x1.emplace_back(config[i].second.x);
        x2.emplace_back(config[i].second.y);
        x3.emplace_back(config[i].second.z);

    }

    std::string path = folder + "/geometry.csv";

    rapidcsv::Document file;
    file.SetColumn<std::string>(0, atom_list);
    file.SetColumnName(0, "atom");

    file.SetColumn<double>(1, x1);
    file.SetColumnName(1, "x");
    file.SetColumn<double>(2, x2);
    file.SetColumnName(2, "y");
    file.SetColumn<double>(3, x3);
    file.SetColumnName(3, "z");

    file.Save(path);
}

void save_scf_data(std::string path,
                   std::string folder_name,
                   scf_data data,
                   std::string method = "RHF",
                   std::string unit = "Bohr")
{

    std::string folder = path + "/" + folder_name;
    create_folder(folder);

    save_metadata(folder, data, method, unit);

    save_csv({data.density}, {"density_matrix"},
              folder + "/" + "density.csv");

    save_csv({data.coefficient}, {"coefficient_matrix"},
              folder + "/" + "coefficient.csv");

    save_csv({data.mo_energy}, {"mo_energies"},
              folder + "/" + "orbital_energies.csv");

    save_basis(data.basis, folder);
    save_geometry(data.system, folder);

    std::cout << "Data successfully saved." << std::endl;
}