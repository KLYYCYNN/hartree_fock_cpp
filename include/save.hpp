#pragma once
#include <map>
#include <fstream>
#include <utility>
#include <filesystem>
#include <nlohmann/json.hpp>
#include "basis.hpp"
#include "rapidcsv.h"


//save matrices to csv, each nested subvector is a column.
inline
void save_csv(const std::vector<std::vector<double>>& data,
              const std::vector<std::string>& column_headers,
              std::string fileName) {

    if (column_headers.size() != data.size()){
        throw std::runtime_error(R"(number of columns is different
                                from number of headers)");
    }

    rapidcsv::Document doc;

    for (int i=0; i<data.size(); ++i){
        doc.SetColumn<double>(i, data[i]);
        doc.SetColumnName(i, column_headers[i]);
    }

    doc.Save(fileName);
}

inline
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

inline
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


inline
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


inline
void rhf_metadata(std::string folder,
                   rhf_data data, 
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
    metadata["electron_number"] = data.Ne;
    metadata["total_energy"] = data.E_tot;
    metadata["electronic_energy"] = data.E_ele;
    metadata["nuclear_energy"] = data.E_nuc;

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


inline
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

inline
void save_rhf_data(std::string path,
                   std::string folder_name,
                   rhf_data data,
                   std::string method = "RHF",
                   std::string unit = "Bohr")
{

    std::string folder = path + "/" + folder_name;
    create_folder(folder);

    rhf_metadata(folder, data, method, unit);

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


inline
void uhf_metadata(std::string folder,
                  uhf_data data, 
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
    metadata["alpha_occupancy"] = data.Na;
    metadata["beta_occupancy"] = data.Nb;
    metadata["electronic_energy"] = data.E_ele;
    metadata["nuclear_energy"] = data.E_nuc;
    metadata["total_energy"] = data.E_tot;

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



inline
void save_uhf_data(std::string path,
                   std::string folder_name,
                   uhf_data data,
                   std::string method = "UHF",
                   std::string unit = "Bohr")
{

    std::string folder = path + "/" + folder_name;
    create_folder(folder);

    uhf_metadata(folder, data, method, unit);

    save_csv({data.Pa, data.Pb},
             {"density_alpha", "density_beta"},
             folder + "/" + "density.csv");

    save_csv({data.Ca, data.Cb},
             {"coefficient_alpha", "coefficient_beta"},
             folder + "/" + "coefficient.csv");
             
    save_basis(data.basis, folder);
    save_geometry(data.system, folder);
    save_csv({data.Ea, data.Eb},
             {"e_a", "e_b"},
             folder + "/" + "orbital_energies.csv");

    std::cout << "Data successfully saved." << std::endl;
}



inline
void save_blscan(const blscan_data& data,
                 const std::string& save_path,
                 const std::string& folder_name)
{
    namespace fs = std::filesystem;

    if (data.distances.size() != data.energies.size() ||
        data.distances.size() != data.homo_energies.size() ||
        data.distances.size() != data.lumo_energies.size() ||
        data.distances.size() != data.iterations.size()) {
        throw std::runtime_error("blscan_data vector size mismatch");
    }

    fs::path out_dir = fs::path(save_path) / folder_name;
    fs::create_directories(out_dir);

    fs::path csv_path = out_dir / "scan_data.csv";
    fs::path json_path = out_dir / "scan_info.json";

    std::vector<std::string> headers = {"distance", "total_energy",
                                        "homo_energy", "lumo_energy",
                                        "iterations"};
    
    save_csv({data.distances, data.energies, data.homo_energies, 
              data.lumo_energies, data.iterations}, 
              headers, csv_path.string());

    nlohmann::ordered_json info;

    info["system"] = data.system;
    info["method"] = "RHF";
    info["basis"] = data.basis;
    info["basis_size"] = data.K;

    info["n_points"] = data.n_points;
    info["min_distance"] = data.min_d;
    info["max_distance"] = data.max_d;

    info["start"] = data.start;
    info["end"] = data.end;
    info["duration_seconds"] = data.duration;

    info["files"] = {
        {"scan_data", "scan_data.csv"}
    };

    std::ofstream js(json_path);

    if (!js.is_open()) {
        throw std::runtime_error("Could not open JSON file: " +
                                 json_path.string());
    }

    js << std::setw(4) << info << "\n";
    js.close();

    std::cout << "Bond-length scan saved to: "
              << out_dir.string() << std::endl;
}


std::pair<std::vector<std::string>, std::vector<int>> 
countElements(const std::vector<std::string>& input) {
    // 1. Count frequencies using a map
    std::map<std::string, int> frequencyMap;
    for (const auto& element : input) {
        frequencyMap[element]++;
    }

    // 2. Prepare output structures
    std::vector<std::string> uniqueElements;
    std::vector<int> counts;
    
    // Reserve memory to avoid reallocations
    uniqueElements.reserve(frequencyMap.size());
    counts.reserve(frequencyMap.size());

    // 3. Populate output vectors
    for (const auto& pair : frequencyMap) {
        uniqueElements.push_back(pair.first);
        counts.push_back(pair.second);
    }

    return {uniqueElements, counts};
}


inline
std::pair<int, int> atom_spin_occupancy(const std::string& atom)
{
    static const std::unordered_map<std::string,
                                    std::pair<int, int>> occ = {

        {"H",  {1, 0}},
        {"He", {1, 1}},

        {"Li", {2, 1}},
        {"Be", {2, 2}},
        {"B",  {3, 2}},
        {"C",  {4, 2}},
        {"N",  {4, 3}},
        {"O",  {5, 3}},
        {"F",  {5, 4}},
        {"Ne", {5, 5}},

        {"Na", {6, 5}},
        {"Mg", {6, 6}},
        {"Al", {7, 6}},
        {"Si", {8, 6}},
        {"P",  {8, 7}},
        {"S",  {9, 7}},
        {"Cl", {9, 8}},
        {"Ar", {9, 9}}
    };

    auto it = occ.find(atom);

    if (it == occ.end()) {
        throw std::runtime_error(
            "Spin occupancy not implemented for atom: " + atom
        );
    }

    return it->second;
}


inline
void save_disstn_data(const disstn_data& data,
                      const std::string& save_path,
                      const std::string& folder_name)
{
    namespace fs = std::filesystem;

    fs::path out_dir = fs::path(save_path) / folder_name;
    fs::create_directories(out_dir);

    std::string folder = out_dir.string();

    save_geometry(data.system, folder);
    save_basis(data.basis, folder);

    save_csv(
        {data.density},
        {"density"},
        (out_dir / "density.csv").string()
    );

    save_csv(
        {data.coefficient},
        {"coefficient"},
        (out_dir / "coefficients.csv").string()
    );

    save_csv(
        {data.mo_energy},
        {"orbital_energy"},
        (out_dir / "orbital_energies.csv").string()
    );

    nlohmann::ordered_json info;

    info["method"] = "RHF molecule + UHF atoms";
    info["basis"] = data.BasisSet;
    info["basis_size"] = data.K;
    info["charge"] = data.charge;
    info["n_electrons"] = data.Ne;
    info["hardware"] = data.hardware;

    info["molecule_energy"] = data.energy;
    info["total_atom_energy"] =
        data.energy + data.disstn_energy;

    info["dissociation_energy_hartree"] = data.disstn_energy;
    info["dissociation_energy_ev"] = data.disstn_energy * 27.211386245988;
    info["dissociation_energy_kj_mol"] = data.disstn_energy * 2625.499638;

    info["atoms"] = data.atoms;
    info["atom_numbers"] = data.atom_numbers;
    info["atom_energies"] = data.energies;

    info["start"] = data.start;
    info["end"] = data.end;
    info["duration_seconds"] = data.duration;

    info["files"] = {
        {"geometry", "geometry.csv"},
        {"basis", "basis.csv"},
        {"density", "density.csv"},
        {"coefficients", "coefficients.csv"},
        {"orbital_energies", "orbital_energies.csv"}
    };

    fs::path json_path = out_dir / "dissociation_info.json";

    std::ofstream js(json_path);

    if (!js.is_open()) {
        throw std::runtime_error(
            "Could not open JSON file: " + json_path.string()
        );
    }

    js << std::setw(4) << info << "\n";
    js.close();

    std::cout << "Dissociation data saved to: "
              << out_dir.string() << std::endl;
}