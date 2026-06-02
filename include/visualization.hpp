#include "cube.hpp"
#include "visual.hpp"


//------------------------------2D visualization-----------------------------
//oooooOOOOO00000OOOOOooooooooooOOOOO00000OOOOOooooooooooOOOOO00000OOOOOooooo
//oooooOOOOO00000OOOOOooooooooooOOOOO00000OOOOOooooooooooOOOOO00000OOOOOooooo
//oooooOOOOO00000OOOOOooooooooooOOOOO00000OOOOOooooooooooOOOOO00000OOOOOooooo
//----------------------visualization options struct-------------------------

struct visual_options{

    int resolution = 512;
    vector3 center;
    double side_length;
    std::string plane = "none";

    std::string dimension = "2D";
    std::string render_hardware = "CPU";
    std::string data_path;
    std::string save_path;
    std::string folder_name;
    std::string basis_path;
    bool render_density = false;

    bool render_homo_minus_1 = false;
    bool render_homo = false;
    bool render_lumo = false;
    bool render_lumo_plus_1 = false;

    bool render_homo_minus_1_alpha = false;
    bool render_homo_alpha = false;
    bool render_lumo_alpha = false;
    bool render_lumo_plus_1_alpha = false;

    bool render_homo_minus_1_beta = false;
    bool render_homo_beta = false;
    bool render_lumo_beta = false;
    bool render_lumo_plus_1_beta = false;

};

//----------------------read visualization options-------------------------


inline
visual_options read_visual_options(const std::string& filename)
{
    std::ifstream file(filename);

    if (!file.is_open()) {
        throw std::runtime_error(
            "Could not open visualization options file: " + filename
        );
    }

    nlohmann::json j;
    file >> j;

    visual_options opt;

    opt.resolution = j.value("resolution", 512);

    auto center = j.at("center");
    opt.center = position_vector(
        center.at(0).get<double>(),
        center.at(1).get<double>(),
        center.at(2).get<double>()
    );

    opt.side_length = j.at("side_length").get<double>();

    opt.plane = j.value("plane", "none");

    opt.dimension = j.value("dimension", "2D");
    opt.render_hardware = j.value("render_hardware", "CPU");

    opt.data_path = j.at("data_path").get<std::string>();
    opt.save_path = j.at("save_path").get<std::string>();

    opt.render_density = j.value("render_density", false);

    opt.render_homo_minus_1 =
        j.value("render_homo_minus_1", j.value("render_homo-1", false));
    opt.render_homo = j.value("render_homo", false);
    opt.render_lumo = j.value("render_lumo", false);
    opt.render_lumo_plus_1 =
        j.value("render_lumo_plus_1", j.value("render_lumo+1", false));

    opt.render_homo_minus_1_alpha =
        j.value("render_homo_minus_1_alpha",
                j.value("render_homo-1_alpha", false));
    opt.render_homo_alpha = j.value("render_homo_alpha", false);
    opt.render_lumo_alpha = j.value("render_lumo_alpha", false);
    opt.render_lumo_plus_1_alpha =
        j.value("render_lumo_plus_1_alpha",
                j.value("render_lumo+1_alpha", false));

    opt.render_homo_minus_1_beta =
        j.value("render_homo_minus_1_beta",
                j.value("render_homo-1_beta", false));
    opt.render_homo_beta = j.value("render_homo_beta", false);
    opt.render_lumo_beta = j.value("render_lumo_beta", false);
    opt.render_lumo_plus_1_beta =
        j.value("render_lumo_plus_1_beta",
                j.value("render_lumo+1_beta", false));

    opt.basis_path = j.at("basis_path").get<std::string>();
    opt.folder_name = j.value("folder_name", "visualization");

    return opt;
}


inline
std::vector<atom> read_geometry_atoms(const std::string& file_name)
{
    rapidcsv::Document file(file_name);

    std::vector<std::string> symbols =
        file.GetColumn<std::string>("atom");

    std::vector<double> x = file.GetColumn<double>("x");
    std::vector<double> y = file.GetColumn<double>("y");
    std::vector<double> z = file.GetColumn<double>("z");

    if (symbols.size() != x.size() ||
        symbols.size() != y.size() ||
        symbols.size() != z.size()) {
        throw std::runtime_error("geometry.csv column size mismatch");
    }

    std::vector<atom> atoms(symbols.size());

    for (size_t i = 0; i < symbols.size(); ++i) {
        atoms[i].Z = nameZ(symbols[i]);
        atoms[i].R = position_vector(x[i], y[i], z[i]);
    }

    return atoms;
}


inline
bool json_contains(const nlohmann::json& j, const std::string& key)
{
    return j.find(key) != j.end();
}


inline
int orbital_index_from_homo_offset(const int n_occ,
                                   const int offset,
                                   const int K)
{
    int idx = n_occ - 1 + offset;

    if (idx < 0 || idx >= K) {
        throw std::runtime_error("orbital index out of range");
    }

    return idx;
}


//----------------------run visualization-------------------------

inline
void run_visualization(const visual_options& opt)
{
    namespace fs = std::filesystem;

    if (opt.dimension != "2D" && opt.dimension != "3D") {
        throw std::runtime_error("dimension must be 2D or 3D");
    }

    if (opt.render_hardware != "CPU" && opt.render_hardware != "GPU") {
        throw std::runtime_error("render_hardware must be CPU or GPU");
    }

    if (opt.dimension == "2D" &&
        opt.plane != "xy" &&
        opt.plane != "yz" &&
        opt.plane != "xz") {
        throw std::runtime_error("2D plane must be xy, yz, or xz");
    }

    fs::path data_dir = fs::path(opt.data_path);
    fs::path out_dir = fs::path(opt.save_path) / opt.folder_name;

    fs::create_directories(out_dir);

    fs::path basis_file = data_dir / "basis.csv";
    fs::path geometry_file = data_dir / "geometry.csv";
    fs::path density_file = data_dir / "density.csv";
    fs::path coeff_file = data_dir / "coefficient.csv";
    fs::path metadata_file = data_dir / "metadata.json";

    std::ifstream meta_in(metadata_file);
    if (!meta_in.is_open()) {
        throw std::runtime_error("Could not open metadata.json");
    }

    nlohmann::json meta;
    meta_in >> meta;

    std::string method = meta.value("method", "RHF");
    std::string basis_name;

    if (json_contains(meta, "basis")) {
        basis_name = meta.at("basis").get<std::string>();
    } else if (json_contains(meta, "BasisSet")) {
        basis_name = meta.at("BasisSet").get<std::string>();
    } else {
        throw std::runtime_error("metadata.json missing basis/BasisSet");
    }

    std::vector<basis_function> basis =
        basis_recon(basis_file.string(), opt.basis_path, basis_name);

    int K = static_cast<int>(basis.size());

    std::vector<atom> atoms =
        read_geometry_atoms(geometry_file.string());

    rapidcsv::Document density_doc(density_file.string());
    rapidcsv::Document coeff_doc(coeff_file.string());

    std::vector<double> density;
    std::vector<double> density_alpha;
    std::vector<double> density_beta;

    std::vector<double> coeff;
    std::vector<double> coeff_alpha;
    std::vector<double> coeff_beta;

    std::pair<std::vector<double>, std::vector<double>> coeff_pair;
    std::pair<int, int> occupation;

    if (method == "RHF") {
        density = density_doc.GetColumn<double>("density_matrix");
        coeff = coeff_doc.GetColumn<double>("coefficient_matrix");

        coeff_pair = {coeff, coeff};

        int Ne = meta.at("electron_number").get<int>();
        int n_occ = Ne / 2;

        occupation = {n_occ, n_occ};
    }
    else if (method == "UHF") {
        density_alpha = density_doc.GetColumn<double>("density_alpha");
        density_beta = density_doc.GetColumn<double>("density_beta");

        coeff_alpha = coeff_doc.GetColumn<double>("coefficient_alpha");
        coeff_beta = coeff_doc.GetColumn<double>("coefficient_beta");

        if (density_alpha.size() != density_beta.size()) {
            throw std::runtime_error("UHF density column size mismatch");
        }

        density.resize(density_alpha.size());

        for (size_t i = 0; i < density.size(); ++i) {
            density[i] = density_alpha[i] + density_beta[i];
        }

        coeff_pair = {coeff_alpha, coeff_beta};

        int Na = meta.at("alpha_occupancy").get<int>();
        int Nb = meta.at("beta_occupancy").get<int>();

        occupation = {Na, Nb};
    }
    else {
        throw std::runtime_error("Unsupported method in metadata.json");
    }

    if (density.size() != static_cast<size_t>(K) * K) {
        throw std::runtime_error("density matrix size does not match basis size");
    }

    if (coeff_pair.first.size() != static_cast<size_t>(K) * K ||
        coeff_pair.second.size() != static_cast<size_t>(K) * K) {
        throw std::runtime_error("coefficient matrix size does not match basis size");
    }

    std::vector<std::string> produced_files;

    if (opt.dimension == "2D") {

        if (opt.render_density) {
            std::vector<double> rho =
                density2d(basis, density,
                          opt.center,
                          opt.side_length,
                          opt.plane,
                          opt.resolution,
                          opt.render_hardware);

            fs::path file = out_dir / ("density_" + opt.plane + ".csv");

            save_matrix_csv(file.string(),
                            rho,
                            opt.resolution,
                            opt.resolution);

            produced_files.push_back(file.filename().string());
        }

        if (method == "RHF") {
            if (opt.render_homo_minus_1) {
                int idx = orbital_index_from_homo_offset(occupation.first,
                                                         -1, K);

                std::vector<double> psi =
                    orbital_2d(basis, coeff_pair.first, idx,
                               opt.center,
                               opt.side_length,
                               opt.plane,
                               opt.resolution,
                               opt.render_hardware);

                fs::path file =
                    out_dir / ("homo_minus_1_" + opt.plane + ".csv");

                save_matrix_csv(file.string(),
                                psi,
                                opt.resolution,
                                opt.resolution);

                produced_files.push_back(file.filename().string());
            }

            if (opt.render_homo) {
                std::vector<double> psi =
                    homo_lumo_2d(basis, coeff_pair, occupation,
                                 opt.center,
                                 opt.side_length,
                                 opt.plane,
                                 opt.resolution,
                                 true,
                                 true,
                                 opt.render_hardware);

                fs::path file = out_dir / ("homo_" + opt.plane + ".csv");

                save_matrix_csv(file.string(),
                                psi,
                                opt.resolution,
                                opt.resolution);

                produced_files.push_back(file.filename().string());
            }

            if (opt.render_lumo) {
                std::vector<double> psi =
                    homo_lumo_2d(basis, coeff_pair, occupation,
                                 opt.center,
                                 opt.side_length,
                                 opt.plane,
                                 opt.resolution,
                                 false,
                                 true,
                                 opt.render_hardware);

                fs::path file = out_dir / ("lumo_" + opt.plane + ".csv");

                save_matrix_csv(file.string(),
                                psi,
                                opt.resolution,
                                opt.resolution);

                produced_files.push_back(file.filename().string());
            }

            if (opt.render_lumo_plus_1) {
                int idx = orbital_index_from_homo_offset(occupation.first,
                                                         2, K);

                std::vector<double> psi =
                    orbital_2d(basis, coeff_pair.first, idx,
                               opt.center,
                               opt.side_length,
                               opt.plane,
                               opt.resolution,
                               opt.render_hardware);

                fs::path file =
                    out_dir / ("lumo_plus_1_" + opt.plane + ".csv");

                save_matrix_csv(file.string(),
                                psi,
                                opt.resolution,
                                opt.resolution);

                produced_files.push_back(file.filename().string());
            }
        }
        else {
            if (opt.render_homo_minus_1_alpha) {
                int idx = orbital_index_from_homo_offset(occupation.first,
                                                         -1, K);

                std::vector<double> psi =
                    orbital_2d(basis, coeff_pair.first, idx,
                               opt.center,
                               opt.side_length,
                               opt.plane,
                               opt.resolution,
                               opt.render_hardware);

                fs::path file =
                    out_dir /
                    ("homo_minus_1_alpha_" + opt.plane + ".csv");

                save_matrix_csv(file.string(),
                                psi,
                                opt.resolution,
                                opt.resolution);

                produced_files.push_back(file.filename().string());
            }

            if (opt.render_homo_alpha) {
                std::vector<double> psi =
                    homo_lumo_2d(basis, coeff_pair, occupation,
                                 opt.center,
                                 opt.side_length,
                                 opt.plane,
                                 opt.resolution,
                                 true,
                                 true,
                                 opt.render_hardware);

                fs::path file = out_dir / ("homo_alpha_" + opt.plane + ".csv");

                save_matrix_csv(file.string(),
                                psi,
                                opt.resolution,
                                opt.resolution);

                produced_files.push_back(file.filename().string());
            }

            if (opt.render_lumo_alpha) {
                std::vector<double> psi =
                    homo_lumo_2d(basis, coeff_pair, occupation,
                                 opt.center,
                                 opt.side_length,
                                 opt.plane,
                                 opt.resolution,
                                 false,
                                 true,
                                 opt.render_hardware);

                fs::path file = out_dir / ("lumo_alpha_" + opt.plane + ".csv");

                save_matrix_csv(file.string(),
                                psi,
                                opt.resolution,
                                opt.resolution);

                produced_files.push_back(file.filename().string());
            }

            if (opt.render_lumo_plus_1_alpha) {
                int idx = orbital_index_from_homo_offset(occupation.first,
                                                         2, K);

                std::vector<double> psi =
                    orbital_2d(basis, coeff_pair.first, idx,
                               opt.center,
                               opt.side_length,
                               opt.plane,
                               opt.resolution,
                               opt.render_hardware);

                fs::path file =
                    out_dir /
                    ("lumo_plus_1_alpha_" + opt.plane + ".csv");

                save_matrix_csv(file.string(),
                                psi,
                                opt.resolution,
                                opt.resolution);

                produced_files.push_back(file.filename().string());
            }

            if (opt.render_homo_minus_1_beta) {
                int idx = orbital_index_from_homo_offset(occupation.second,
                                                         -1, K);

                std::vector<double> psi =
                    orbital_2d(basis, coeff_pair.second, idx,
                               opt.center,
                               opt.side_length,
                               opt.plane,
                               opt.resolution,
                               opt.render_hardware);

                fs::path file =
                    out_dir /
                    ("homo_minus_1_beta_" + opt.plane + ".csv");

                save_matrix_csv(file.string(),
                                psi,
                                opt.resolution,
                                opt.resolution);

                produced_files.push_back(file.filename().string());
            }

            if (opt.render_homo_beta) {
                std::vector<double> psi =
                    homo_lumo_2d(basis, coeff_pair, occupation,
                                 opt.center,
                                 opt.side_length,
                                 opt.plane,
                                 opt.resolution,
                                 true,
                                 false,
                                 opt.render_hardware);

                fs::path file = out_dir / ("homo_beta_" + opt.plane + ".csv");

                save_matrix_csv(file.string(),
                                psi,
                                opt.resolution,
                                opt.resolution);

                produced_files.push_back(file.filename().string());
            }

            if (opt.render_lumo_beta) {
                std::vector<double> psi =
                    homo_lumo_2d(basis, coeff_pair, occupation,
                                 opt.center,
                                 opt.side_length,
                                 opt.plane,
                                 opt.resolution,
                                 false,
                                 false,
                                 opt.render_hardware);

                fs::path file = out_dir / ("lumo_beta_" + opt.plane + ".csv");

                save_matrix_csv(file.string(),
                                psi,
                                opt.resolution,
                                opt.resolution);

                produced_files.push_back(file.filename().string());
            }

            if (opt.render_lumo_plus_1_beta) {
                int idx = orbital_index_from_homo_offset(occupation.second,
                                                         2, K);

                std::vector<double> psi =
                    orbital_2d(basis, coeff_pair.second, idx,
                               opt.center,
                               opt.side_length,
                               opt.plane,
                               opt.resolution,
                               opt.render_hardware);

                fs::path file =
                    out_dir /
                    ("lumo_plus_1_beta_" + opt.plane + ".csv");

                save_matrix_csv(file.string(),
                                psi,
                                opt.resolution,
                                opt.resolution);

                produced_files.push_back(file.filename().string());
            }
        }
    }

    else if (opt.dimension == "3D") {

        if (opt.render_density) {
            cube_data cube =
                density3d(basis,
                          density,
                          opt.center,
                          opt.side_length,
                          opt.resolution,
                          opt.render_hardware);

            fs::path file = out_dir / "density.cube";

            write_cube_file(file.string(), cube, atoms,
                            "Electron density");

            produced_files.push_back(file.filename().string());
        }

        if (method == "RHF") {
            if (opt.render_homo_minus_1) {
                int idx = orbital_index_from_homo_offset(occupation.first,
                                                         -1, K);

                cube_data cube =
                    orbital_3d(basis, coeff_pair.first, idx,
                               opt.center,
                               opt.side_length,
                               opt.resolution,
                               opt.render_hardware);

                fs::path file = out_dir / "homo_minus_1.cube";

                write_cube_file(file.string(), cube, atoms,
                                "RHF HOMO-1");

                produced_files.push_back(file.filename().string());
            }

            if (opt.render_homo) {
                cube_data cube =
                    homo_lumo_3d(basis, coeff_pair, occupation,
                                 opt.center,
                                 opt.side_length,
                                 opt.resolution,
                                 true,
                                 true,
                                 opt.render_hardware);

                fs::path file = out_dir / "homo.cube";

                write_cube_file(file.string(), cube, atoms,
                                "RHF HOMO");

                produced_files.push_back(file.filename().string());
            }

            if (opt.render_lumo) {
                cube_data cube =
                    homo_lumo_3d(basis, coeff_pair, occupation,
                                 opt.center,
                                 opt.side_length,
                                 opt.resolution,
                                 false,
                                 true,
                                 opt.render_hardware);

                fs::path file = out_dir / "lumo.cube";

                write_cube_file(file.string(), cube, atoms,
                                "RHF LUMO");

                produced_files.push_back(file.filename().string());
            }

            if (opt.render_lumo_plus_1) {
                int idx = orbital_index_from_homo_offset(occupation.first,
                                                         2, K);

                cube_data cube =
                    orbital_3d(basis, coeff_pair.first, idx,
                               opt.center,
                               opt.side_length,
                               opt.resolution,
                               opt.render_hardware);

                fs::path file = out_dir / "lumo_plus_1.cube";

                write_cube_file(file.string(), cube, atoms,
                                "RHF LUMO+1");

                produced_files.push_back(file.filename().string());
            }
        }
        else {
            if (opt.render_homo_minus_1_alpha) {
                int idx = orbital_index_from_homo_offset(occupation.first,
                                                         -1, K);

                cube_data cube =
                    orbital_3d(basis, coeff_pair.first, idx,
                               opt.center,
                               opt.side_length,
                               opt.resolution,
                               opt.render_hardware);

                fs::path file = out_dir / "homo_minus_1_alpha.cube";

                write_cube_file(file.string(), cube, atoms,
                                "UHF alpha HOMO-1");

                produced_files.push_back(file.filename().string());
            }

            if (opt.render_homo_alpha) {
                cube_data cube =
                    homo_lumo_3d(basis, coeff_pair, occupation,
                                 opt.center,
                                 opt.side_length,
                                 opt.resolution,
                                 true,
                                 true,
                                 opt.render_hardware);

                fs::path file = out_dir / "homo_alpha.cube";

                write_cube_file(file.string(), cube, atoms,
                                "UHF alpha HOMO");

                produced_files.push_back(file.filename().string());
            }

            if (opt.render_lumo_alpha) {
                cube_data cube =
                    homo_lumo_3d(basis, coeff_pair, occupation,
                                 opt.center,
                                 opt.side_length,
                                 opt.resolution,
                                 false,
                                 true,
                                 opt.render_hardware);

                fs::path file = out_dir / "lumo_alpha.cube";

                write_cube_file(file.string(), cube, atoms,
                                "UHF alpha LUMO");

                produced_files.push_back(file.filename().string());
            }

            if (opt.render_lumo_plus_1_alpha) {
                int idx = orbital_index_from_homo_offset(occupation.first,
                                                         2, K);

                cube_data cube =
                    orbital_3d(basis, coeff_pair.first, idx,
                               opt.center,
                               opt.side_length,
                               opt.resolution,
                               opt.render_hardware);

                fs::path file = out_dir / "lumo_plus_1_alpha.cube";

                write_cube_file(file.string(), cube, atoms,
                                "UHF alpha LUMO+1");

                produced_files.push_back(file.filename().string());
            }

            if (opt.render_homo_minus_1_beta) {
                int idx = orbital_index_from_homo_offset(occupation.second,
                                                         -1, K);

                cube_data cube =
                    orbital_3d(basis, coeff_pair.second, idx,
                               opt.center,
                               opt.side_length,
                               opt.resolution,
                               opt.render_hardware);

                fs::path file = out_dir / "homo_minus_1_beta.cube";

                write_cube_file(file.string(), cube, atoms,
                                "UHF beta HOMO-1");

                produced_files.push_back(file.filename().string());
            }

            if (opt.render_homo_beta) {
                cube_data cube =
                    homo_lumo_3d(basis, coeff_pair, occupation,
                                 opt.center,
                                 opt.side_length,
                                 opt.resolution,
                                 true,
                                 false,
                                 opt.render_hardware);

                fs::path file = out_dir / "homo_beta.cube";

                write_cube_file(file.string(), cube, atoms,
                                "UHF beta HOMO");

                produced_files.push_back(file.filename().string());
            }

            if (opt.render_lumo_beta) {
                cube_data cube =
                    homo_lumo_3d(basis, coeff_pair, occupation,
                                 opt.center,
                                 opt.side_length,
                                 opt.resolution,
                                 false,
                                 false,
                                 opt.render_hardware);

                fs::path file = out_dir / "lumo_beta.cube";

                write_cube_file(file.string(), cube, atoms,
                                "UHF beta LUMO");

                produced_files.push_back(file.filename().string());
            }

            if (opt.render_lumo_plus_1_beta) {
                int idx = orbital_index_from_homo_offset(occupation.second,
                                                         2, K);

                cube_data cube =
                    orbital_3d(basis, coeff_pair.second, idx,
                               opt.center,
                               opt.side_length,
                               opt.resolution,
                               opt.render_hardware);

                fs::path file = out_dir / "lumo_plus_1_beta.cube";

                write_cube_file(file.string(), cube, atoms,
                                "UHF beta LUMO+1");

                produced_files.push_back(file.filename().string());
            }
        }
    }

    nlohmann::ordered_json info;

    info["source_folder"] = opt.data_path;
    info["save_folder"] = out_dir.string();

    info["method"] = method;
    info["basis"] = basis_name;
    info["basis_size"] = K;

    info["dimension"] = opt.dimension;
    info["resolution"] = opt.resolution;

    info["center"] = {
        opt.center.x,
        opt.center.y,
        opt.center.z
    };

    info["side_length"] = opt.side_length;

    if (opt.dimension == "2D") {
        info["plane"] = opt.plane;
    }

    info["render_hardware"] = opt.render_hardware;

    info["render_density"] = opt.render_density;

    info["render_homo_minus_1"] = opt.render_homo_minus_1;
    info["render_homo"] = opt.render_homo;
    info["render_lumo"] = opt.render_lumo;
    info["render_lumo_plus_1"] = opt.render_lumo_plus_1;

    info["render_homo_minus_1_alpha"] = opt.render_homo_minus_1_alpha;
    info["render_homo_alpha"] = opt.render_homo_alpha;
    info["render_lumo_alpha"] = opt.render_lumo_alpha;
    info["render_lumo_plus_1_alpha"] = opt.render_lumo_plus_1_alpha;

    info["render_homo_minus_1_beta"] = opt.render_homo_minus_1_beta;
    info["render_homo_beta"] = opt.render_homo_beta;
    info["render_lumo_beta"] = opt.render_lumo_beta;
    info["render_lumo_plus_1_beta"] = opt.render_lumo_plus_1_beta;

    info["files"] = produced_files;

    std::ofstream js(out_dir / "visualization_info.json");

    if (!js.is_open()) {
        throw std::runtime_error("Could not write visualization_info.json");
    }

    js << std::setw(4) << info << "\n";
    js.close();

    std::cout << "Visualization files saved to: "
              << out_dir.string() << std::endl;
}
