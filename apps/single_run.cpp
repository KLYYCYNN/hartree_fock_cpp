#include "run.hpp"


int main(int argc, char** argv)
{
    try {
        if (argc < 2) {
            std::cerr << "Usage: " << argv[0] << " input.json\n";
            return 1;
        }

        scf_options opt = read_scf_options(argv[1]);

        run_scf_from_options(opt);

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}