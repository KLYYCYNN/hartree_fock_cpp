#include <iostream>
#include <exception>
#include "run.hpp"

int main(int argc, char* argv[])
{
    try {

        if (argc != 2) {
            std::cerr
                << "Usage: "
                << argv[0]
                << " <input.json>\n";

            return 1;
        }

        const std::string input_file = argv[1];

        std::cout << "Reading input file...\n";

        bl_scan_options opt =
            read_blscan_options(input_file);

        std::cout << "Starting bond-length scan...\n";

        run_bl_scan(opt);

        std::cout << "Bond-length scan completed successfully.\n";

        return 0;
    }
    catch (const std::exception& e) {

        std::cerr
            << "Error: "
            << e.what()
            << std::endl;

        return 1;
    }
}