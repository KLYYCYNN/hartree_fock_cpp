#include <iostream>
#include <exception>
#include "visualization.hpp"

int main(int argc, char* argv[])
{
    try {
        if (argc != 2) {
            std::cerr << "Usage: "
                      << argv[0]
                      << " <visual_options.json>\n";
            return 1;
        }

        std::string input_file = argv[1];

        std::cout << "Reading visualization input...\n";

        visual_options opt = read_visual_options(input_file);

        std::cout << "Starting visualization...\n";

        run_visualization(opt);

        std::cout << "Visualization completed successfully.\n";

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "Error: "
                  << e.what()
                  << std::endl;

        return 1;
    }
}