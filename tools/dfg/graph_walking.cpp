#include <iostream>
#include <iomanip>

#include <dfa/dfa.hpp>
#include <util/data_file.hpp>

int main(int argc, char** argv) {
    using namespace sw::dfa;

    if (argc != 2) {
	    std::cerr << "Usage: " << argv[0] << " <DFG file>\n";
        return EXIT_SUCCESS; // exit with success for CI purposes
    }

    std::string dataFileName{ argv[1] };
    if (!std::filesystem::exists(dataFileName)) {
        // search for the file in the data directory
        try {
            dataFileName = generateDataFile(argv[1]);
            std::cout << "Data: " << dataFileName << std::endl;
        }
        catch (const std::runtime_error& e) {
            std::cerr << "Error: " << e.what() << std::endl;
            return EXIT_SUCCESS; // exit with success for CI purposes
        }
    }

    DomainFlowGraph dfg(dataFileName); // Deep Learning graph
    dfg.load(dataFileName);

	// walk the graph
	for (const auto& [nodeId, node] : dfg.graph.nodes()) {
		std::cout << "Node ID: " << nodeId << ", Name: " << node.getName() << " Depth: " << node.getDepth() << std::endl;
		std::cout << "  Operator: " << node.getOperator() << std::endl;
		if (node.getNrInputs() > 0) {
            std::cout << "  Inputs:\n";
            for (int i = 0; i < node.getNrInputs(); ++i) {
                std::cout << "    Operand   : " << i << " : " << node.getOperandType(i) << '\n';
            }
		}

		if (node.getNrAttributes() > 0) {
			std::cout << "  Attributes:\n";
			for (auto& pair : node.getAttributes()) {
				std::cout << "    Attribute : " << pair.first << " : " << pair.second << '\n';
			}
		}

        if (node.getNrOutputs() > 0) {
            std::cout << "  Outputs:\n";
            for (int i = 0; i < node.getNrOutputs(); ++i) {
                std::cout << "    Result    : " << i << " : " << node.getResultType(i) << '\n';
            }
        }

	}

    // Report on the SOURCES and SINKS
    bool first = true;
    std::cout << "SOURCE: ";
    for (const auto& srcId : dfg.source) {
        if (!first) std::cout << ", ";
        std::cout << srcId;
        first = false;
    }
    first = true;
    std::cout << "\nSINK: ";
    for (const auto& snkId : dfg.sink) {
        if (!first) std::cout << ", ";
        std::cout << snkId;
        first = false;
    }

    return EXIT_SUCCESS;
}
