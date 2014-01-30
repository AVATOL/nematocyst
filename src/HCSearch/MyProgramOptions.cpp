#include <iostream>
#include "MyProgramOptions.hpp"

namespace MyProgramOptions
{
	ProgramOptions::ProgramOptions()
	{
		inputDir = "";
		outputDir = "";

		timeBound = 0;

		demoMode = false;
		schedule = vector< Modes >();
	}

	ProgramOptions ProgramOptions::parseArguments(int argc, char* argv[])
	{
		ProgramOptions po;

		//TODO
		po.inputDir = "DataPreprocessed/NematoBinary";
		po.outputDir = "Results/NematoExperiment";

		return po;
	}

	void ProgramOptions::printUsage(int argc, char* argv[])
	{
		cerr << endl;
		cerr << "Program usage: ./HCSearch INPUT_DIR OUTPUT_DIR TIMEBOUND "
			<< "[--demo] [--learn=(H|C|COH)]* [--infer=(HC|HL|LC|LL)]* ... [--option=value]" << endl;

		cerr << "Allowed options:" << endl;
		cerr << "\t--help\t\t" << ": produce help message" << endl;
		cerr << "\t--demo\t\t" << ": run the demo program (ignores --learn and --infer)" << endl;
		cerr << "\t--learn=arg\t" << ": learning" << endl;
		cerr << "\t\t\t\tH: learn heuristic" << endl;
		cerr << "\t\t\t\tC: learn cost" << endl;
		cerr << "\t\t\t\tCOH: learn cost with oracle H" << endl;
		cerr << "\t\t\t\tALL: short-hand for H, C, COH in that order" << endl;
		cerr << "\t--infer=arg\t" << ": inference" << endl;
		cerr << "\t\t\t\tHC: learned heuristic and cost" << endl;
		cerr << "\t\t\t\tHL: learned heuristic and oracle cost" << endl;
		cerr << "\t\t\t\tLC: oracle heuristic and learned cost" << endl;
		cerr << "\t\t\t\tLL: oracle heuristic and cost" << endl;
		cerr << "\t\t\t\tALL: short-hand for HC, HL, LC, LL in that order" << endl;
		cerr << endl;

		cerr << "Notes:" << endl;
		cerr << "* The first three arguments are required. They are the input directory, output directory and time bound." << endl;
		cerr << "* Can use multiple --infer and --learn options in any order to define a schedule. Must come after the mandatory arguments.";
		cerr << endl;
	}
}