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
}