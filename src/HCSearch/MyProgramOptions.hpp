#pragma once
#include <vector>
#include <string>

using namespace std;

namespace MyProgramOptions
{
	/*!
	 * @brief Stores program options.
	 */
	class ProgramOptions
	{
	public:
		// constants

		const enum Modes { LEARN_H, LEARN_C, LEARN_C_ORACLE_H, 
			INFER_HC, INFER_HL, INFER_LC, INFER_LL };

		const enum SearchProcedureMode { GREEDY, BREADTH_BEAM, BEST_BEAM };
		const enum FeaturesMode { STANDARD };
		const enum InitialFunctionMode { LOG_REG };
		const enum SuccessorsMode { FLIPBIT, STOCHASTIC };
		const enum LossMode { HAMMING };

		const enum StochasticCutMode { STATE, EDGES };

		// input and output directories

		string inputDir;
		string outputDir;

		// time bound

		int timeBound;

		// schedule or demo

		bool demoMode;
		vector< Modes > schedule;

		// options

		SearchProcedureMode searchProcedureMode;
		FeaturesMode heuristicFeaturesMode;
		FeaturesMode costFeaturesMode;
		InitialFunctionMode initialFunctionMode;
		SuccessorsMode successorsMode;
		LossMode lossMode;

		StochasticCutMode stochasticCutMode;
		int beamSize;
		double cutParam;

	public:
		ProgramOptions();

		/*!
		 * @brief Parse command line arguments to get program options.
		 */
		static ProgramOptions parseArguments(int argc, char* argv[]);

		/*!
		 * @brief Print command line usage message.
		 */
		static void printUsage();
	};
}