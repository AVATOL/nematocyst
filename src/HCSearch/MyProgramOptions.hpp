#pragma once
#include <vector>
#include <string>
#include "DataStructures.hpp"

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

		enum SearchProcedureMode { GREEDY, BREADTH_BEAM, BEST_BEAM };
		enum FeaturesMode { STANDARD };
		enum InitialFunctionMode { LOG_REG };
		enum SuccessorsMode { FLIPBIT, FLIPBIT_NEIGHBORS, 
			STOCHASTIC, STOCHASTIC_NEIGHBORS, STOCHASTIC_CONFIDENCES_NEIGHBORS, 
			CUT_SCHEDULE, CUT_SCHEDULE_NEIGHBORS, CUT_SCHEDULE_CONFIDENCES_NEIGHBORS };
		enum LossMode { HAMMING };

		enum StochasticCutMode { STATE, EDGES };

		// input and output directories

		string inputDir;
		string outputDir;
		string splitsFolderName;

		// time bound

		int timeBound;

		// schedule or demo

		bool printUsageMode;
		bool demoMode;
		vector< HCSearch::SearchType > schedule;

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

		bool saveAnytimePredictions;
		HCSearch::RankerType rankLearnerType;
		bool saveFeaturesFiles;
		int numTrainIterations;
		int numTestIterations;
		bool verboseMode;
		int boundSuccessorCandidates;
		int uniqueIterId;

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