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
		enum FeaturesMode { STANDARD, STANDARD2, STANDARD3, DENSE_CRF, UNARY, UNARY2, STANDARD_BIGRAM, STANDARD3_BIGRAM };
		enum InitialFunctionMode { LOG_REG };
		enum SuccessorsMode { FLIPBIT, FLIPBIT_NEIGHBORS, FLIPBIT_CONFIDENCES_NEIGHBORS, 
			STOCHASTIC, STOCHASTIC_NEIGHBORS, STOCHASTIC_CONFIDENCES_NEIGHBORS, 
			CUT_SCHEDULE, CUT_SCHEDULE_NEIGHBORS, CUT_SCHEDULE_CONFIDENCES_NEIGHBORS };
		enum LossMode { HAMMING, PIXEL_HAMMING };

		enum StochasticCutMode { STATE, EDGES };

		// input and output directories

		string inputDir;
		string outputDir;
		string splitsFolderName;
		string splitsTrainName;
		string splitsValidName;
		string splitsTestName;

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
		bool saveOutputMask;

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