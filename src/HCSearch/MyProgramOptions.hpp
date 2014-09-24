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
		enum FeaturesMode { STANDARD, STANDARD_CONTEXT, STANDARD_ALT, STANDARD_CONF, DENSE_CRF, UNARY, UNARY_CONF, STANDARD_PAIR_COUNTS, STANDARD_CONF_PAIR_COUNTS, STANDARD_PRUNE };
		enum InitialFunctionMode { LOG_REG };
		enum SuccessorsMode { FLIPBIT, FLIPBIT_NEIGHBORS, FLIPBIT_CONFIDENCES_NEIGHBORS, 
			STOCHASTIC, STOCHASTIC_NEIGHBORS, STOCHASTIC_CONFIDENCES_NEIGHBORS, 
			CUT_SCHEDULE, CUT_SCHEDULE_NEIGHBORS, CUT_SCHEDULE_CONFIDENCES_NEIGHBORS,
			STOCHASTIC_SCHEDULE, STOCHASTIC_SCHEDULE_NEIGHBORS, STOCHASTIC_SCHEDULE_CONFIDENCES_NEIGHBORS,
			STOCHASTIC_CONSTRAINED };
		enum LossMode { HAMMING, PIXEL_HAMMING };
		enum PruneMode { NO_PRUNE, RANKER_PRUNE, ORACLE_PRUNE, SIMULATED_RANKER_PRUNE };

		enum StochasticCutMode { STATE, EDGES };

		// input and output directories

		string baseDir;
		string inputDir;
		string outputDir;
		string splitsFolderName;
		string splitsTrainName;
		string splitsValidName;
		string splitsTestName;
		string nodesFolderName;
		string edgesFolderName;
		string edgeFeaturesFolderName;

		string logsFolderName;
		string modelsFolderName;
		string resultsFolderName;
		string tempFolderName;

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
		FeaturesMode pruneFeaturesMode;
		InitialFunctionMode initialFunctionMode;
		SuccessorsMode successorsMode;
		LossMode lossMode;

		StochasticCutMode stochasticCutMode;
		int beamSize;
		double cutParam;

		PruneMode pruneMode;

		bool saveAnytimePredictions;
		HCSearch::RankerType rankLearnerType;
		bool saveFeaturesFiles;
		int numTrainIterations;
		int numTestIterations;
		bool verboseMode;
		int uniqueIterId;
		bool saveOutputMask;
		double pruneRatio;
		double badPruneRatio;
		bool useEdgeWeights;

		bool nodeClamp;
		bool edgeClamp;
		double nodeClampThreshold;
		double edgeClampPositiveThreshold;
		double edgeClampNegativeThreshold;

		double lambda1;
		double lambda2;
		double lambda3;

		double maxCuttingThreshold;
		double minCuttingThreshold;

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