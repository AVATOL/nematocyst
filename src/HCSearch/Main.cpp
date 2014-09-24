#include <iostream>
#include <ctime>
#include "../HCSearchLib/HCSearch.hpp"
#include "MyFileSystem.hpp"
#include "MyProgramOptions.hpp"
#include "Main.hpp"
#include "Demo.hpp"

using namespace std;
using namespace MyLogger;

int main(int argc, char* argv[])
{
	// initialize HCSearch
	HCSearch::Setup::initialize(argc, argv);

	// parse arguments
	MyProgramOptions::ProgramOptions po = MyProgramOptions::ProgramOptions::parseArguments(argc, argv);

	// print usage
	if (po.printUsageMode)
	{
		if (HCSearch::Global::settings->RANK == 0)
			MyProgramOptions::ProgramOptions::printUsage();
		HCSearch::Setup::finalize();
		return 0;
	}

	// configure settings
	HCSearch::Global::settings->paths->BASE_PATH = po.baseDir;
	HCSearch::Global::settings->paths->INPUT_SPLITS_FOLDER_NAME = po.splitsFolderName;
	HCSearch::Global::settings->paths->INPUT_SPLITS_TRAIN_FILE_BASE = po.splitsTrainName;
	HCSearch::Global::settings->paths->INPUT_SPLITS_VALIDATION_FILE_BASE = po.splitsValidName;
	HCSearch::Global::settings->paths->INPUT_SPLITS_TEST_FILE_BASE = po.splitsTestName;
	HCSearch::Global::settings->paths->INPUT_NODES_FOLDER_NAME = po.nodesFolderName;
	HCSearch::Global::settings->paths->INPUT_EDGES_FOLDER_NAME = po.edgesFolderName;
	HCSearch::Global::settings->paths->INPUT_EDGE_FEATURES_FOLDER_NAME = po.edgeFeaturesFolderName;

	HCSearch::Global::settings->paths->OUTPUT_LOGS_FOLDER_NAME = po.logsFolderName;
	HCSearch::Global::settings->paths->OUTPUT_MODELS_FOLDER_NAME = po.modelsFolderName;
	HCSearch::Global::settings->paths->OUTPUT_RESULTS_FOLDER_NAME = po.resultsFolderName;
	HCSearch::Global::settings->paths->OUTPUT_TEMP_FOLDER_NAME = po.tempFolderName;

	HCSearch::Global::settings->USE_EDGE_WEIGHTS = po.useEdgeWeights;

	HCSearch::Setup::configure(po.inputDir, po.outputDir);
	if (po.verboseMode)
		Logger::setLogLevel(DEBUG);

	// print useful information
	printInfo(po);

	// demo or run full program
	if (po.demoMode)
		demo(po.timeBound);
	else
		run(po);

	// finalize
	HCSearch::Setup::finalize();
    return 0;
}

HCSearch::SearchSpace* setupSearchSpace(MyProgramOptions::ProgramOptions po)
{
	LOG() << "=== Search Space ===" << endl;

	// select some loss function
	LOG() << "Loss function: ";
	HCSearch::ILossFunction* lossFunc = NULL;
	switch (po.lossMode)
	{
	case MyProgramOptions::ProgramOptions::HAMMING:
		LOG() << "Hamming loss" << endl;
		lossFunc = new HCSearch::HammingLoss();
		break;
	case MyProgramOptions::ProgramOptions::PIXEL_HAMMING:
		LOG() << "Pixel Hamming loss" << endl;
		lossFunc = new HCSearch::PixelHammingLoss();
		break;
	default:
		LOG(ERROR) << "undefined loss mode.";
	}

	// select heuristic feature function
	LOG() << "Heuristic feature function: ";
	HCSearch::IFeatureFunction* heuristicFeatFunc = NULL;
	switch (po.heuristicFeaturesMode)
	{
	case MyProgramOptions::ProgramOptions::STANDARD:
		LOG() << "standard CRF features" << endl;
		LOG() << "\tlambda1=" << po.lambda1 << endl;
		LOG() << "\tlambda2=" << po.lambda2 << endl;
		LOG() << "\tlambda3=" << po.lambda3 << endl;
		heuristicFeatFunc = new HCSearch::StandardFeatures(po.lambda1, po.lambda2, po.lambda3);
		break;
	case MyProgramOptions::ProgramOptions::STANDARD_CONTEXT:
		LOG() << "standard context CRF features" << endl;
		heuristicFeatFunc = new HCSearch::StandardContextFeatures();
		break;
	case MyProgramOptions::ProgramOptions::STANDARD_ALT:
		LOG() << "standard 2 CRF features" << endl;
		heuristicFeatFunc = new HCSearch::StandardAltFeatures();
		break;
	case MyProgramOptions::ProgramOptions::STANDARD_CONF:
		LOG() << "standard 3 CRF features" << endl;
		heuristicFeatFunc = new HCSearch::StandardConfFeatures();
		break;
	case MyProgramOptions::ProgramOptions::UNARY:
		LOG() << "unary CRF features" << endl;
		heuristicFeatFunc = new HCSearch::UnaryFeatures();
		break;
	case MyProgramOptions::ProgramOptions::UNARY_CONF:
		LOG() << "unary confidences CRF features" << endl;
		heuristicFeatFunc = new HCSearch::UnaryConfFeatures();
		break;
	case MyProgramOptions::ProgramOptions::STANDARD_PAIR_COUNTS:
		LOG() << "pairwise bigram CRF features" << endl;
		heuristicFeatFunc = new HCSearch::StandardPairwiseCountsFeatures();
		break;
	case MyProgramOptions::ProgramOptions::STANDARD_CONF_PAIR_COUNTS:
		LOG() << "pairwise bigram confidences CRF features" << endl;
		heuristicFeatFunc = new HCSearch::StandardConfPairwiseCountsFeatures();
		break;
	case MyProgramOptions::ProgramOptions::DENSE_CRF:
		LOG() << "dense CRF features" << endl;
		heuristicFeatFunc = new HCSearch::DenseCRFFeatures();
		break;
	default:
		LOG(ERROR) << "undefined feature mode.";
	}

	// select cost feature function
	LOG() << "Cost feature function: ";
	HCSearch::IFeatureFunction* costFeatFunc = NULL;
	switch (po.heuristicFeaturesMode)
	{
	case MyProgramOptions::ProgramOptions::STANDARD:
		LOG() << "standard CRF features" << endl;
		LOG() << "\tlambda1=" << po.lambda1 << endl;
		LOG() << "\tlambda2=" << po.lambda2 << endl;
		LOG() << "\tlambda3=" << po.lambda3 << endl;
		costFeatFunc = new HCSearch::StandardFeatures(po.lambda1, po.lambda2, po.lambda3);
		break;
	case MyProgramOptions::ProgramOptions::STANDARD_CONTEXT:
		LOG() << "standard context CRF features" << endl;
		costFeatFunc = new HCSearch::StandardContextFeatures();
		break;
	case MyProgramOptions::ProgramOptions::STANDARD_ALT:
		LOG() << "standard 2 CRF features" << endl;
		costFeatFunc = new HCSearch::StandardAltFeatures();
		break;
	case MyProgramOptions::ProgramOptions::STANDARD_CONF:
		LOG() << "standard 3 CRF features" << endl;
		costFeatFunc = new HCSearch::StandardConfFeatures();
		break;
	case MyProgramOptions::ProgramOptions::UNARY:
		LOG() << "unary CRF features" << endl;
		costFeatFunc = new HCSearch::UnaryFeatures();
		break;
	case MyProgramOptions::ProgramOptions::UNARY_CONF:
		LOG() << "unary confidences CRF features" << endl;
		costFeatFunc = new HCSearch::UnaryConfFeatures();
		break;
	case MyProgramOptions::ProgramOptions::STANDARD_PAIR_COUNTS:
		LOG() << "pairwise bigram CRF features" << endl;
		costFeatFunc = new HCSearch::StandardPairwiseCountsFeatures();
		break;
	case MyProgramOptions::ProgramOptions::STANDARD_CONF_PAIR_COUNTS:
		LOG() << "pairwise bigram confidences CRF features" << endl;
		costFeatFunc = new HCSearch::StandardConfPairwiseCountsFeatures();
		break;
	case MyProgramOptions::ProgramOptions::DENSE_CRF:
		LOG() << "dense CRF features" << endl;
		costFeatFunc = new HCSearch::DenseCRFFeatures();
		break;
	default:
		LOG(ERROR) << "undefined feature mode.";
	}

	// select prune feature function
	LOG() << "Prune feature function: ";
	HCSearch::IFeatureFunction* pruneFeatFunc = NULL;
	switch (po.pruneFeaturesMode)
	{
	case MyProgramOptions::ProgramOptions::STANDARD:
		LOG() << "standard CRF features" << endl;
		LOG() << "\tlambda1=" << po.lambda1 << endl;
		LOG() << "\tlambda2=" << po.lambda2 << endl;
		LOG() << "\tlambda3=" << po.lambda3 << endl;
		pruneFeatFunc = new HCSearch::StandardFeatures(po.lambda1, po.lambda2, po.lambda3);
		break;
	case MyProgramOptions::ProgramOptions::STANDARD_CONTEXT:
		LOG() << "standard context CRF features" << endl;
		pruneFeatFunc = new HCSearch::StandardContextFeatures();
		break;
	case MyProgramOptions::ProgramOptions::STANDARD_ALT:
		LOG() << "standard 2 CRF features" << endl;
		pruneFeatFunc = new HCSearch::StandardAltFeatures();
		break;
	case MyProgramOptions::ProgramOptions::STANDARD_CONF:
		LOG() << "standard 3 CRF features" << endl;
		pruneFeatFunc = new HCSearch::StandardConfFeatures();
		break;
	case MyProgramOptions::ProgramOptions::UNARY:
		LOG() << "unary CRF features" << endl;
		pruneFeatFunc = new HCSearch::UnaryFeatures();
		break;
	case MyProgramOptions::ProgramOptions::UNARY_CONF:
		LOG() << "unary confidences CRF features" << endl;
		pruneFeatFunc = new HCSearch::UnaryConfFeatures();
		break;
	case MyProgramOptions::ProgramOptions::STANDARD_PAIR_COUNTS:
		LOG() << "pairwise bigram CRF features" << endl;
		pruneFeatFunc = new HCSearch::StandardPairwiseCountsFeatures();
		break;
	case MyProgramOptions::ProgramOptions::STANDARD_CONF_PAIR_COUNTS:
		LOG() << "pairwise bigram confidences CRF features" << endl;
		pruneFeatFunc = new HCSearch::StandardConfPairwiseCountsFeatures();
		break;
	case MyProgramOptions::ProgramOptions::DENSE_CRF:
		LOG() << "dense CRF features" << endl;
		pruneFeatFunc = new HCSearch::DenseCRFFeatures();
		break;
	case MyProgramOptions::ProgramOptions::STANDARD_PRUNE:
		LOG() << "standard prune features" << endl;
		pruneFeatFunc = new HCSearch::StandardPruneFeatures();
		break;
	default:
		LOG(ERROR) << "undefined feature mode.";
	}

	// select some successor function
	LOG() << "Successor function: ";
	HCSearch::ISuccessorFunction* successor = NULL;
	bool cutEdgesIndependently = po.stochasticCutMode == MyProgramOptions::ProgramOptions::EDGES;
	switch (po.successorsMode)
	{
	case MyProgramOptions::ProgramOptions::FLIPBIT:
		LOG() << "flipbit" << endl;
		successor = new HCSearch::FlipbitSuccessor();
		break;
	case MyProgramOptions::ProgramOptions::FLIPBIT_NEIGHBORS:
		LOG() << "flipbit neighbors" << endl;
		successor = new HCSearch::FlipbitNeighborSuccessor();
		break;
	case MyProgramOptions::ProgramOptions::FLIPBIT_CONFIDENCES_NEIGHBORS:
		LOG() << "flipbit confidences neighbors" << endl;
		successor = new HCSearch::FlipbitConfidencesNeighborSuccessor();
		break;
	case MyProgramOptions::ProgramOptions::STOCHASTIC:
		LOG() << "stochastic" << endl;
		LOG() << "\tCut edges independently: " << cutEdgesIndependently << endl;
		LOG() << "\tTemperature parameter: " << po.cutParam << endl;
		LOG() << "\tMax threshold for cutting: " << po.maxCuttingThreshold << endl;
		LOG() << "\tMin threshold for cutting: " << po.minCuttingThreshold << endl;
		successor = new HCSearch::StochasticSuccessor(cutEdgesIndependently, po.cutParam, po.maxCuttingThreshold, po.minCuttingThreshold);
		break;
	case MyProgramOptions::ProgramOptions::STOCHASTIC_NEIGHBORS:
		LOG() << "stochastic neighbors" << endl;
		LOG() << "\tCut edges independently: " << cutEdgesIndependently << endl;
		LOG() << "\tTemperature parameter: " << po.cutParam << endl;
		LOG() << "\tMax threshold for cutting: " << po.maxCuttingThreshold << endl;
		LOG() << "\tMin threshold for cutting: " << po.minCuttingThreshold << endl;
		successor = new HCSearch::StochasticNeighborSuccessor(cutEdgesIndependently, po.cutParam, po.maxCuttingThreshold, po.minCuttingThreshold);
		break;
	case MyProgramOptions::ProgramOptions::STOCHASTIC_CONFIDENCES_NEIGHBORS:
		LOG() << "stochastic confidences neighbors" << endl;
		LOG() << "\tCut edges independently: " << cutEdgesIndependently << endl;
		LOG() << "\tTemperature parameter: " << po.cutParam << endl;
		LOG() << "\tMax threshold for cutting: " << po.maxCuttingThreshold << endl;
		LOG() << "\tMin threshold for cutting: " << po.minCuttingThreshold << endl;
		successor = new HCSearch::StochasticConfidencesNeighborSuccessor(cutEdgesIndependently, po.cutParam, po.maxCuttingThreshold, po.minCuttingThreshold);
		break;
	case MyProgramOptions::ProgramOptions::CUT_SCHEDULE:
		LOG() << "cut schedule" << endl;
		LOG() << "\tTemperature parameter: " << po.cutParam << endl;
		successor = new HCSearch::CutScheduleSuccessor(po.cutParam);
		break;
	case MyProgramOptions::ProgramOptions::CUT_SCHEDULE_NEIGHBORS:
		LOG() << "cut schedule neighbors" << endl;
		LOG() << "\tTemperature parameter: " << po.cutParam << endl;
		successor = new HCSearch::CutScheduleNeighborSuccessor(po.cutParam);
		break;
	case MyProgramOptions::ProgramOptions::CUT_SCHEDULE_CONFIDENCES_NEIGHBORS:
		LOG() << "cut schedule confidences neighbors" << endl;
		LOG() << "\tTemperature parameter: " << po.cutParam << endl;
		successor = new HCSearch::CutScheduleConfidencesNeighborSuccessor(po.cutParam);
		break;
	case MyProgramOptions::ProgramOptions::STOCHASTIC_SCHEDULE:
		LOG() << "stochastic schedule" << endl;
		LOG() << "\tCut edges independently: " << cutEdgesIndependently << endl;
		LOG() << "\tTemperature parameter: " << po.cutParam << endl;
		LOG() << "\tNode clamping: " << po.nodeClamp << ", Edge clamping: " << po.edgeClamp << endl;
		LOG() << "\t\tNode clamp threshold: " << po.nodeClampThreshold << endl;
		LOG() << "\t\tEdge clamp positive threshold: " << po.edgeClampPositiveThreshold << endl;
		LOG() << "\t\tEdge clamp negative threshold: " << po.edgeClampNegativeThreshold << endl;
		successor = new HCSearch::StochasticScheduleSuccessor(cutEdgesIndependently, po.cutParam, 
			po.nodeClamp, po.edgeClamp, po.nodeClampThreshold, po.edgeClampPositiveThreshold, po.edgeClampNegativeThreshold);
		break;
	case MyProgramOptions::ProgramOptions::STOCHASTIC_SCHEDULE_NEIGHBORS:
		LOG() << "stochastic schedule neighbors" << endl;
		LOG() << "\tCut edges independently: " << cutEdgesIndependently << endl;
		LOG() << "\tTemperature parameter: " << po.cutParam << endl;
		LOG() << "\tNode clamping: " << po.nodeClamp << ", Edge clamping: " << po.edgeClamp << endl;
		LOG() << "\t\tNode clamp threshold: " << po.nodeClampThreshold << endl;
		LOG() << "\t\tEdge clamp positive threshold: " << po.edgeClampPositiveThreshold << endl;
		LOG() << "\t\tEdge clamp negative threshold: " << po.edgeClampNegativeThreshold << endl;
		successor = new HCSearch::StochasticScheduleNeighborSuccessor(cutEdgesIndependently, po.cutParam, 
			po.nodeClamp, po.edgeClamp, po.nodeClampThreshold, po.edgeClampPositiveThreshold, po.edgeClampNegativeThreshold);
		break;
	case MyProgramOptions::ProgramOptions::STOCHASTIC_SCHEDULE_CONFIDENCES_NEIGHBORS:
		LOG() << "stochastic schedule confidences neighbors" << endl;
		LOG() << "\tCut edges independently: " << cutEdgesIndependently << endl;
		LOG() << "\tTemperature parameter: " << po.cutParam << endl;
		LOG() << "\tNode clamping: " << po.nodeClamp << ", Edge clamping: " << po.edgeClamp << endl;
		LOG() << "\t\tNode clamp threshold: " << po.nodeClampThreshold << endl;
		LOG() << "\t\tEdge clamp positive threshold: " << po.edgeClampPositiveThreshold << endl;
		LOG() << "\t\tEdge clamp negative threshold: " << po.edgeClampNegativeThreshold << endl;
		successor = new HCSearch::StochasticScheduleConfidencesNeighborSuccessor(cutEdgesIndependently, po.cutParam, 
			po.nodeClamp, po.edgeClamp, po.nodeClampThreshold, po.edgeClampPositiveThreshold, po.edgeClampNegativeThreshold);
		break;
	case MyProgramOptions::ProgramOptions::STOCHASTIC_CONSTRAINED:
		LOG() << "stochastic constrained" << endl;
		LOG() << "\tCut edges independently: " << cutEdgesIndependently << endl;
		LOG() << "\tTemperature parameter: " << po.cutParam << endl;
		LOG() << "\tNode clamping: " << po.nodeClamp << ", Edge clamping: " << po.edgeClamp << endl;
		LOG() << "\t\tNode clamp threshold: " << po.nodeClampThreshold << endl;
		LOG() << "\t\tEdge clamp positive threshold: " << po.edgeClampPositiveThreshold << endl;
		LOG() << "\t\tEdge clamp negative threshold: " << po.edgeClampNegativeThreshold << endl;
		successor = new HCSearch::StochasticConstrainedSuccessor(cutEdgesIndependently, po.cutParam, 
			po.nodeClamp, po.edgeClamp, po.nodeClampThreshold, po.edgeClampPositiveThreshold, po.edgeClampNegativeThreshold);
		break;
	default:
		LOG(ERROR) << "undefined successor mode.";
	}

	// use IID logistic regression as initial state prediction function
	LOG() << "Initial state prediction function: ";
	LOG() << "IID logistic regression" << endl;
	HCSearch::IInitialPredictionFunction* initPredFunc = new HCSearch::MutexLogRegInit(); //new HCSearch::LogRegInit();

	// select some pruning function
	LOG() << "Pruning function: ";
	HCSearch::IPruneFunction* pruneFunc = NULL;
	switch (po.pruneMode)
	{
	case MyProgramOptions::ProgramOptions::NO_PRUNE:
		LOG() << "no prune" << endl;
		pruneFunc = new HCSearch::NoPrune();
		break;
	case MyProgramOptions::ProgramOptions::RANKER_PRUNE:
		LOG() << "ranker prune" << endl;
		pruneFunc = new HCSearch::RankerPrune(po.pruneRatio, pruneFeatFunc);
		break;
	case MyProgramOptions::ProgramOptions::ORACLE_PRUNE:
		LOG() << "oracle prune" << endl;
		pruneFunc = new HCSearch::OraclePrune(lossFunc, po.badPruneRatio);
		break;
	case MyProgramOptions::ProgramOptions::SIMULATED_RANKER_PRUNE:
		LOG() << "simulated ranker prune" << endl;
		pruneFunc = new HCSearch::SimulatedRankerPrune(po.pruneRatio, pruneFeatFunc);
		break;
	default:
		LOG(ERROR) << "undefined prune mode.";
	}

	LOG() << endl;

	// construct search space from these functions that we specified
	return new HCSearch::SearchSpace(heuristicFeatFunc, costFeatFunc, initPredFunc, successor, pruneFunc, lossFunc);
}

HCSearch::ISearchProcedure* setupSearchProcedure(MyProgramOptions::ProgramOptions po)
{
	LOG() << "=== Search Procedure ===" << endl;

	HCSearch::ISearchProcedure* searchProcedure = NULL;
	switch (po.searchProcedureMode)
	{
	case MyProgramOptions::ProgramOptions::GREEDY:
		LOG() << "Using greedy search." << endl;
		searchProcedure = new HCSearch::GreedySearchProcedure();
		break;
	case MyProgramOptions::ProgramOptions::BREADTH_BEAM:
		LOG() << "Using breadth-first beam search." << endl;
		LOG() << "Beam size=" << po.beamSize << endl;
		searchProcedure = new HCSearch::BreadthFirstBeamSearchProcedure(po.beamSize);
		break;
	case MyProgramOptions::ProgramOptions::BEST_BEAM:
		LOG() << "Using best-first beam search." << endl;
		LOG() << "Beam size=" << po.beamSize << endl;
		searchProcedure = new HCSearch::BestFirstBeamSearchProcedure(po.beamSize);
		break;
	default:
		LOG(ERROR) << "undefined search procedure mode.";
	}

	LOG() << endl;

	return searchProcedure;
}

void run(MyProgramOptions::ProgramOptions po)
{
	clock_t tic = clock();

	// time bound
	int timeBound = po.timeBound;

	// paths
	string heuristicModelPath = HCSearch::Global::settings->paths->OUTPUT_HEURISTIC_MODEL_FILE;
	string costModelPath = HCSearch::Global::settings->paths->OUTPUT_COST_H_MODEL_FILE;
	string costOracleHModelPath = HCSearch::Global::settings->paths->OUTPUT_COST_ORACLE_H_MODEL_FILE;
	string pruneModelPath = HCSearch::Global::settings->paths->OUTPUT_PRUNE_MODEL_FILE;
	string mutexPath = HCSearch::Global::settings->paths->OUTPUT_MUTEX_FILE;

	// params
	HCSearch::RankerType rankerType = po.rankLearnerType;

	// datasets
	vector< HCSearch::ImgFeatures* > XTrain;
	vector< HCSearch::ImgLabeling* > YTrain;
	vector< HCSearch::ImgFeatures* > XValidation;
	vector< HCSearch::ImgLabeling* > YValidation;
	vector< HCSearch::ImgFeatures* > XTest;
	vector< HCSearch::ImgLabeling* > YTest;

	// load dataset
	HCSearch::Dataset::loadDataset(XTrain, YTrain, XValidation, YValidation, XTest, YTest);

	// load search space functions and search space
	HCSearch::SearchSpace* searchSpace = setupSearchSpace(po);

	// load search procedure
	HCSearch::ISearchProcedure* searchProcedure = setupSearchProcedure(po);

	// run the appropriate mode
	for (vector< HCSearch::SearchType >::iterator it = po.schedule.begin();
		it != po.schedule.end(); ++it)
	{
		HCSearch::Global::settings->stats->resetSuccessorCount();

		HCSearch::SearchType mode = *it;

		if (mode != HCSearch::DISCOVER_PAIRWISE && po.pruneFeaturesMode == MyProgramOptions::ProgramOptions::STANDARD_PRUNE
			&& po.pruneMode == MyProgramOptions::ProgramOptions::RANKER_PRUNE)
		{
			HCSearch::IPruneFunction* pruneFunc = searchSpace->getPruneFunction();
			HCSearch::RankerPrune* pruneCast = dynamic_cast<HCSearch::RankerPrune*>(pruneFunc);
			HCSearch::IFeatureFunction* featFunc = pruneCast->getFeatureFunction();
			HCSearch::StandardPruneFeatures* featCast = dynamic_cast<HCSearch::StandardPruneFeatures*>(featFunc);
			HCSearch::IInitialPredictionFunction* initPredFunc = searchSpace->getInitialPredictionFunction();
			HCSearch::MutexLogRegInit* initPredFuncCast = dynamic_cast<HCSearch::MutexLogRegInit*>(initPredFunc);

			if (MyFileSystem::FileSystem::checkFileExists(mutexPath))
			{
				map<string, int> mutex = HCSearch::Model::loadPairwiseConstraints(mutexPath);
				featCast->setMutex(mutex);
				initPredFuncCast->setMutex(mutex);
			}
		}
		else if (mode != HCSearch::DISCOVER_PAIRWISE && po.pruneFeaturesMode == MyProgramOptions::ProgramOptions::STANDARD_PRUNE
			&& po.pruneMode == MyProgramOptions::ProgramOptions::SIMULATED_RANKER_PRUNE)
		{
			HCSearch::IPruneFunction* pruneFunc = searchSpace->getPruneFunction();
			HCSearch::SimulatedRankerPrune* pruneCast = dynamic_cast<HCSearch::SimulatedRankerPrune*>(pruneFunc);
			HCSearch::IFeatureFunction* featFunc = pruneCast->getFeatureFunction();
			HCSearch::StandardPruneFeatures* featCast = dynamic_cast<HCSearch::StandardPruneFeatures*>(featFunc);
			HCSearch::IInitialPredictionFunction* initPredFunc = searchSpace->getInitialPredictionFunction();
			HCSearch::MutexLogRegInit* initPredFuncCast = dynamic_cast<HCSearch::MutexLogRegInit*>(initPredFunc);

			if (MyFileSystem::FileSystem::checkFileExists(mutexPath))
			{
				map<string, int> mutex = HCSearch::Model::loadPairwiseConstraints(mutexPath);
				featCast->setMutex(mutex);
				initPredFuncCast->setMutex(mutex);
			}
		}

		if (po.pruneMode == MyProgramOptions::ProgramOptions::RANKER_PRUNE)
		{
			HCSearch::IPruneFunction* pruneFunc = searchSpace->getPruneFunction();
			HCSearch::RankerPrune* pruneCast = dynamic_cast<HCSearch::RankerPrune*>(pruneFunc);
			if (po.rankLearnerType == HCSearch::VW_RANK)
			{
				HCSearch::IRankModel* pruneModel;
				if (MyFileSystem::FileSystem::checkFileExists(pruneModelPath))
				{
					pruneModel = HCSearch::Model::loadModel(pruneModelPath, HCSearch::VW_RANK);
					LOG() << endl << "Loaded pruning model." << endl << endl;
				}
				else
				{
					pruneModel = new HCSearch::VWRankModel();
					LOG() << endl << "Creating empty pruning model." << endl << endl;
				}
				pruneCast->setRanker(pruneModel);
			}
			else
			{
				LOG(ERROR) << "only VW supported for prune ranker model" << endl;
				HCSearch::abort();
			}
		}

		switch (mode)
		{
		case HCSearch::LEARN_H:
		{
			LOG() << "=== Learning H ===" << endl;

			// learn heuristic, save heuristic model
			HCSearch::IRankModel* heuristicModel = HCSearch::Learning::learnH(XTrain, YTrain, XValidation, YValidation, 
				timeBound, searchSpace, searchProcedure, po.rankLearnerType, po.numTrainIterations);
			
			if (HCSearch::Global::settings->RANK == 0)
			{
				HCSearch::Model::saveModel(heuristicModel, heuristicModelPath, rankerType);
				if (po.saveFeaturesFiles && HCSearch::RankerTypeSaveable[po.rankLearnerType])
					MyFileSystem::FileSystem::copyFile(HCSearch::Global::settings->paths->OUTPUT_HEURISTIC_FEATURES_FILE, 
						HCSearch::Global::settings->paths->OUTPUT_ARCHIVED_HEURISTIC_FEATURES_FILE);
			}
			
			MyFileSystem::FileSystem::deleteFile(HCSearch::Global::settings->paths->OUTPUT_HEURISTIC_FEATURES_FILE);
			delete heuristicModel;

#ifdef USE_MPI
		MPI::Synchronize::masterWait("LEARNHSTART");
		MPI::Synchronize::slavesWait("LEARNHEND");
#endif

			break;
		}
		case HCSearch::LEARN_C:
		{
			LOG() << "=== Learning C with Learned H ===" << endl;

			// load heuristic, learn cost, save cost model
			HCSearch::IRankModel* heuristicModel = HCSearch::Model::loadModel(heuristicModelPath, rankerType);
			HCSearch::IRankModel* costModel = HCSearch::Learning::learnC(XTrain, YTrain, XValidation, YValidation, 
				heuristicModel, timeBound, searchSpace, searchProcedure, po.rankLearnerType, po.numTrainIterations);
			
			if (HCSearch::Global::settings->RANK == 0)
			{
				HCSearch::Model::saveModel(costModel, costModelPath, rankerType);
				if (po.saveFeaturesFiles && HCSearch::RankerTypeSaveable[po.rankLearnerType])
					MyFileSystem::FileSystem::copyFile(HCSearch::Global::settings->paths->OUTPUT_COST_H_FEATURES_FILE, 
						HCSearch::Global::settings->paths->OUTPUT_ARCHIVED_COST_H_FEATURES_FILE);
			}
			
			MyFileSystem::FileSystem::deleteFile(HCSearch::Global::settings->paths->OUTPUT_COST_H_FEATURES_FILE);
			delete heuristicModel;
			delete costModel;

#ifdef USE_MPI
		MPI::Synchronize::masterWait("LEARNCSTART");
		MPI::Synchronize::slavesWait("LEARNCEND");
#endif

			break;
		}
		case HCSearch::LEARN_C_ORACLE_H:
		{
			LOG() << "=== Learning C with Oracle H ===" << endl;

			// learn cost, save cost model
			HCSearch::IRankModel* costOracleHModel = HCSearch::Learning::learnCWithOracleH(XTrain, YTrain, XValidation, YValidation, 
				timeBound, searchSpace, searchProcedure, po.rankLearnerType, po.numTrainIterations);
			
			if (HCSearch::Global::settings->RANK == 0)
			{
				HCSearch::Model::saveModel(costOracleHModel, costOracleHModelPath, rankerType);
				if (po.saveFeaturesFiles && HCSearch::RankerTypeSaveable[po.rankLearnerType])
					MyFileSystem::FileSystem::copyFile(HCSearch::Global::settings->paths->OUTPUT_COST_ORACLE_H_FEATURES_FILE, 
						HCSearch::Global::settings->paths->OUTPUT_ARCHIVED_COST_ORACLE_H_FEATURES_FILE);
			}
			
			MyFileSystem::FileSystem::deleteFile(HCSearch::Global::settings->paths->OUTPUT_COST_ORACLE_H_FEATURES_FILE);
			delete costOracleHModel;

#ifdef USE_MPI
		MPI::Synchronize::masterWait("LEARNCOHSTART");
		MPI::Synchronize::slavesWait("LEARNCOHEND");
#endif

			break;
		}
		case HCSearch::LEARN_PRUNE:
		{
			LOG() << "=== Learning P ===" << endl;

			// learn cost, save cost model
			if (po.pruneMode == MyProgramOptions::ProgramOptions::RANKER_PRUNE)
			{
				HCSearch::IRankModel* pruneModel = HCSearch::Learning::learnP(XTrain, YTrain, XValidation, YValidation, 
					timeBound, searchSpace, searchProcedure, HCSearch::VW_RANK, po.numTrainIterations);
				
				// set the prune function
				HCSearch::IPruneFunction* pruneFunc = searchSpace->getPruneFunction();
				HCSearch::RankerPrune* pruneCast = dynamic_cast<HCSearch::RankerPrune*>(pruneFunc);
				pruneCast->setRanker(pruneModel);

				if (HCSearch::Global::settings->RANK == 0)
				{
					pruneModel->save(pruneModelPath);
					if (po.saveFeaturesFiles)
						MyFileSystem::FileSystem::copyFile(HCSearch::Global::settings->paths->OUTPUT_PRUNE_FEATURES_FILE, 
							HCSearch::Global::settings->paths->OUTPUT_ARCHIVED_PRUNE_FEATURES_FILE);
				}
				
				MyFileSystem::FileSystem::deleteFile(HCSearch::Global::settings->paths->OUTPUT_PRUNE_FEATURES_FILE);
				delete pruneModel;
			}
			else
			{
				LOG(ERROR) << "unsupported pruning" << endl;
			}

#ifdef USE_MPI
		MPI::Synchronize::masterWait("LEARNPSTART");
		MPI::Synchronize::slavesWait("LEARNPEND");
#endif

			break;
		}
		case HCSearch::DISCOVER_PAIRWISE:
		{
			LOG() << "=== Discovering Mutex ===" << endl;

			// discover mutex constraints
			map<string, int> pairwiseConstraints = HCSearch::Learning::discoverPairwiseClassConstraints(XTrain, YTrain);
			
			if (HCSearch::Global::settings->RANK == 0)
			{
				HCSearch::Model::savePairwiseConstraints(pairwiseConstraints, mutexPath);
			}

#ifdef USE_MPI
		MPI::Synchronize::masterWait("DISCOVERMUTEXSTART");
		MPI::Synchronize::slavesWait("DISCOVERMUTEXEND");
#endif

			break;
		}
		case HCSearch::LL:
		{
			LOG() << "=== Inference LL ===" << endl;

			// run LL search on test examples
			int start, end;
			HCSearch::Dataset::computeTaskRange(HCSearch::Global::settings->RANK, XTest.size(), 
				HCSearch::Global::settings->NUM_PROCESSES, start, end);
			for (int i = start; i < end; i++)
			{
				for (int iter = 0; iter < po.numTestIterations; iter++)
				{
					if (po.numTestIterations == 1)
						iter = po.uniqueIterId;

					LOG() << endl << "LL Search: (iter " << iter << ") beginning search on " << XTest[i]->getFileName() << " (example " << i << ")..." << endl;

					// setup meta
					HCSearch::ISearchProcedure::SearchMetadata meta;
					meta.saveAnytimePredictions = po.saveAnytimePredictions;
					meta.setType = HCSearch::TEST;
					meta.exampleName = XTest[i]->getFileName();
					meta.iter = iter;

					// inference
					HCSearch::ImgLabeling YPred = HCSearch::Inference::runLLSearch(XTest[i], YTest[i], 
						timeBound, searchSpace, searchProcedure, meta);
				
					// save the prediction
					stringstream ssPredictNodes;
					ssPredictNodes << HCSearch::Global::settings->paths->OUTPUT_RESULTS_DIR << "final" 
						<< "_nodes_" << HCSearch::SearchTypeStrings[HCSearch::LL] 
						<< "_" << HCSearch::DatasetTypeStrings[meta.setType] 
						<< "_time" << timeBound 
							<< "_fold" << meta.iter 
							<< "_" << meta.exampleName << ".txt";
					HCSearch::SavePrediction::saveLabels(YPred, ssPredictNodes.str());

					// save the prediction mask
					if (po.saveOutputMask)
					{
						stringstream ssPredictSegments;
						ssPredictSegments << HCSearch::Global::settings->paths->OUTPUT_RESULTS_DIR << "final"
							<< "_" << HCSearch::SearchTypeStrings[HCSearch::LL] 
							<< "_" << HCSearch::DatasetTypeStrings[meta.setType] 
							<< "_time" << timeBound 
								<< "_fold" << meta.iter 
								<< "_" << meta.exampleName << ".txt";
						HCSearch::SavePrediction::saveLabelMask(*XTest[i], YPred, ssPredictSegments.str());
					}

					if (po.numTestIterations == 1)
						break;
				}
			}

#ifdef USE_MPI
		MPI::Synchronize::masterWait("INFERLLSTART");
		MPI::Synchronize::slavesWait("INFERLLEND");
#endif

			break;
		}
		case HCSearch::HL:
		{
			LOG() << "=== Inference HL ===" << endl;

			// load heuristic, run HL search on test examples
			HCSearch::IRankModel* heuristicModel = HCSearch::Model::loadModel(heuristicModelPath, rankerType);

			int start, end;
			HCSearch::Dataset::computeTaskRange(HCSearch::Global::settings->RANK, XTest.size(), 
				HCSearch::Global::settings->NUM_PROCESSES, start, end);
			for (int i = start; i < end; i++)
			{
				for (int iter = 0; iter < po.numTestIterations; iter++)
				{
					if (po.numTestIterations == 1)
						iter = po.uniqueIterId;

					LOG() << endl << "HL Search: (iter " << iter << ") beginning search on " << XTest[i]->getFileName() << " (example " << i << ")..." << endl;

					// setup meta
					HCSearch::ISearchProcedure::SearchMetadata meta;
					meta.saveAnytimePredictions = po.saveAnytimePredictions;
					meta.setType = HCSearch::TEST;
					meta.exampleName = XTest[i]->getFileName();
					meta.iter = iter;

					// inference
					HCSearch::ImgLabeling YPred = HCSearch::Inference::runHLSearch(XTest[i], YTest[i], 
						timeBound, searchSpace, searchProcedure, heuristicModel, meta);
				
					// save the prediction
					stringstream ssPredictNodes;
					ssPredictNodes << HCSearch::Global::settings->paths->OUTPUT_RESULTS_DIR << "final" 
						<< "_nodes_" << HCSearch::SearchTypeStrings[HCSearch::HL] 
						<< "_" << HCSearch::DatasetTypeStrings[meta.setType] 
						<< "_time" << timeBound 
							<< "_fold" << meta.iter 
							<< "_" << meta.exampleName << ".txt";
					HCSearch::SavePrediction::saveLabels(YPred, ssPredictNodes.str());

					// save the prediction mask
					if (po.saveOutputMask)
					{
						stringstream ssPredictSegments;
						ssPredictSegments << HCSearch::Global::settings->paths->OUTPUT_RESULTS_DIR << "final"
							<< "_" << HCSearch::SearchTypeStrings[HCSearch::HL] 
							<< "_" << HCSearch::DatasetTypeStrings[meta.setType] 
							<< "_time" << timeBound 
								<< "_fold" << meta.iter 
								<< "_" << meta.exampleName << ".txt";
						HCSearch::SavePrediction::saveLabelMask(*XTest[i], YPred, ssPredictSegments.str());
					}

					if (po.numTestIterations == 1)
						break;
				}
			}

			delete heuristicModel;
			
#ifdef USE_MPI
		MPI::Synchronize::masterWait("INFERHLSTART");
		MPI::Synchronize::slavesWait("INFERHLEND");
#endif
			
			break;
		}
		case HCSearch::LC:
		{
			LOG() << "=== Inference LC ===" << endl;

			// load cost oracle H, run LC search on test examples
			HCSearch::IRankModel* costModel = HCSearch::Model::loadModel(costOracleHModelPath, rankerType);

			int start, end;
			HCSearch::Dataset::computeTaskRange(HCSearch::Global::settings->RANK, XTest.size(), 
				HCSearch::Global::settings->NUM_PROCESSES, start, end);
			for (int i = start; i < end; i++)
			{
				for (int iter = 0; iter < po.numTestIterations; iter++)
				{
					if (po.numTestIterations == 1)
						iter = po.uniqueIterId;

					LOG() << endl << "LC Search: (iter " << iter << ") beginning search on " << XTest[i]->getFileName() << " (example " << i << ")..." << endl;

					// setup meta
					HCSearch::ISearchProcedure::SearchMetadata meta;
					meta.saveAnytimePredictions = po.saveAnytimePredictions;
					meta.setType = HCSearch::TEST;
					meta.exampleName = XTest[i]->getFileName();
					meta.iter = iter;

					// inference
					HCSearch::ImgLabeling YPred = HCSearch::Inference::runLCSearch(XTest[i], YTest[i], 
						timeBound, searchSpace, searchProcedure, costModel, meta);
				
					// save the prediction
					stringstream ssPredictNodes;
					ssPredictNodes << HCSearch::Global::settings->paths->OUTPUT_RESULTS_DIR << "final" 
						<< "_nodes_" << HCSearch::SearchTypeStrings[HCSearch::LC] 
						<< "_" << HCSearch::DatasetTypeStrings[meta.setType] 
						<< "_time" << timeBound 
							<< "_fold" << meta.iter 
							<< "_" << meta.exampleName << ".txt";
					HCSearch::SavePrediction::saveLabels(YPred, ssPredictNodes.str());

					// save the prediction mask
					if (po.saveOutputMask)
					{
						stringstream ssPredictSegments;
						ssPredictSegments << HCSearch::Global::settings->paths->OUTPUT_RESULTS_DIR << "final"
							<< "_" << HCSearch::SearchTypeStrings[HCSearch::LC] 
							<< "_" << HCSearch::DatasetTypeStrings[meta.setType] 
							<< "_time" << timeBound 
								<< "_fold" << meta.iter 
								<< "_" << meta.exampleName << ".txt";
						HCSearch::SavePrediction::saveLabelMask(*XTest[i], YPred, ssPredictSegments.str());
					}

					if (po.numTestIterations == 1)
						break;
				}
			}

			delete costModel;

#ifdef USE_MPI
		MPI::Synchronize::masterWait("INFERLCSTART");
		MPI::Synchronize::slavesWait("INFERLCEND");
#endif

			break;
		}
		case HCSearch::HC:
		{
			LOG() << "=== Inference HC ===" << endl;

			// load heuristic and cost, run HC search on test examples
			HCSearch::IRankModel* heuristicModel = HCSearch::Model::loadModel(heuristicModelPath, rankerType);
			HCSearch::IRankModel* costModel = HCSearch::Model::loadModel(costModelPath, rankerType);

			int start, end;
			HCSearch::Dataset::computeTaskRange(HCSearch::Global::settings->RANK, XTest.size(), 
				HCSearch::Global::settings->NUM_PROCESSES, start, end);
			for (int i = start; i < end; i++)
			{
				for (int iter = 0; iter < po.numTestIterations; iter++)
				{
					if (po.numTestIterations == 1)
						iter = po.uniqueIterId;

					LOG() << endl << "HC Search: (iter " << iter << ") beginning search on " << XTest[i]->getFileName() << " (example " << i << ")..." << endl;

					// setup meta
					HCSearch::ISearchProcedure::SearchMetadata meta;
					meta.saveAnytimePredictions = po.saveAnytimePredictions;
					meta.setType = HCSearch::TEST;
					meta.exampleName = XTest[i]->getFileName();
					meta.iter = iter;

					// inference
					HCSearch::ImgLabeling YPred = HCSearch::Inference::runHCSearch(XTest[i], YTest[i], 
						timeBound, searchSpace, searchProcedure, heuristicModel, costModel, meta);

					// save the prediction
					stringstream ssPredictNodes;
					ssPredictNodes << HCSearch::Global::settings->paths->OUTPUT_RESULTS_DIR << "final" 
						<< "_nodes_" << HCSearch::SearchTypeStrings[HCSearch::HC] 
						<< "_" << HCSearch::DatasetTypeStrings[meta.setType] 
						<< "_time" << timeBound 
							<< "_fold" << meta.iter 
							<< "_" << meta.exampleName << ".txt";
					HCSearch::SavePrediction::saveLabels(YPred, ssPredictNodes.str());

					// save the prediction mask
					if (po.saveOutputMask)
					{
						stringstream ssPredictSegments;
						ssPredictSegments << HCSearch::Global::settings->paths->OUTPUT_RESULTS_DIR << "final"
							<< "_" << HCSearch::SearchTypeStrings[HCSearch::HC] 
							<< "_" << HCSearch::DatasetTypeStrings[meta.setType] 
							<< "_time" << timeBound 
								<< "_fold" << meta.iter 
								<< "_" << meta.exampleName << ".txt";
						HCSearch::SavePrediction::saveLabelMask(*XTest[i], YPred, ssPredictSegments.str());
					}

					if (po.numTestIterations == 1)
						break;
				}
			}

			delete heuristicModel;
			delete costModel;

#ifdef USE_MPI
		MPI::Synchronize::masterWait("INFERHCSTART");
		MPI::Synchronize::slavesWait("INFERHCEND");
#endif

			break;
		}
		default:
			LOG(ERROR) << "invalid mode!";
		}

		LOG() << "Average number of successor candidates=" << HCSearch::Global::settings->stats->getSuccessorAverage() << endl;
	}

	// clean up
	delete searchSpace;
	delete searchProcedure;
	HCSearch::Dataset::unloadDataset(XTrain, YTrain, XValidation, YValidation, XTest, YTest);

	clock_t toc = clock();
	LOG() << "total run time: " << (double)(toc - tic)/CLOCKS_PER_SEC << endl << endl;
}

void printInfo(MyProgramOptions::ProgramOptions po)
{
	LOG() << "=== Program Schedule ===" << endl;

	int cnt = 1;
	for (vector< HCSearch::SearchType >::iterator it = po.schedule.begin();
		it != po.schedule.end(); ++it)
	{
		HCSearch::SearchType mode = *it;
		LOG() << cnt << ". " << HCSearch::SearchTypeStrings[mode] << endl;
		cnt++;
	}

	LOG() << endl;

	if (!po.demoMode)
	{
		LOG() << "=== Program Options ===" << endl;
		LOG() << "Rank learner: " << HCSearch::RankerTypeStrings[po.rankLearnerType] << endl;
		LOG() << "Num training iterations: " << po.numTrainIterations << endl;
		LOG() << "Num testing iterations: " << po.numTestIterations << endl;
		if (po.numTestIterations == 1)
		{
			LOG() << "\tUnique iteration ID: " << po.uniqueIterId << endl;
		}
		LOG() << "Save anytime predictions: " << po.saveAnytimePredictions << endl;
		LOG() << "Save features files: " << po.saveFeaturesFiles << endl;
		LOG() << "Save output label masks: " << po.saveOutputMask << endl;
	}
	else
	{
		LOG() << "=== USING DEMO MODE ===" << endl;
	}

	LOG() << endl;

	if (po.verboseMode)
	{
		LOG() << "=== Paths ===" << endl;

		LOG() << "EXTERNAL_DIR: " << HCSearch::Global::settings->paths->EXTERNAL_DIR << endl;
		LOG() << "INPUT_DIR: " << HCSearch::Global::settings->paths->INPUT_DIR  << endl;
		LOG() << "OUTPUT_DIR: " << HCSearch::Global::settings->paths->OUTPUT_DIR  << endl;
		LOG() << "LIBLINEAR_DIR: " << HCSearch::Global::settings->paths->LIBLINEAR_DIR  << endl;
		LOG() << "LIBSVM_DIR: " << HCSearch::Global::settings->paths->LIBSVM_DIR  << endl;
		LOG() << "SVMRANK_DIR: " << HCSearch::Global::settings->paths->SVMRANK_DIR  << endl;
		LOG() << "VW_DIR: " << HCSearch::Global::settings->paths->VOWPALWABBIT_DIR  << endl;

		LOG() << endl;

		LOG() << "INPUT_NODES_DIR: " << HCSearch::Global::settings->paths->INPUT_NODES_DIR << endl;
		LOG() << "INPUT_EDGES_DIR: " << HCSearch::Global::settings->paths->INPUT_EDGES_DIR  << endl;
		LOG() << "INPUT_META_DIR: " << HCSearch::Global::settings->paths->INPUT_META_DIR  << endl;
		LOG() << "INPUT_SEGMENTS_DIR: " << HCSearch::Global::settings->paths->INPUT_SEGMENTS_DIR  << endl;
		LOG() << "INPUT_SPLITS_DIR: " << HCSearch::Global::settings->paths->INPUT_SPLITS_DIR  << endl;
		LOG() << "INPUT_SPLITS_FOLDER_NAME: " << HCSearch::Global::settings->paths->INPUT_SPLITS_FOLDER_NAME  << endl;

		LOG() << "INPUT_SPLITS_TRAIN_FILE: " << HCSearch::Global::settings->paths->INPUT_SPLITS_TRAIN_FILE << endl;
		LOG() << "INPUT_SPLITS_VALIDATION_FILE: " << HCSearch::Global::settings->paths->INPUT_SPLITS_VALIDATION_FILE  << endl;
		LOG() << "INPUT_SPLITS_TEST_FILE: " << HCSearch::Global::settings->paths->INPUT_SPLITS_TEST_FILE  << endl;

		LOG() << "INPUT_METADATA_FILE: " << HCSearch::Global::settings->paths->INPUT_METADATA_FILE  << endl;
		LOG() << "INPUT_CODEBOOK_FILE: " << HCSearch::Global::settings->paths->INPUT_CODEBOOK_FILE  << endl;
		LOG() << "INPUT_SPLITS_FOLDER_NAME: " << HCSearch::Global::settings->paths->INPUT_INITFUNC_TRAINING_FILE  << endl;

		LOG() << endl;

		LOG() << "OUTPUT_LOGS_DIR: " << HCSearch::Global::settings->paths->OUTPUT_LOGS_DIR << endl;
		LOG() << "OUTPUT_MODELS_DIR: " << HCSearch::Global::settings->paths->OUTPUT_MODELS_DIR  << endl;
		LOG() << "OUTPUT_RESULTS_DIR: " << HCSearch::Global::settings->paths->OUTPUT_RESULTS_DIR  << endl;
		LOG() << "OUTPUT_TEMP_DIR: " << HCSearch::Global::settings->paths->OUTPUT_TEMP_DIR  << endl;

		LOG() << "OUTPUT_HEURISTIC_FEATURES_FILE: " << HCSearch::Global::settings->paths->OUTPUT_HEURISTIC_FEATURES_FILE  << endl;
		LOG() << "OUTPUT_COST_H_FEATURES_FILE: " << HCSearch::Global::settings->paths->OUTPUT_COST_H_FEATURES_FILE  << endl;
		LOG() << "OUTPUT_COST_ORACLE_H_FEATURES_FILE: " << HCSearch::Global::settings->paths->OUTPUT_COST_ORACLE_H_FEATURES_FILE << endl;

		LOG() << "OUTPUT_ARCHIVED_HEURISTIC_FEATURES_FILE: " << HCSearch::Global::settings->paths->OUTPUT_ARCHIVED_HEURISTIC_FEATURES_FILE  << endl;
		LOG() << "OUTPUT_ARCHIVED_COST_H_FEATURES_FILE: " << HCSearch::Global::settings->paths->OUTPUT_ARCHIVED_COST_H_FEATURES_FILE  << endl;
		LOG() << "OUTPUT_ARCHIVED_COST_ORACLE_H_FEATURES_FILE: " << HCSearch::Global::settings->paths->OUTPUT_ARCHIVED_COST_ORACLE_H_FEATURES_FILE  << endl;

		LOG() << "OUTPUT_HEURISTIC_MODEL_FILE: " << HCSearch::Global::settings->paths->OUTPUT_HEURISTIC_MODEL_FILE  << endl;
		LOG() << "OUTPUT_COST_H_MODEL_FILE: " << HCSearch::Global::settings->paths->OUTPUT_COST_H_MODEL_FILE  << endl;
		LOG() << "OUTPUT_COST_ORACLE_H_MODEL_FILE: " << HCSearch::Global::settings->paths->OUTPUT_COST_ORACLE_H_MODEL_FILE  << endl;

		LOG() << "OUTPUT_LOG_FILE: " << HCSearch::Global::settings->paths->OUTPUT_LOG_FILE  << endl;

		LOG() << "OUTPUT_INITFUNC_MODEL_FILE: " << HCSearch::Global::settings->paths->OUTPUT_INITFUNC_MODEL_FILE  << endl;
		LOG() << "OUTPUT_INITFUNC_FEATURES_FILE: " << HCSearch::Global::settings->paths->OUTPUT_INITFUNC_FEATURES_FILE << endl;
		LOG() << "OUTPUT_INITFUNC_PREDICT_FILE: " << HCSearch::Global::settings->paths->OUTPUT_INITFUNC_PREDICT_FILE  << endl;

		LOG() << endl;
	}
}