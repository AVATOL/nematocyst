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
	HCSearch::Global::settings->paths->INPUT_SPLITS_FOLDER_NAME = po.splitsFolderName;
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

	// use Hamming loss function
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
		heuristicFeatFunc = new HCSearch::StandardFeatures();
		break;
	case MyProgramOptions::ProgramOptions::STANDARD_DENSE:
		LOG() << "dense standard features" << endl;
		heuristicFeatFunc = new HCSearch::DenseStandardFeatures();
		break;
	case MyProgramOptions::ProgramOptions::DENSE_CRF:
		LOG() << "dense CRF features" << endl;
		heuristicFeatFunc = new HCSearch::DenseCRFFeatures();
		break;
	case MyProgramOptions::ProgramOptions::UNARY_ONLY:
		LOG() << "unary features" << endl;
		heuristicFeatFunc = new HCSearch::UnaryConfidencesFeatures();
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
		costFeatFunc = new HCSearch::StandardFeatures();
		break;
	case MyProgramOptions::ProgramOptions::STANDARD_DENSE:
		LOG() << "dense standard features" << endl;
		costFeatFunc = new HCSearch::DenseStandardFeatures();
		break;
	case MyProgramOptions::ProgramOptions::DENSE_CRF:
		LOG() << "dense CRF features" << endl;
		costFeatFunc = new HCSearch::DenseCRFFeatures();
		break;
	case MyProgramOptions::ProgramOptions::UNARY_ONLY:
		LOG() << "unary features" << endl;
		costFeatFunc = new HCSearch::UnaryConfidencesFeatures();
		break;
	default:
		LOG(ERROR) << "undefined feature mode.";
	}

	// use stochastic successor function
	LOG() << "Successor function: ";
	HCSearch::ISuccessorFunction* successor = NULL;
	bool cutEdgesIndependently = po.stochasticCutMode == MyProgramOptions::ProgramOptions::EDGES;
	switch (po.successorsMode)
	{
	case MyProgramOptions::ProgramOptions::FLIPBIT:
		LOG() << "flipbit" << endl;
		LOG() << "\tMax num candidates: " << po.boundSuccessorCandidates << endl;
		successor = new HCSearch::FlipbitSuccessor(po.boundSuccessorCandidates);
		break;
	case MyProgramOptions::ProgramOptions::FLIPBIT_NEIGHBORS:
		LOG() << "flipbit neighbors" << endl;
		LOG() << "\tMax num candidates: " << po.boundSuccessorCandidates << endl;
		successor = new HCSearch::FlipbitNeighborSuccessor(po.boundSuccessorCandidates);
		break;
	case MyProgramOptions::ProgramOptions::STOCHASTIC:
		LOG() << "stochastic" << endl;
		LOG() << "\tCut edges independently: " << cutEdgesIndependently << endl;
		LOG() << "\tTemperature parameter: " << po.cutParam << endl;
		LOG() << "\tMax num candidates: " << po.boundSuccessorCandidates << endl;
		successor = new HCSearch::StochasticSuccessor(cutEdgesIndependently, po.cutParam, po.boundSuccessorCandidates);
		break;
	case MyProgramOptions::ProgramOptions::STOCHASTIC_NEIGHBORS:
		LOG() << "stochastic neighbors" << endl;
		LOG() << "\tCut edges independently: " << cutEdgesIndependently << endl;
		LOG() << "\tTemperature parameter: " << po.cutParam << endl;
		LOG() << "\tMax num candidates: " << po.boundSuccessorCandidates << endl;
		successor = new HCSearch::StochasticNeighborSuccessor(cutEdgesIndependently, po.cutParam, po.boundSuccessorCandidates);
		break;
	case MyProgramOptions::ProgramOptions::STOCHASTIC_CONFIDENCES_NEIGHBORS:
		LOG() << "stochastic confidences neighbors" << endl;
		LOG() << "\tCut edges independently: " << cutEdgesIndependently << endl;
		LOG() << "\tTemperature parameter: " << po.cutParam << endl;
		LOG() << "\tMax num candidates: " << po.boundSuccessorCandidates << endl;
		successor = new HCSearch::StochasticConfidencesNeighborSuccessor(cutEdgesIndependently, po.cutParam, po.boundSuccessorCandidates);
		break;
	case MyProgramOptions::ProgramOptions::CUT_SCHEDULE:
		LOG() << "cut schedule" << endl;
		LOG() << "\tTemperature parameter: " << po.cutParam << endl;
		LOG() << "\tMax num candidates: " << po.boundSuccessorCandidates << endl;
		successor = new HCSearch::CutScheduleSuccessor(po.cutParam, po.boundSuccessorCandidates);
		break;
	case MyProgramOptions::ProgramOptions::CUT_SCHEDULE_NEIGHBORS:
		LOG() << "cut schedule neighbors" << endl;
		LOG() << "\tTemperature parameter: " << po.cutParam << endl;
		LOG() << "\tMax num candidates: " << po.boundSuccessorCandidates << endl;
		successor = new HCSearch::CutScheduleNeighborSuccessor(po.cutParam, po.boundSuccessorCandidates);
		break;
	case MyProgramOptions::ProgramOptions::CUT_SCHEDULE_CONFIDENCES_NEIGHBORS:
		LOG() << "cut schedule confidences neighbors" << endl;
		LOG() << "\tTemperature parameter: " << po.cutParam << endl;
		LOG() << "\tMax num candidates: " << po.boundSuccessorCandidates << endl;
		successor = new HCSearch::CutScheduleConfidencesNeighborSuccessor(po.cutParam, po.boundSuccessorCandidates);
		break;
	default:
		LOG(ERROR) << "undefined successor mode.";
	}

	// use IID logistic regression as initial state prediction function
	LOG() << "Initial state prediction function: ";
	LOG() << "IID logistic regression" << endl;
	HCSearch::IInitialPredictionFunction* initPredFunc = new HCSearch::LogRegInit();

	LOG() << endl;

	// construct search space from these functions that we specified
	return new HCSearch::SearchSpace(heuristicFeatFunc, costFeatFunc, initPredFunc, successor, lossFunc);
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
	string costRandomHModelPath = HCSearch::Global::settings->paths->OUTPUT_COST_RANDOM_H_MODEL_FILE;

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
		case HCSearch::LEARN_C_RANDOM_H:
		{
			LOG() << "=== Learning C with Random H ===" << endl;

			// learn cost, save cost model
			HCSearch::IRankModel* costRandomHModel = HCSearch::Learning::learnCWithRandomH(XTrain, YTrain, XValidation, YValidation, 
				timeBound, searchSpace, searchProcedure, po.rankLearnerType, po.numTrainIterations);
			
			if (HCSearch::Global::settings->RANK == 0)
			{
				HCSearch::Model::saveModel(costRandomHModel, costRandomHModelPath, rankerType);
				if (po.saveFeaturesFiles && HCSearch::RankerTypeSaveable[po.rankLearnerType])
					MyFileSystem::FileSystem::copyFile(HCSearch::Global::settings->paths->OUTPUT_COST_RANDOM_H_FEATURES_FILE, 
						HCSearch::Global::settings->paths->OUTPUT_ARCHIVED_COST_RANDOM_H_FEATURES_FILE);
			}
			
			MyFileSystem::FileSystem::deleteFile(HCSearch::Global::settings->paths->OUTPUT_COST_RANDOM_H_FEATURES_FILE);
			delete costRandomHModel;

#ifdef USE_MPI
		MPI::Synchronize::masterWait("LEARNCRHSTART");
		MPI::Synchronize::slavesWait("LEARNCRHEND");
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
					HCSearch::ImgLabeling YPred = HCSearch::Inference::runHCSearch(XTest[i], 
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
		case HCSearch::RL:
		{
			LOG() << "=== Inference RL ===" << endl;

			// run RL search on test examples
			int start, end;
			HCSearch::Dataset::computeTaskRange(HCSearch::Global::settings->RANK, XTest.size(), 
				HCSearch::Global::settings->NUM_PROCESSES, start, end);
			for (int i = start; i < end; i++)
			{
				for (int iter = 0; iter < po.numTestIterations; iter++)
				{
					if (po.numTestIterations == 1)
						iter = po.uniqueIterId;

					LOG() << endl << "RL Search: (iter " << iter << ") beginning search on " << XTest[i]->getFileName() << " (example " << i << ")..." << endl;

					// setup meta
					HCSearch::ISearchProcedure::SearchMetadata meta;
					meta.saveAnytimePredictions = po.saveAnytimePredictions;
					meta.setType = HCSearch::TEST;
					meta.exampleName = XTest[i]->getFileName();
					meta.iter = iter;

					// inference
					HCSearch::ImgLabeling YPred = HCSearch::Inference::runRLSearch(XTest[i], YTest[i], 
						timeBound, searchSpace, searchProcedure, meta);
				
					// save the prediction
					stringstream ssPredictNodes;
					ssPredictNodes << HCSearch::Global::settings->paths->OUTPUT_RESULTS_DIR << "final" 
						<< "_nodes_" << HCSearch::SearchTypeStrings[HCSearch::RL] 
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
							<< "_" << HCSearch::SearchTypeStrings[HCSearch::RL] 
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
		MPI::Synchronize::masterWait("INFERRLSTART");
		MPI::Synchronize::slavesWait("INFERRLEND");
#endif

			break;
		}
		case HCSearch::RC:
		{
			LOG() << "=== Inference RC ===" << endl;

			// load cost random H, run RC search on test examples
			HCSearch::IRankModel* costModel = HCSearch::Model::loadModel(costRandomHModelPath, rankerType);

			int start, end;
			HCSearch::Dataset::computeTaskRange(HCSearch::Global::settings->RANK, XTest.size(), 
				HCSearch::Global::settings->NUM_PROCESSES, start, end);
			for (int i = start; i < end; i++)
			{
				for (int iter = 0; iter < po.numTestIterations; iter++)
				{
					if (po.numTestIterations == 1)
						iter = po.uniqueIterId;

					LOG() << endl << "RC Search: (iter " << iter << ") beginning search on " << XTest[i]->getFileName() << " (example " << i << ")..." << endl;

					// setup meta
					HCSearch::ISearchProcedure::SearchMetadata meta;
					meta.saveAnytimePredictions = po.saveAnytimePredictions;
					meta.setType = HCSearch::TEST;
					meta.exampleName = XTest[i]->getFileName();
					meta.iter = iter;

					// inference
					HCSearch::ImgLabeling YPred = HCSearch::Inference::runRCSearch(XTest[i], 
						timeBound, searchSpace, searchProcedure, costModel, meta);
				
					// save the prediction
					stringstream ssPredictNodes;
					ssPredictNodes << HCSearch::Global::settings->paths->OUTPUT_RESULTS_DIR << "final" 
						<< "_nodes_" << HCSearch::SearchTypeStrings[HCSearch::RC] 
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
							<< "_" << HCSearch::SearchTypeStrings[HCSearch::RC] 
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
		MPI::Synchronize::masterWait("INFERRCSTART");
		MPI::Synchronize::slavesWait("INFERRCEND");
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
		LOG() << "OUTPUT_COST_RANDOM_H_FEATURES_FILE: " << HCSearch::Global::settings->paths->OUTPUT_COST_RANDOM_H_FEATURES_FILE << endl;

		LOG() << "OUTPUT_ARCHIVED_HEURISTIC_FEATURES_FILE: " << HCSearch::Global::settings->paths->OUTPUT_ARCHIVED_HEURISTIC_FEATURES_FILE  << endl;
		LOG() << "OUTPUT_ARCHIVED_COST_H_FEATURES_FILE: " << HCSearch::Global::settings->paths->OUTPUT_ARCHIVED_COST_H_FEATURES_FILE  << endl;
		LOG() << "OUTPUT_ARCHIVED_COST_ORACLE_H_FEATURES_FILE: " << HCSearch::Global::settings->paths->OUTPUT_ARCHIVED_COST_ORACLE_H_FEATURES_FILE  << endl;
		LOG() << "OUTPUT_ARCHIVED_COST_RANDOM_H_FEATURES_FILE: " << HCSearch::Global::settings->paths->OUTPUT_ARCHIVED_COST_RANDOM_H_FEATURES_FILE  << endl;

		LOG() << "OUTPUT_HEURISTIC_ONLINE_WEIGHTS_FILE: " << HCSearch::Global::settings->paths->OUTPUT_HEURISTIC_ONLINE_WEIGHTS_FILE  << endl;
		LOG() << "OUTPUT_COST_H_ONLINE_WEIGHTS_FILE: " << HCSearch::Global::settings->paths->OUTPUT_COST_H_ONLINE_WEIGHTS_FILE  << endl;
		LOG() << "OUTPUT_COST_ORACLE_H_ONLINE_WEIGHTS_FILE: " << HCSearch::Global::settings->paths->OUTPUT_COST_ORACLE_H_ONLINE_WEIGHTS_FILE << endl;
		LOG() << "OUTPUT_COST_RANDOM_H_ONLINE_WEIGHTS_FILE: " << HCSearch::Global::settings->paths->OUTPUT_COST_RANDOM_H_ONLINE_WEIGHTS_FILE << endl;

		LOG() << "OUTPUT_HEURISTIC_MODEL_FILE: " << HCSearch::Global::settings->paths->OUTPUT_HEURISTIC_MODEL_FILE  << endl;
		LOG() << "OUTPUT_COST_H_MODEL_FILE: " << HCSearch::Global::settings->paths->OUTPUT_COST_H_MODEL_FILE  << endl;
		LOG() << "OUTPUT_COST_ORACLE_H_MODEL_FILE: " << HCSearch::Global::settings->paths->OUTPUT_COST_ORACLE_H_MODEL_FILE  << endl;
		LOG() << "OUTPUT_COST_RANDOM_H_MODEL_FILE: " << HCSearch::Global::settings->paths->OUTPUT_COST_RANDOM_H_MODEL_FILE  << endl;

		LOG() << "OUTPUT_LOG_FILE: " << HCSearch::Global::settings->paths->OUTPUT_LOG_FILE  << endl;

		LOG() << "OUTPUT_INITFUNC_MODEL_FILE: " << HCSearch::Global::settings->paths->OUTPUT_INITFUNC_MODEL_FILE  << endl;
		LOG() << "OUTPUT_INITFUNC_FEATURES_FILE: " << HCSearch::Global::settings->paths->OUTPUT_INITFUNC_FEATURES_FILE << endl;
		LOG() << "OUTPUT_INITFUNC_PREDICT_FILE: " << HCSearch::Global::settings->paths->OUTPUT_INITFUNC_PREDICT_FILE  << endl;

		LOG() << endl;
	}
}