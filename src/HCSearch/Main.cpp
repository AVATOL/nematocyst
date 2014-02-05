#include <iostream>
#include "../HCSearchLib/HCSearch.hpp"
#include "MyFileSystem.hpp"
#include "MyProgramOptions.hpp"
#include "Main.hpp"
#include "Demo.hpp"

using namespace std;

int main(int argc, char* argv[])
{
	// initialize HCSearch
	HCSearch::Setup::initialize(argc, argv);

#ifdef USE_MPI
	//TODO
	cout << "Warning: this MPI implementation is under development. It may not work..." << endl << endl;
#endif

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
	cout << "=== Search Space ===" << endl;

	// use standard CRF features for heuristic feature function
	cout << "Heuristic feature function: ";
	cout << "standard CRF features" << endl;
	HCSearch::IFeatureFunction* heuristicFeatFunc = new HCSearch::StandardFeatures();

	// use standard CRF features for cost feature function
	cout << "Cost feature function: ";
	cout << "standard CRF features" << endl;
	HCSearch::IFeatureFunction* costFeatFunc = new HCSearch::StandardFeatures();

	// use IID logistic regression as initial state prediction function
	cout << "Initial state prediction function: ";
	cout << "IID logistic regression" << endl;
	HCSearch::IInitialPredictionFunction* initPredFunc = new HCSearch::LogRegInit();

	// use stochastic successor function
	cout << "Successor function: ";
	HCSearch::ISuccessorFunction* successor = NULL;
	switch (po.successorsMode)
	{
	case MyProgramOptions::ProgramOptions::FLIPBIT:
		cout << "flipbit" << endl;
		successor = new HCSearch::FlipbitSuccessor();
		break;
	case MyProgramOptions::ProgramOptions::STOCHASTIC:
		cout << "stochastic (T=" << po.cutParam << ")" << endl;
		successor = new HCSearch::StochasticSuccessor(po.cutParam);
		break;
	default:
		cerr << "[Error] undefined successor mode." << endl;
	}

	// use Hamming loss function
	cout << "Loss function: ";
	cout << "Hamming loss" << endl;
	HCSearch::ILossFunction* lossFunc = new HCSearch::HammingLoss();

	cout << endl;

	// construct search space from these functions that we specified
	return new HCSearch::SearchSpace(heuristicFeatFunc, costFeatFunc, initPredFunc, successor, lossFunc);
}

HCSearch::ISearchProcedure* setupSearchProcedure(MyProgramOptions::ProgramOptions po)
{
	cout << "=== Search Procedure ===" << endl;

	HCSearch::ISearchProcedure* searchProcedure = NULL;
	switch (po.searchProcedureMode)
	{
	case MyProgramOptions::ProgramOptions::GREEDY:
		cout << "Using greedy search." << endl;
		searchProcedure = new HCSearch::GreedySearchProcedure();
		break;
	case MyProgramOptions::ProgramOptions::BREADTH_BEAM:
		cout << "Using breadth-first beam search." << endl;
		cout << "Beam size=" << po.beamSize << endl;
		searchProcedure = new HCSearch::BreadthFirstBeamSearchProcedure(po.beamSize);
		break;
	case MyProgramOptions::ProgramOptions::BEST_BEAM:
		cout << "Using best-first beam search." << endl;
		cout << "Beam size=" << po.beamSize << endl;
		searchProcedure = new HCSearch::BestFirstBeamSearchProcedure(po.beamSize);
		break;
	default:
		cerr << "[Error] undefined search procedure mode." << endl;
	}

	cout << endl;

	return searchProcedure;
}

void run(MyProgramOptions::ProgramOptions po)
{
	// time bound
	int timeBound = po.timeBound;

	// paths
	string heuristicModelPath = HCSearch::Global::settings->paths->OUTPUT_HEURISTIC_MODEL_FILE;
	string costModelPath = HCSearch::Global::settings->paths->OUTPUT_COST_H_MODEL_FILE;
	string costOracleHModelPath = HCSearch::Global::settings->paths->OUTPUT_COST_ORACLE_H_MODEL_FILE;

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
		HCSearch::SearchType mode = *it;
		switch (mode)
		{
		case HCSearch::LEARN_H:
		{
			cout << "=== Learning H ===" << endl;

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
			
			delete heuristicModel;
			break;
		}
		case HCSearch::LEARN_C:
		{
			cout << "=== Learning C with Learned H ===" << endl;

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
			
			delete heuristicModel;
			delete costModel;
			break;
		}
		case HCSearch::LEARN_C_ORACLE_H:
		{
			cout << "=== Learning C with Oracle H ===" << endl;

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
			
			delete costOracleHModel;
			break;
		}
		case HCSearch::LL:
		{
			cout << "=== Inference LL ===" << endl;

			// run LL search on test examples
			int start, end;
			HCSearch::Dataset::computeTaskRange(HCSearch::Global::settings->RANK, XTest.size(), 
				HCSearch::Global::settings->NUM_PROCESSES, start, end);
			for (int i = start; i < end; i++)
			{
				for (int iter = 0; iter < po.numTestIterations; iter++)
				{
					cout << endl << "LL Search: (iter " << iter << ") beginning search on " << XTest[i]->getFileName() << " (example " << i << ")..." << endl;

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
						<< "_" << HCSearch::SearchTypeStrings[HCSearch::LL] 
						<< "_" << HCSearch::DatasetTypeStrings[meta.setType] 
						<< "_time" << timeBound 
							<< "_fold" << meta.iter 
							<< "_" << meta.exampleName << ".txt";
					HCSearch::SavePrediction::saveLabels(YPred, ssPredictNodes.str());
				}
			}

			break;
		}
		case HCSearch::HL:
		{
			cout << "=== Inference HL ===" << endl;

			// load heuristic, run HL search on test examples
			HCSearch::IRankModel* heuristicModel = HCSearch::Model::loadModel(heuristicModelPath, rankerType);

			int start, end;
			HCSearch::Dataset::computeTaskRange(HCSearch::Global::settings->RANK, XTest.size(), 
				HCSearch::Global::settings->NUM_PROCESSES, start, end);
			for (int i = start; i < end; i++)
			{
				for (int iter = 0; iter < po.numTestIterations; iter++)
				{
					cout << endl << "HL Search: (iter " << iter << ") beginning search on " << XTest[i]->getFileName() << " (example " << i << ")..." << endl;

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
						<< "_" << HCSearch::SearchTypeStrings[HCSearch::HL] 
						<< "_" << HCSearch::DatasetTypeStrings[meta.setType] 
						<< "_time" << timeBound 
							<< "_fold" << meta.iter 
							<< "_" << meta.exampleName << ".txt";
					HCSearch::SavePrediction::saveLabels(YPred, ssPredictNodes.str());
				}
			}

			delete heuristicModel;
			break;
		}
		case HCSearch::LC:
		{
			cout << "=== Inference LC ===" << endl;

			// load cost oracle H, run LC search on test examples
			HCSearch::IRankModel* costModel = HCSearch::Model::loadModel(costOracleHModelPath, rankerType);

			int start, end;
			HCSearch::Dataset::computeTaskRange(HCSearch::Global::settings->RANK, XTest.size(), 
				HCSearch::Global::settings->NUM_PROCESSES, start, end);
			for (int i = start; i < end; i++)
			{
				for (int iter = 0; iter < po.numTestIterations; iter++)
				{
					cout << endl << "LC Search: (iter " << iter << ") beginning search on " << XTest[i]->getFileName() << " (example " << i << ")..." << endl;

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
						<< "_" << HCSearch::SearchTypeStrings[HCSearch::LC] 
						<< "_" << HCSearch::DatasetTypeStrings[meta.setType] 
						<< "_time" << timeBound 
							<< "_fold" << meta.iter 
							<< "_" << meta.exampleName << ".txt";
					HCSearch::SavePrediction::saveLabels(YPred, ssPredictNodes.str());
				}
			}

			delete costModel;
			break;
		}
		case HCSearch::HC:
		{
			cout << "=== Inference HC ===" << endl;

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
					cout << endl << "HC Search: (iter " << iter << ") beginning search on " << XTest[i]->getFileName() << " (example " << i << ")..." << endl;

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
						<< "_" << HCSearch::SearchTypeStrings[HCSearch::HC] 
						<< "_" << HCSearch::DatasetTypeStrings[meta.setType] 
						<< "_time" << timeBound 
							<< "_fold" << meta.iter 
							<< "_" << meta.exampleName << ".txt";
					HCSearch::SavePrediction::saveLabels(YPred, ssPredictNodes.str());
				}
			}

			delete heuristicModel;
			delete costModel;
			break;
		}
		default:
			cerr << "Error!" << endl;
		}
	}

	// clean up
	delete searchSpace;
	delete searchProcedure;
	HCSearch::Dataset::unloadDataset(XTrain, YTrain, XValidation, YValidation, XTest, YTest);
}

void printInfo(MyProgramOptions::ProgramOptions po)
{
	cout << "=== Program Schedule ===" << endl;

	int cnt = 1;
	for (vector< HCSearch::SearchType >::iterator it = po.schedule.begin();
		it != po.schedule.end(); ++it)
	{
		HCSearch::SearchType mode = *it;
		cout << cnt << ". " << HCSearch::SearchTypeStrings[mode] << endl;
		cnt++;
	}

	cout << endl;

	cout << "=== Program Options ===" << endl;
	cout << "Rank learner: " << HCSearch::RankerTypeStrings[po.rankLearnerType] << endl;
	cout << "Num training iterations: " << po.numTrainIterations << endl;
	cout << "Num testing iterations: " << po.numTestIterations << endl;
	cout << "Save anytime predictions: " << po.saveAnytimePredictions << endl;
	cout << "Save features files: " << po.saveFeaturesFiles << endl;
	cout << endl;

	if (po.verboseMode)
	{
		cout << "=== Paths ===" << endl;

		cout << "EXTERNAL_DIR: " << HCSearch::Global::settings->paths->EXTERNAL_DIR << endl;
		cout << "INPUT_DIR: " << HCSearch::Global::settings->paths->INPUT_DIR  << endl;
		cout << "OUTPUT_DIR: " << HCSearch::Global::settings->paths->OUTPUT_DIR  << endl;
		cout << "LIBLINEAR_DIR: " << HCSearch::Global::settings->paths->LIBLINEAR_DIR  << endl;
		cout << "LIBSVM_DIR: " << HCSearch::Global::settings->paths->LIBSVM_DIR  << endl;
		cout << "SVMRANK_DIR: " << HCSearch::Global::settings->paths->SVMRANK_DIR  << endl;

		cout << endl;

		cout << "INPUT_NODES_DIR: " << HCSearch::Global::settings->paths->INPUT_NODES_DIR << endl;
		cout << "INPUT_EDGES_DIR: " << HCSearch::Global::settings->paths->INPUT_EDGES_DIR  << endl;
		cout << "INPUT_META_DIR: " << HCSearch::Global::settings->paths->INPUT_META_DIR  << endl;
		cout << "INPUT_SEGMENTS_DIR: " << HCSearch::Global::settings->paths->INPUT_SEGMENTS_DIR  << endl;
		cout << "INPUT_SPLITS_DIR: " << HCSearch::Global::settings->paths->INPUT_SPLITS_DIR  << endl;
		cout << "INPUT_SPLITS_FOLDER_NAME: " << HCSearch::Global::settings->paths->INPUT_SPLITS_FOLDER_NAME  << endl;

		cout << "INPUT_SPLITS_TRAIN_FILE: " << HCSearch::Global::settings->paths->INPUT_SPLITS_TRAIN_FILE << endl;
		cout << "INPUT_SPLITS_VALIDATION_FILE: " << HCSearch::Global::settings->paths->INPUT_SPLITS_VALIDATION_FILE  << endl;
		cout << "INPUT_SPLITS_TEST_FILE: " << HCSearch::Global::settings->paths->INPUT_SPLITS_TEST_FILE  << endl;

		cout << "INPUT_METADATA_FILE: " << HCSearch::Global::settings->paths->INPUT_METADATA_FILE  << endl;
		cout << "INPUT_CODEBOOK_FILE: " << HCSearch::Global::settings->paths->INPUT_CODEBOOK_FILE  << endl;
		cout << "INPUT_SPLITS_FOLDER_NAME: " << HCSearch::Global::settings->paths->INPUT_INITFUNC_TRAINING_FILE  << endl;

		cout << endl;

		cout << "OUTPUT_LOGS_DIR: " << HCSearch::Global::settings->paths->OUTPUT_LOGS_DIR << endl;
		cout << "OUTPUT_MODELS_DIR: " << HCSearch::Global::settings->paths->OUTPUT_MODELS_DIR  << endl;
		cout << "OUTPUT_RESULTS_DIR: " << HCSearch::Global::settings->paths->OUTPUT_RESULTS_DIR  << endl;
		cout << "OUTPUT_TEMP_DIR: " << HCSearch::Global::settings->paths->OUTPUT_TEMP_DIR  << endl;

		cout << "OUTPUT_HEURISTIC_FEATURES_FILE: " << HCSearch::Global::settings->paths->OUTPUT_HEURISTIC_FEATURES_FILE  << endl;
		cout << "OUTPUT_COST_H_FEATURES_FILE: " << HCSearch::Global::settings->paths->OUTPUT_COST_H_FEATURES_FILE  << endl;
		cout << "OUTPUT_COST_ORACLE_H_FEATURES_FILE: " << HCSearch::Global::settings->paths->OUTPUT_COST_ORACLE_H_FEATURES_FILE << endl;

		cout << "OUTPUT_ARCHIVED_HEURISTIC_FEATURES_FILE: " << HCSearch::Global::settings->paths->OUTPUT_ARCHIVED_HEURISTIC_FEATURES_FILE  << endl;
		cout << "OUTPUT_ARCHIVED_COST_H_FEATURES_FILE: " << HCSearch::Global::settings->paths->OUTPUT_ARCHIVED_COST_H_FEATURES_FILE  << endl;
		cout << "OUTPUT_ARCHIVED_COST_ORACLE_H_FEATURES_FILE: " << HCSearch::Global::settings->paths->OUTPUT_ARCHIVED_COST_ORACLE_H_FEATURES_FILE  << endl;

		cout << "OUTPUT_HEURISTIC_ONLINE_WEIGHTS_FILE: " << HCSearch::Global::settings->paths->OUTPUT_HEURISTIC_ONLINE_WEIGHTS_FILE  << endl;
		cout << "OUTPUT_COST_H_ONLINE_WEIGHTS_FILE: " << HCSearch::Global::settings->paths->OUTPUT_COST_H_ONLINE_WEIGHTS_FILE  << endl;
		cout << "OUTPUT_COST_ORACLE_H_ONLINE_WEIGHTS_FILE: " << HCSearch::Global::settings->paths->OUTPUT_COST_ORACLE_H_ONLINE_WEIGHTS_FILE << endl;

		cout << "OUTPUT_HEURISTIC_MODEL_FILE: " << HCSearch::Global::settings->paths->OUTPUT_HEURISTIC_MODEL_FILE  << endl;
		cout << "OUTPUT_COST_H_MODEL_FILE: " << HCSearch::Global::settings->paths->OUTPUT_COST_H_MODEL_FILE  << endl;
		cout << "OUTPUT_COST_ORACLE_H_MODEL_FILE: " << HCSearch::Global::settings->paths->OUTPUT_COST_ORACLE_H_MODEL_FILE  << endl;

		cout << "OUTPUT_LOG_FILE: " << HCSearch::Global::settings->paths->OUTPUT_LOG_FILE  << endl;

		cout << "OUTPUT_INITFUNC_MODEL_FILE: " << HCSearch::Global::settings->paths->OUTPUT_INITFUNC_MODEL_FILE  << endl;
		cout << "OUTPUT_INITFUNC_FEATURES_FILE: " << HCSearch::Global::settings->paths->OUTPUT_INITFUNC_FEATURES_FILE << endl;
		cout << "OUTPUT_INITFUNC_PREDICT_FILE: " << HCSearch::Global::settings->paths->OUTPUT_INITFUNC_PREDICT_FILE  << endl;

		cout << endl;
	}
}