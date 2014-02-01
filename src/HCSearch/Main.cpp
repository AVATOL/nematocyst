#include <iostream>
#include "../HCSearchLib/HCSearch.hpp"
#include "MyProgramOptions.hpp"
#include "Main.hpp"

using namespace std;

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
	HCSearch::Setup::configure(po.inputDir, po.outputDir);

	// demo or run full program
	if (po.demoMode)
		demo(po);
	else
		run(po);

	// finalize
	HCSearch::Setup::finalize();
    return 0;
}

HCSearch::SearchSpace* setupSearchSpace()
{
	// use standard CRF features for both heuristic and cost feature functions
	HCSearch::IFeatureFunction* heuristicFeatFunc = new HCSearch::StandardFeatures();
	HCSearch::IFeatureFunction* costFeatFunc = new HCSearch::StandardFeatures();

	// use IID logistic regression as initial state prediction function
	HCSearch::IInitialPredictionFunction* logRegInitPredFunc = new HCSearch::LogRegInit();

	// use stochastic successor function
	HCSearch::ISuccessorFunction* stochasticSuccessor = new HCSearch::StochasticSuccessor();

	// use Hamming loss function
	HCSearch::ILossFunction* lossFunc = new HCSearch::HammingLoss();

	// construct search space from these functions that we specified
	return new HCSearch::SearchSpace(heuristicFeatFunc, costFeatFunc, logRegInitPredFunc, stochasticSuccessor, lossFunc);
}

void demo(MyProgramOptions::ProgramOptions po)
{
	// time bound
	int timeBound = po.timeBound;

	// paths
	string heuristicModelPath = HCSearch::Global::settings->paths->OUTPUT_HEURISTIC_MODEL_FILE;
	string costModelPath = HCSearch::Global::settings->paths->OUTPUT_COST_H_MODEL_FILE;
	string costOracleHModelPath = HCSearch::Global::settings->paths->OUTPUT_COST_ORACLE_H_MODEL_FILE;

	// params
	HCSearch::RankerType rankerType = HCSearch::ONLINE_RANK;

	// datasets
	vector< HCSearch::ImgFeatures* > XTrain;
	vector< HCSearch::ImgLabeling* > YTrain;
	vector< HCSearch::ImgFeatures* > XValidation;
	vector< HCSearch::ImgLabeling* > YValidation;
	vector< HCSearch::ImgFeatures* > XTest;
	vector< HCSearch::ImgLabeling* > YTest;

	// load dataset
	HCSearch::Dataset::loadDataset(XTrain, YTrain, XValidation, YValidation, XTest, YTest);
	const int numTestExamples = XTest.size();

	// load search space functions and search space
	HCSearch::SearchSpace* searchSpace = setupSearchSpace();

	// load search procedure
	HCSearch::ISearchProcedure* searchProcedure = new HCSearch::GreedySearchProcedure();

	// train H
	HCSearch::IRankModel* heuristicModel = HCSearch::Learning::learnH(XTrain, YTrain, XValidation, YValidation, timeBound, searchSpace, searchProcedure);

	// train C
	HCSearch::IRankModel* costModel = HCSearch::Learning::learnC(XTrain, YTrain, XValidation, YValidation, heuristicModel, timeBound, searchSpace, searchProcedure);

	// train C with oracle H
	HCSearch::IRankModel* costOracleHModel = HCSearch::Learning::learnCWithOracleH(XTrain, YTrain, XValidation, YValidation, timeBound, searchSpace, searchProcedure);

	// save model
	HCSearch::Model::saveModel(heuristicModel, heuristicModelPath, rankerType);
	HCSearch::Model::saveModel(costModel, costModelPath, rankerType);
	HCSearch::Model::saveModel(costOracleHModel, costOracleHModelPath, rankerType);

	// to demonstrate loading...
	delete heuristicModel;
	delete costModel;
	delete costOracleHModel;

	// load model
	heuristicModel = HCSearch::Model::loadModel(heuristicModelPath, rankerType);
	costModel = HCSearch::Model::loadModel(costModelPath, rankerType);
	costOracleHModel = HCSearch::Model::loadModel(costOracleHModelPath, rankerType);

	// set up metadata for first test example
	HCSearch::ISearchProcedure::SearchMetadata searchMetadata;
	searchMetadata.saveAnytimePredictions = false;
	searchMetadata.exampleName; //TODO
	searchMetadata.iter; //TODO
	searchMetadata.setType = HCSearch::TEST;

	// infer LL
	HCSearch::Inference::runLLSearch(XTest[0], YTest[0], timeBound, searchSpace, searchProcedure, searchMetadata);

	// infer HL
	HCSearch::Inference::runHLSearch(XTest[0], YTest[0], timeBound, searchSpace, searchProcedure, heuristicModel, searchMetadata);

	// infer LC
	HCSearch::Inference::runLCSearch(XTest[0], YTest[0], timeBound, searchSpace, searchProcedure, costOracleHModel, searchMetadata);

	// infer HC
	HCSearch::Inference::runHCSearch(XTest[0], timeBound, searchSpace, searchProcedure, heuristicModel, costModel, searchMetadata);

	// clean up
	delete searchSpace;
	delete searchProcedure;
	delete heuristicModel;
	delete costModel;
	delete costOracleHModel;
	HCSearch::Dataset::unloadDataset(XTrain, YTrain, XValidation, YValidation, XTest, YTest);
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
		cout << "stochastic" << endl;
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
	HCSearch::RankerType rankerType = HCSearch::ONLINE_RANK; //TODO

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
	typedef MyProgramOptions::ProgramOptions ProgramOptions_t;
	for (vector< ProgramOptions_t::Modes >::iterator it = po.schedule.begin();
		it != po.schedule.end(); ++it)
	{
		ProgramOptions_t::Modes mode = *it;
		switch (mode)
		{
		case ProgramOptions_t::LEARN_H:
		{
			cout << "=== Learning H ===" << endl;

			// learn heuristic, save heuristic model
			HCSearch::IRankModel* heuristicModel = HCSearch::Learning::learnH(XTrain, YTrain, XValidation, YValidation, timeBound, searchSpace, searchProcedure);
			
			if (HCSearch::Global::settings->RANK == 0)
				HCSearch::Model::saveModel(heuristicModel, heuristicModelPath, rankerType);
			
			delete heuristicModel;
			break;
		}
		case ProgramOptions_t::LEARN_C:
		{
			cout << "=== Learning C with Learned H ===" << endl;

			// load heuristic, learn cost, save cost model
			HCSearch::IRankModel* heuristicModel = HCSearch::Model::loadModel(heuristicModelPath, rankerType);
			HCSearch::IRankModel* costModel = HCSearch::Learning::learnC(XTrain, YTrain, XValidation, YValidation, heuristicModel, timeBound, searchSpace, searchProcedure);
			
			if (HCSearch::Global::settings->RANK == 0)
				HCSearch::Model::saveModel(costModel, costModelPath, rankerType);
			
			delete heuristicModel;
			delete costModel;
			break;
		}
		case ProgramOptions_t::LEARN_C_ORACLE_H:
		{
			cout << "=== Learning C with Oracle H ===" << endl;

			// learn cost, save cost model
			HCSearch::IRankModel* costOracleHModel = HCSearch::Learning::learnCWithOracleH(XTrain, YTrain, XValidation, YValidation, timeBound, searchSpace, searchProcedure);
			
			if (HCSearch::Global::settings->RANK == 0)
				HCSearch::Model::saveModel(costOracleHModel, costOracleHModelPath, rankerType);
			
			delete costOracleHModel;
			break;
		}
		case ProgramOptions_t::INFER_LL:
		{
			cout << "=== Inference LL ===" << endl;

			// run LL search on test examples
			int start, end;
			HCSearch::Dataset::computeTaskRange(HCSearch::Global::settings->RANK, XTest.size(), 
				HCSearch::Global::settings->NUM_PROCESSES, start, end);
			for (int i = start; i < end; i++)
			{
				cout << endl << "LL Search: beginning search on " << XTest[i]->getFileName() << " (example " << i << ")..." << endl;

				// setup meta
				HCSearch::ISearchProcedure::SearchMetadata meta;
				meta.saveAnytimePredictions = true;
				meta.setType = HCSearch::TEST;
				meta.exampleName = XTest[i]->getFileName();
				meta.iter = 0; //TODO

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

			break;
		}
		case ProgramOptions_t::INFER_HL:
		{
			cout << "=== Inference HL ===" << endl;

			// load heuristic, run HL search on test examples
			HCSearch::IRankModel* heuristicModel = HCSearch::Model::loadModel(heuristicModelPath, rankerType);

			int start, end;
			HCSearch::Dataset::computeTaskRange(HCSearch::Global::settings->RANK, XTest.size(), 
				HCSearch::Global::settings->NUM_PROCESSES, start, end);
			for (int i = start; i < end; i++)
			{
				cout << endl << "HL Search: beginning search on " << XTest[i]->getFileName() << " (example " << i << ")..." << endl;

				// setup meta
				HCSearch::ISearchProcedure::SearchMetadata meta;
				meta.saveAnytimePredictions = true;
				meta.setType = HCSearch::TEST;
				meta.exampleName = XTest[i]->getFileName();
				meta.iter = 0; //TODO

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

			delete heuristicModel;
			break;
		}
		case ProgramOptions_t::INFER_LC:
		{
			cout << "=== Inference LC ===" << endl;

			// load cost oracle H, run LC search on test examples
			HCSearch::IRankModel* costModel = HCSearch::Model::loadModel(costOracleHModelPath, rankerType);

			int start, end;
			HCSearch::Dataset::computeTaskRange(HCSearch::Global::settings->RANK, XTest.size(), 
				HCSearch::Global::settings->NUM_PROCESSES, start, end);
			for (int i = start; i < end; i++)
			{
				cout << endl << "LC Search: beginning search on " << XTest[i]->getFileName() << " (example " << i << ")..." << endl;

				// setup meta
				HCSearch::ISearchProcedure::SearchMetadata meta;
				meta.saveAnytimePredictions = true;
				meta.setType = HCSearch::TEST;
				meta.exampleName = XTest[i]->getFileName();
				meta.iter = 0; //TODO

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

			delete costModel;
			break;
		}
		case ProgramOptions_t::INFER_HC:
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
				cout << endl << "HC Search: beginning search on " << XTest[i]->getFileName() << " (example " << i << ")..." << endl;

				// setup meta
				HCSearch::ISearchProcedure::SearchMetadata meta;
				meta.saveAnytimePredictions = true;
				meta.setType = HCSearch::TEST;
				meta.exampleName = XTest[i]->getFileName();
				meta.iter = 0; //TODO

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
