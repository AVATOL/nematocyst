#include <iostream>
#include "HCSearch.hpp" 
#include "Main.hpp"
#include "MyProgramOptions.hpp"

using namespace std;

int main(int argc, char* argv[])
{
	// initialize HCSearch
	HCSearch::Setup::initialize(argc, argv);

	// parse arguments
	MyProgramOptions::ProgramOptions po = MyProgramOptions::ProgramOptions::parseArguments(argc, argv);

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
	string heuristicModelPath = HCSearch::Global::settings->paths->EXPERIMENT_HEURISTIC_MODEL_FILE;
	string costModelPath = HCSearch::Global::settings->paths->EXPERIMENT_COST_H_MODEL_FILE;
	string costOracleHModelPath = HCSearch::Global::settings->paths->EXPERIMENT_COST_L_MODEL_FILE;

	// params
	enum HCSearch::RankerType rankerType = HCSearch::RankerType::ONLINE_RANK;

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
	searchMetadata.exampleName; //TODO
	searchMetadata.iter; //TODO
	searchMetadata.setType = HCSearch::DatasetType::TEST;

	// infer LL
	HCSearch::Inference::runLLSearch(XTest[0], YTest[0], timeBound, searchSpace, searchProcedure, searchMetadata);

	// infer HL
	HCSearch::Inference::runHLSearch(XTest[0], YTest[0], timeBound, searchSpace, searchProcedure, heuristicModel, searchMetadata);

	// infer LC
	HCSearch::Inference::runLCSearch(XTest[0], YTest[0], timeBound, searchSpace, searchProcedure, costOracleHModel, searchMetadata);

	// infer HC
	HCSearch::Inference::runHCSearch(XTest[0], timeBound, searchSpace, searchProcedure, heuristicModel, costModel, searchMetadata);

	// clean up
	HCSearch::Dataset::unloadDataset(XTrain, YTrain, XValidation, YValidation, XTest, YTest);
}

HCSearch::SearchSpace* setupSearchSpace(MyProgramOptions::ProgramOptions po)
{
	// use standard CRF features for both heuristic and cost feature functions
	HCSearch::IFeatureFunction* heuristicFeatFunc = new HCSearch::StandardFeatures();
	HCSearch::IFeatureFunction* costFeatFunc = new HCSearch::StandardFeatures();

	// use IID logistic regression as initial state prediction function
	HCSearch::IInitialPredictionFunction* initPredFunc = new HCSearch::LogRegInit();

	// use stochastic successor function
	HCSearch::ISuccessorFunction* successor = NULL;
	switch (po.successorsMode)
	{
	case MyProgramOptions::ProgramOptions::FLIPBIT:
		successor = new HCSearch::FlipbitSuccessor();
		break;
	case MyProgramOptions::ProgramOptions::STOCHASTIC:
		successor = new HCSearch::StochasticSuccessor(po.cutParam);
		break;
	default:
		cerr << "[Error] undefined successor mode." << endl;
	}

	// use Hamming loss function
	HCSearch::ILossFunction* lossFunc = new HCSearch::HammingLoss();

	// construct search space from these functions that we specified
	return new HCSearch::SearchSpace(heuristicFeatFunc, costFeatFunc, initPredFunc, successor, lossFunc);
}

HCSearch::ISearchProcedure* setupSearchProcedure(MyProgramOptions::ProgramOptions po)
{
	HCSearch::ISearchProcedure* searchProcedure = NULL;
	switch (po.searchProcedureMode)
	{
	case MyProgramOptions::ProgramOptions::GREEDY:
		searchProcedure = new HCSearch::GreedySearchProcedure();
		break;
	case MyProgramOptions::ProgramOptions::BREADTH_BEAM:
		searchProcedure = new HCSearch::BreadthFirstBeamSearchProcedure(po.beamSize);
		break;
	case MyProgramOptions::ProgramOptions::BEST_BEAM:
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
	string heuristicModelPath = HCSearch::Global::settings->paths->EXPERIMENT_HEURISTIC_MODEL_FILE;
	string costModelPath = HCSearch::Global::settings->paths->EXPERIMENT_COST_H_MODEL_FILE;
	string costOracleHModelPath = HCSearch::Global::settings->paths->EXPERIMENT_COST_L_MODEL_FILE;

	// params
	HCSearch::RankerType rankerType = HCSearch::RankerType::ONLINE_RANK; //TODO

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
	typedef enum MyProgramOptions::ProgramOptions::Modes Modes_t;
	for (vector< Modes_t >::iterator it = po.schedule.begin();
		it != po.schedule.end(); ++it)
	{
		Modes_t mode = *it;
		switch (mode)
		{
		case Modes_t::LEARN_H:
			{
			HCSearch::IRankModel* heuristicModel = HCSearch::Learning::learnH(XTrain, YTrain, XValidation, YValidation, timeBound, searchSpace, searchProcedure);
			HCSearch::Model::saveModel(heuristicModel, heuristicModelPath, rankerType);
			break;
			}
		case Modes_t::LEARN_C:
			{
			HCSearch::IRankModel* heuristicModel = heuristicModel = HCSearch::Model::loadModel(heuristicModelPath, rankerType);
			HCSearch::IRankModel* costModel = HCSearch::Learning::learnC(XTrain, YTrain, XValidation, YValidation, heuristicModel, timeBound, searchSpace, searchProcedure);
			HCSearch::Model::saveModel(costModel, costModelPath, rankerType);
			break;
			}
		case Modes_t::LEARN_C_ORACLE_H:
			{
			HCSearch::IRankModel* costOracleHModel = HCSearch::Learning::learnCWithOracleH(XTrain, YTrain, XValidation, YValidation, timeBound, searchSpace, searchProcedure);
			HCSearch::Model::saveModel(costOracleHModel, costOracleHModelPath, rankerType);
			break;
			}
		case Modes_t::INFER_LL:
			{
			//TODO
			break;
			}
		case Modes_t::INFER_HL:
			{
			//TODO
			break;
			}
		case Modes_t::INFER_LC:
			{
			//TODO
			break;
			}
		case Modes_t::INFER_HC:
			{
			//TODO
			break;
			}
		default:
			cerr << "Error!" << endl;
		}
	}

	// clean up
	HCSearch::Dataset::unloadDataset(XTrain, YTrain, XValidation, YValidation, XTest, YTest);
}
