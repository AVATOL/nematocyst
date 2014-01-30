#include <iostream>
#include "HCSearch.hpp" 
#include "Main.hpp"
#include "MyProgramOptions.hpp"

using namespace std;

int main(int argc, char* argv[])
{
	// initialize HCSearch
	HCSearch::Setup::initialize(argc, argv);

	cout << "Rank=" << HCSearch::Global::settings->RANK << endl;

	// parse arguments
	MyProgramOptions::ProgramOptions po = MyProgramOptions::ProgramOptions::parseArguments(argc, argv);

	// configure settings
	HCSearch::Setup::configure(po.inputDir, po.outputDir);

	// demo
	if (po.demoMode)
		demo();
	else
		run(po);

	if (HCSearch::Global::settings->RANK == 0)
		HCSearch::abort();

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

void demo()
{
	// time bound
	int timeBound = 50;

	// paths
	string heuristicModelPath = "TODO";
	string costModelPath = "TODO";
	string costOracleHModelPath = "TODO";

	// params
	HCSearch::RankerType rankerType = HCSearch::RankerType::ONLINE_RANK;

	// Datasets
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

void run(MyProgramOptions::ProgramOptions po)
{
	typedef MyProgramOptions::ProgramOptions::Modes Modes_t;

	// set up search space
	//TODO

	// set up search procedure
	//TODO

	// run the appropriate mode
	for (vector< Modes_t >::iterator it = po.schedule.begin();
		it != po.schedule.end(); ++it)
	{
		Modes_t mode = *it;

		switch (mode)
		{
		case Modes_t::LEARN_H:
			//TODO
			break;
		case Modes_t::LEARN_C:
			//TODO
			break;
		case Modes_t::LEARN_C_ORACLE_H:
			//TODO
			break;
		case Modes_t::INFER_LL:
			//TODO
			break;
		case Modes_t::INFER_HL:
			//TODO
			break;
		case Modes_t::INFER_LC:
			//TODO
			break;
		case Modes_t::INFER_HC:
			//TODO
			break;
		default:
			cerr << "Error!" << endl;
		}
	}
}
