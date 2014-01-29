#include <iostream>
#include "HCSearch.hpp" 

using namespace std;

HCSearch::SearchSpace* setupSearchSpace();
void demo(int argc, char* argv[]);

HCSearch::SearchSpace* setupSearchSpace()
{
	// use standard CRF features for both heuristic and cost feature functions
	HCSearch::IFeatureFunction* heuristicFeatFunc = new HCSearch::StandardFeatures();
	HCSearch::IFeatureFunction* costFeatFunc = new HCSearch::StandardFeatures();

	// use IID logistic regression as initial prediction function
	HCSearch::IInitialPredictionFunction* logRegInitPredFunc = new HCSearch::LogRegInit();

	// use stochastic successor function
	HCSearch::ISuccessorFunction* stochasticSuccessor = new HCSearch::StochasticSuccessor();

	// use Hamming loss function
	HCSearch::ILossFunction* lossFunc = new HCSearch::HammingLoss();

	// construct search space from these functions that we specified
	return new HCSearch::SearchSpace(heuristicFeatFunc, costFeatFunc, logRegInitPredFunc, stochasticSuccessor, lossFunc);
}

void demo(int argc, char* argv[])
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
	HCSearch::Utility::loadDataset(XTrain, YTrain, XValidation, YValidation, XTest, YTest);
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
	HCSearch::Utility::saveModel(heuristicModel, heuristicModelPath, rankerType);
	HCSearch::Utility::saveModel(costModel, costModelPath, rankerType);
	HCSearch::Utility::saveModel(costOracleHModel, costOracleHModelPath, rankerType);

	// to demonstrate loading...
	delete heuristicModel;
	delete costModel;
	delete costOracleHModel;

	// load model
	heuristicModel = HCSearch::Utility::loadModel(heuristicModelPath, rankerType);
	costModel = HCSearch::Utility::loadModel(costModelPath, rankerType);
	costOracleHModel = HCSearch::Utility::loadModel(costOracleHModelPath, rankerType);

	// set up metadat for first test example
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
	HCSearch::Utility::unloadDataset(XTrain, YTrain, XValidation, YValidation, XTest, YTest);
}

#ifdef USE_MPI

int main(int argc, char* argv[])
{
	// initialize HCSearch
	HCSearch::Setup::initialize(argc, argv);

	cout << "Rank=" << HCSearch::Global::settings->RANK << endl;

	// parse arguments
	//TODO

	// configure settings
	HCSearch::Setup::configure("dataNematoDemo", "EXPERIMENT_output");

	// demo
	//demo(argc, argv);

	if (HCSearch::Global::settings->RANK == 0)
		HCSearch::abort();

	// finalize
	HCSearch::Setup::finalize();
    return 0;
}

#else

int main(int argc, char* argv[])
{
	// initialize HCSearch
	HCSearch::Setup::initialize(argc, argv);

	// parse arguments
	//TODO

	// demo
	demo(argc, argv);

	// finalize
	HCSearch::Setup::finalize();

    return 0;
}

#endif
