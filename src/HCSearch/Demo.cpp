#include "Demo.hpp"

void demo(int timeBound)
{
	// paths
	string heuristicModelPath = HCSearch::Global::settings->paths->OUTPUT_HEURISTIC_MODEL_FILE;
	string costModelPath = HCSearch::Global::settings->paths->OUTPUT_COST_H_MODEL_FILE;

	// params
	HCSearch::RankerType rankerType = HCSearch::SVM_RANK;

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

	// save model
	HCSearch::Model::saveModel(heuristicModel, heuristicModelPath, rankerType);
	HCSearch::Model::saveModel(costModel, costModelPath, rankerType);

	// to demonstrate loading...
	delete heuristicModel;
	delete costModel;

	// load model
	heuristicModel = HCSearch::Model::loadModel(heuristicModelPath, rankerType);
	costModel = HCSearch::Model::loadModel(costModelPath, rankerType);

	// run HC search on all test examples
	for (int i = 0; i < numTestExamples; i++)
	{
		// set up metadata
		HCSearch::ISearchProcedure::SearchMetadata searchMetadata;
		searchMetadata.saveAnytimePredictions = false;
		searchMetadata.exampleName = XTest[i]->filename;
		searchMetadata.iter = 0;
		searchMetadata.setType = HCSearch::TEST;

		// infer HC
		HCSearch::Inference::runHCSearch(XTest[i], timeBound, searchSpace, searchProcedure, heuristicModel, costModel, searchMetadata);
	}

	// clean up
	delete searchSpace;
	delete searchProcedure;
	delete heuristicModel;
	delete costModel;
	HCSearch::Dataset::unloadDataset(XTrain, YTrain, XValidation, YValidation, XTest, YTest);
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
