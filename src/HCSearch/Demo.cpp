#include "Demo.hpp"

void demo(int timeBound)
{
	// This demo appears in the Quick Start (API) guide.

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
	HCSearch::IFeatureFunction* heuristicFeatFunc = new HCSearch::StandardFeatures();
	HCSearch::IFeatureFunction* costFeatFunc = new HCSearch::StandardFeatures();
	HCSearch::IInitialPredictionFunction* logRegInitPredFunc = new HCSearch::LogRegInit();
	HCSearch::ISuccessorFunction* stochasticSuccessor = new HCSearch::StochasticSuccessor();
	HCSearch::ILossFunction* lossFunc = new HCSearch::HammingLoss();
	HCSearch::SearchSpace* searchSpace = new  HCSearch::SearchSpace(heuristicFeatFunc, costFeatFunc, logRegInitPredFunc, stochasticSuccessor, lossFunc);

	// load search procedure
	HCSearch::ISearchProcedure* searchProcedure = new HCSearch::GreedySearchProcedure();

	// train H
	HCSearch::IRankModel* heuristicModel = HCSearch::Learning::learnH(XTrain, YTrain, XValidation, YValidation, 
	timeBound, searchSpace, searchProcedure, HCSearch::SVM_RANK, 1);

	// train C
	HCSearch::IRankModel* costModel = HCSearch::Learning::learnC(XTrain, YTrain, XValidation, YValidation, 
	heuristicModel, timeBound, searchSpace, searchProcedure, HCSearch::SVM_RANK, 1);

	// run HC search inference on the first test example for demo
	HCSearch::ISearchProcedure::SearchMetadata searchMetadata; // no meta data needed for this demo
	HCSearch::Inference::runHCSearch(XTest[0], timeBound, searchSpace, searchProcedure, heuristicModel, costModel, searchMetadata);

	// save models for later use
	HCSearch::Model::saveModel(heuristicModel, "path/to/heuristic/model.txt", HCSearch::SVM_RANK);
	HCSearch::Model::saveModel(costModel, "path/to/cost/model.txt", HCSearch::SVM_RANK);

	// clean up
	delete searchSpace;
	delete searchProcedure;
	delete heuristicModel;
	delete costModel;
	HCSearch::Dataset::unloadDataset(XTrain, YTrain, XValidation, YValidation, XTest, YTest);
}
