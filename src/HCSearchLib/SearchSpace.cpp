#include "SearchSpace.hpp"
#include "MyLogger.hpp"

using namespace std;

namespace HCSearch
{
	/**************** Search Space ****************/

	SearchSpace::SearchSpace()
	{
		this->heuristicFeatureFunction = NULL;
		this->costFeatureFunction = NULL;
		this->initialPredictionFunction = NULL;
		this->successorFunction = NULL;
		this->lossFunction = NULL;
	}

	SearchSpace::SearchSpace(IFeatureFunction* heuristicFeatureFunction, IFeatureFunction* costFeatureFunction,
		IInitialPredictionFunction* initialPredictionFunction, ISuccessorFunction* successorFunction,
		ILossFunction* lossFunction)
	{
		this->heuristicFeatureFunction = heuristicFeatureFunction;
		this->costFeatureFunction = costFeatureFunction;
		this->initialPredictionFunction = initialPredictionFunction;
		this->successorFunction = successorFunction;
		this->lossFunction = lossFunction;
	}

	SearchSpace::~SearchSpace()
	{
		delete this->heuristicFeatureFunction;
		delete this->costFeatureFunction;
		delete this->initialPredictionFunction;
		delete this->successorFunction;
		delete this->lossFunction;
	}

	RankFeatures SearchSpace::computeHeuristicFeatures(ImgFeatures& X, ImgLabeling& Y)
	{
		if (this->heuristicFeatureFunction == NULL)
		{
			LOG(ERROR) << "heuristic feature function is null";
			abort();
		}

		return this->heuristicFeatureFunction->computeFeatures(X, Y);
	}

	RankFeatures SearchSpace::computeCostFeatures(ImgFeatures& X, ImgLabeling& Y)
	{
		if (this->costFeatureFunction == NULL)
		{
			LOG(ERROR) << "cost feature function is null";
			abort();
		}

		return this->costFeatureFunction->computeFeatures(X, Y);
	}

	ImgLabeling SearchSpace::getInitialPrediction(ImgFeatures& X)
	{
		if (this->initialPredictionFunction == NULL)
		{
			LOG(ERROR) << "initial pred feature function is null";
			abort();
		}

		return this->initialPredictionFunction->getInitialPrediction(X);
	}

	vector< ImgCandidate > SearchSpace::generateSuccessors(ImgFeatures& X, ImgLabeling& YPred)
	{
		if (this->successorFunction == NULL)
		{
			LOG(ERROR) << "successor function is null";
			abort();
		}

		return this->successorFunction->generateSuccessors(X, YPred);
	}

	double SearchSpace::computeLoss(ImgLabeling& YPred, const ImgLabeling& YTruth)
	{
		if (this->lossFunction == NULL)
		{
			LOG(ERROR) << "loss function is null";
			abort();
		}

		return this->lossFunction->computeLoss(YPred, YTruth);
	}
}