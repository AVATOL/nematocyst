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
		this->pruneFunction = NULL;
		this->lossFunction = NULL;
	}

	SearchSpace::SearchSpace(IFeatureFunction* heuristicFeatureFunction, IFeatureFunction* costFeatureFunction,
		IInitialPredictionFunction* initialPredictionFunction, ISuccessorFunction* successorFunction,
		IPruneFunction* pruneFunction, ILossFunction* lossFunction)
	{
		this->heuristicFeatureFunction = heuristicFeatureFunction;
		this->costFeatureFunction = costFeatureFunction;
		this->initialPredictionFunction = initialPredictionFunction;
		this->successorFunction = successorFunction;
		this->pruneFunction = pruneFunction;
		this->lossFunction = lossFunction;
	}

	SearchSpace::~SearchSpace()
	{
		delete this->pruneFunction;
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

		set<int> action;
		return this->heuristicFeatureFunction->computeFeatures(X, Y, action);
	}

	RankFeatures SearchSpace::computeCostFeatures(ImgFeatures& X, ImgLabeling& Y)
	{
		if (this->costFeatureFunction == NULL)
		{
			LOG(ERROR) << "cost feature function is null";
			abort();
		}

		set<int> action;
		return this->costFeatureFunction->computeFeatures(X, Y, action);
	}

	RankFeatures SearchSpace::computePruneFeatures(ImgFeatures& X, ImgLabeling& Y, set<int> action)
	{
		if (this->pruneFunction->getFeatureFunction() == NULL)
		{
			LOG(ERROR) << "prune feature function is null";
			abort();
		}

		return this->pruneFunction->getFeatureFunction()->computeFeatures(X, Y, action);
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

	vector< ImgCandidate > SearchSpace::generateSuccessors(ImgFeatures& X, ImgLabeling& YPred, int timeStep, int timeBound)
	{
		if (this->successorFunction == NULL)
		{
			LOG(ERROR) << "successor function is null";
			abort();
		}

		return this->successorFunction->generateSuccessors(X, YPred, timeStep, timeBound);
	}

	vector< ImgCandidate > SearchSpace::pruneSuccessors(ImgFeatures& X, ImgLabeling& YPred, vector< ImgCandidate >& YCandidates, ImgLabeling* YTruth, ILossFunction* lossFunc)
	{
		if (this->pruneFunction == NULL)
		{
			LOG(DEBUG) << "prune function is null; no pruning done";
			return YCandidates;
		}
		else
		{
			return this->pruneFunction->pruneSuccessors(X, YPred, YCandidates, YTruth, lossFunc);
		}
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

	IPruneFunction* SearchSpace::getPruneFunction()
	{
		return this->pruneFunction;
	}

	IInitialPredictionFunction* SearchSpace::getInitialPredictionFunction()
	{
		return this->initialPredictionFunction;
	}

	ILossFunction* SearchSpace::getLossFunction()
	{
		return this->lossFunction;
	}
}