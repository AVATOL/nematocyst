#include "PruneFunction.hpp"

namespace HCSearch
{
	/**************** No Prune ****************/

	NoPrune::NoPrune()
	{
	}

	NoPrune::~NoPrune()
	{
	}
		
	vector< ImgCandidate > NoPrune::pruneSuccessors(ImgFeatures& X, vector< ImgCandidate >& YCandidates)
	{
		return YCandidates;
	}

	/**************** Domain Knowledge ****************/

	ClassifierPrune::ClassifierPrune()
	{
		this->featureFunction = NULL;
	}

	ClassifierPrune::ClassifierPrune(IFeatureFunction* featureFunction)
	{
		this->featureFunction = featureFunction;
	}

	ClassifierPrune::~ClassifierPrune()
	{
	}
	
	vector< ImgCandidate > ClassifierPrune::pruneSuccessors(ImgFeatures& X, vector< ImgCandidate >& YCandidates)
	{
		//TODO

		vector< ImgCandidate > YPrunedCandidates = YCandidates;

		for (vector<ImgCandidate>::iterator it = YCandidates.begin(); it != YCandidates.end(); ++it)
		{
			ImgCandidate YCand = *it;
			RankFeatures features = this->featureFunction->computeFeatures(X, YCand.labeling);
			// TODO: write features to file
		}

		//TODO: run SVM classifier
		
		// TODO: read SVM results and remove bad from the set

		return YPrunedCandidates;
	}
}