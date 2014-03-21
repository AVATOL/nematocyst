#include "PruneFunction.hpp"
#include "MyLogger.hpp"

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

	/**************** Classifier Prune ****************/

	ClassifierPrune::ClassifierPrune()
	{
		this->featureFunction = NULL;
		this->classifier = NULL;
	}

	ClassifierPrune::ClassifierPrune(IFeatureFunction* featureFunction, IClassifierModel* classifier)
	{
		this->featureFunction = featureFunction;
		this->classifier = classifier;
	}

	ClassifierPrune::~ClassifierPrune()
	{
	}
	
	vector< ImgCandidate > ClassifierPrune::pruneSuccessors(ImgFeatures& X, vector< ImgCandidate >& YCandidates)
	{
		vector< ImgCandidate > YPrunedCandidates;

		// get pruning features of candidates
		vector<ClassifierFeatures> featuresList;
		for (vector<ImgCandidate>::iterator it = YCandidates.begin(); it != YCandidates.end(); ++it)
		{
			ImgCandidate YCand = *it;
			ClassifierFeatures features = this->featureFunction->computeFeatures(X, YCand.labeling);
			featuresList.push_back(features);
		}

		// run classifier
		vector<int> classes = this->classifier->classify(featuresList);
		
		// remove bad candidates (keep good candidates)
		const int numOriginalCandidates = YCandidates.size();
		for (int i = 0; i < numOriginalCandidates; i++)
		{
			if (classes[i] > 0) // positive class is assumed positive
			{
				YPrunedCandidates.push_back(YCandidates[i]);
			}
		}

		LOG() << "num of successors before pruning=" << numOriginalCandidates << endl;
		LOG() << "\tnum of successors after pruning=" << YPrunedCandidates.size() << endl;

		return YPrunedCandidates;
	}

	IClassifierModel* ClassifierPrune::getClassifier()
	{
		return this->classifier;
	}
}