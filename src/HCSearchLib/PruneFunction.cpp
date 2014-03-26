#include "PruneFunction.hpp"
#include "MyLogger.hpp"

namespace HCSearch
{
	/**************** Prune Abstract Class ****************/

	IFeatureFunction* IPruneFunction::getFeatureFunction()
	{
		return this->featureFunction;
	}

	/**************** No Prune ****************/

	NoPrune::NoPrune()
	{
	}

	NoPrune::~NoPrune()
	{
	}
		
	vector< ImgCandidate > NoPrune::pruneSuccessors(ImgFeatures& X, ImgLabeling& Y, vector< ImgCandidate >& YCandidates)
	{
		return YCandidates;
	}

	/**************** Classifier Prune ****************/

	ClassifierPrune::ClassifierPrune()
	{
		this->featureFunction = NULL;
		this->classifier = NULL;
	}

	ClassifierPrune::ClassifierPrune(IFeatureFunction* featureFunction)
	{
		this->featureFunction = featureFunction;
		this->classifier = NULL;
	}

	ClassifierPrune::~ClassifierPrune()
	{
	}
	
	vector< ImgCandidate > ClassifierPrune::pruneSuccessors(ImgFeatures& X, ImgLabeling& Y, vector< ImgCandidate >& YCandidates)
	{
		vector< ImgCandidate > YPrunedCandidates;

		// get pruning features of candidates
		vector<ClassifierFeatures> featuresList;
		for (vector<ImgCandidate>::iterator it = YCandidates.begin(); it != YCandidates.end(); ++it)
		{
			ImgCandidate YCand = *it;
			set<int> action = YCand.action;
			ClassifierFeatures features = this->featureFunction->computeFeatures(X, YCand.labeling, action);
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

	void ClassifierPrune::setClassifier(IClassifierModel* classifier)
	{
		this->classifier = classifier;
	}

	/**************** Oracle Prune ****************/

	OraclePrune::OraclePrune()
	{
		this->featureFunction = NULL;
		this->lossFunction = NULL;
		this->YTruth = NULL;
	}

	OraclePrune::OraclePrune(ILossFunction* lossFunction)
	{
		this->featureFunction = NULL;
		this->lossFunction = lossFunction;
		this->YTruth = NULL;
	}

	OraclePrune::~OraclePrune()
	{
	}
	
	vector< ImgCandidate > OraclePrune::pruneSuccessors(ImgFeatures& X, ImgLabeling& Y, vector< ImgCandidate >& YCandidates)
	{
		if (YTruth == NULL)
		{
			LOG(ERROR) << "YTruth is null for oracle pruning and in the inside function";
			abort();
		}

		vector< ImgCandidate > YPrunedCandidates;

		double prevLoss = this->lossFunction->computeLoss(Y, *YTruth);

		// remove bad candidates (keep good candidates)
		vector<ClassifierFeatures> featuresList;
		for (vector<ImgCandidate>::iterator it = YCandidates.begin(); it != YCandidates.end(); ++it)
		{
			ImgCandidate YCand = *it;
			double candLoss = this->lossFunction->computeLoss(YCand.labeling, *YTruth);
			
			if (candLoss < prevLoss)
				YPrunedCandidates.push_back(YCand);
		}

		LOG() << "num of successors before pruning=" << YCandidates.size() << endl;
		LOG() << "\tnum of successors after pruning=" << YPrunedCandidates.size() << endl;

		return YPrunedCandidates;
	}

	vector< ImgCandidate > OraclePrune::pruneSuccessors(ImgFeatures& X, ImgLabeling& Y, vector< ImgCandidate >& YCandidates, ImgLabeling* YTruth)
	{
		if (YTruth == NULL)
		{
			LOG(ERROR) << "YTruth is null for oracle pruning";
			abort();
		}

		if (this->YTruth != NULL)
		{
			LOG(WARNING) << "YTruth was not null when begin to oracle prune";
		}

		this->YTruth = YTruth;
		vector<ImgCandidate> pruned = pruneSuccessors(X, Y, YCandidates);
		this->YTruth = NULL;
		return pruned;
	}

	ILossFunction* OraclePrune::getLossFunction()
	{
		return this->lossFunction;
	}
}