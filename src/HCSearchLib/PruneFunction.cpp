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

	/**************** Ranker Prune ****************/

	const double RankerPrune::DEFAULT_PRUNE_FRACTION = 0.5;

	RankerPrune::RankerPrune()
	{
		this->pruneFraction = DEFAULT_PRUNE_FRACTION;
		this->featureFunction = NULL;
		this->ranker = NULL;
	}

	RankerPrune::RankerPrune(double pruneFraction)
	{
		this->pruneFraction = pruneFraction;
		this->featureFunction = NULL;
		this->ranker = NULL;
	}

	RankerPrune::RankerPrune(IFeatureFunction* featureFunction)
	{
		this->pruneFraction = DEFAULT_PRUNE_FRACTION;
		this->featureFunction = featureFunction;
		this->ranker = NULL;
	}

	RankerPrune::RankerPrune(double pruneFraction, IFeatureFunction* featureFunction)
	{
		this->pruneFraction = pruneFraction;
		this->featureFunction = featureFunction;
		this->ranker = NULL;
	}

	RankerPrune::~RankerPrune()
	{
	}
	
	vector< ImgCandidate > RankerPrune::pruneSuccessors(ImgFeatures& X, ImgLabeling& Y, vector< ImgCandidate >& YCandidates)
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

		// run ranker
		vector<double> ranks = this->ranker->rank(featuresList);

		// sort by rank
		RankNodePQ rankPQ;
		const int numOriginalCandidates = YCandidates.size();
		const int numNewCandidates = static_cast<int>((1-pruneFraction)*numOriginalCandidates);
		for (int i = 0; i < numOriginalCandidates; i++)
		{
			RankPruneNode rankNode;
			rankNode.rank = ranks[i];
			rankNode.YCandidate = YCandidates[i];
			rankPQ.push(rankNode);
		}

		// remove bad candidates (keep good candidates)
		for (int i = 0; i < numNewCandidates; i++)
		{
			if (rankPQ.empty())
				break;

			RankPruneNode rankNode = rankPQ.top();
			rankPQ.pop();
			YPrunedCandidates.push_back(rankNode.YCandidate);
		}

		LOG() << "num of successors before pruning=" << numOriginalCandidates << endl;
		LOG() << "\tnum of successors after pruning=" << YPrunedCandidates.size() << endl;

		return YPrunedCandidates;
	}

	IRankModel* RankerPrune::getRanker()
	{
		return this->ranker;
	}

	void RankerPrune::setRanker(IRankModel* ranker)
	{
		this->ranker = ranker;
	}

	bool CompareRankPruneNodes::operator() (RankPruneNode& lhs, RankPruneNode& rhs) const
	{
		return lhs.rank > rhs.rank;
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