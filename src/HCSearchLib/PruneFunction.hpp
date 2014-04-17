#ifndef PRUNEFUNCTION_HPP
#define PRUNEFUNCTION_HPP

#include "FeatureFunction.hpp"
#include "DataStructures.hpp"
#include "MyGraphAlgorithms.hpp"
#include "LossFunction.hpp"

namespace HCSearch
{
	/**************** Search Space Function Abstract Definitions ****************/

	/*!
	 * @brief Pruning function interface
	 */
	class IPruneFunction
	{
	protected:
		IFeatureFunction* featureFunction; //!< Pruning feature function

	public:
		virtual ~IPruneFunction() {}
		
		/*!
		 * @brief Prune successors.
		 */
		virtual vector< ImgCandidate > pruneSuccessors(ImgFeatures& X, ImgLabeling& Y, vector< ImgCandidate >& YCandidates, ImgLabeling* YTruth, ILossFunction* lossFunc)=0;

		IFeatureFunction* getFeatureFunction();
	};

	/**************** Prune Functions ****************/

	/*!
	 * @brief No pruning.
	 */
	class NoPrune : public IPruneFunction
	{
	public:
		NoPrune();
		~NoPrune();
		
		virtual vector< ImgCandidate > pruneSuccessors(ImgFeatures& X, ImgLabeling& Y, vector< ImgCandidate >& YCandidates, ImgLabeling* YTruth, ILossFunction* lossFunc);
	};

	/*!
	 * @brief Pruning function using classifer for good/bad actions.
	 */
	class ClassifierPrune : public IPruneFunction
	{
		IClassifierModel* classifier;

	public:
		ClassifierPrune();
		ClassifierPrune(IFeatureFunction* featureFunction);
		~ClassifierPrune();
		
		virtual vector< ImgCandidate > pruneSuccessors(ImgFeatures& X, ImgLabeling& Y, vector< ImgCandidate >& YCandidates, ImgLabeling* YTruth, ILossFunction* lossFunc);
		
		IClassifierModel* getClassifier();
		void setClassifier(IClassifierModel* classifier);
	};

	struct RankPruneNode
	{
		double rank;
		ImgCandidate YCandidate;
	};

	class CompareRankPruneNodes
	{
	public:
		bool operator() (RankPruneNode& lhs, RankPruneNode& rhs) const;
	};

	typedef priority_queue<RankPruneNode, vector<RankPruneNode>, CompareRankPruneNodes> RankNodePQ;

	/*!
	 * @brief Pruning function using ranker for ranking good/bad actions.
	 */
	class RankerPrune : public IPruneFunction
	{
		static const double DEFAULT_PRUNE_FRACTION;

		IRankModel* ranker;
		double pruneFraction; //!< 0 = no pruning, 1 = prune everything

	public:
		RankerPrune();
		RankerPrune(double pruneFraction);
		RankerPrune(IFeatureFunction* featureFunction);
		RankerPrune(double pruneFraction, IFeatureFunction* featureFunction);
		~RankerPrune();
		
		virtual vector< ImgCandidate > pruneSuccessors(ImgFeatures& X, ImgLabeling& Y, vector< ImgCandidate >& YCandidates, ImgLabeling* YTruth, ILossFunction* lossFunc);
		
		IRankModel* getRanker();
		void setRanker(IRankModel* ranker);
	};

	/*!
	 * @brief Pruning function using ranker for ranking good/bad actions.
	 */
	class SimulatedRankerPrune : public IPruneFunction
	{
		static const double DEFAULT_PRUNE_FRACTION;
		
		double pruneFraction; //!< 0 = no pruning, 1 = prune everything

	public:
		SimulatedRankerPrune();
		SimulatedRankerPrune(double pruneFraction);
		SimulatedRankerPrune(IFeatureFunction* featureFunction);
		SimulatedRankerPrune(double pruneFraction, IFeatureFunction* featureFunction);
		~SimulatedRankerPrune();
		
		virtual vector< ImgCandidate > pruneSuccessors(ImgFeatures& X, ImgLabeling& Y, vector< ImgCandidate >& YCandidates, ImgLabeling* YTruth, ILossFunction* lossFunc);
	};

	/*!
	 * @brief Oracle pruning function.
	 */
	class OraclePrune : public IPruneFunction
	{
		static const double DEFAULT_PRUNE_FRACTION;

	protected:
		ILossFunction* lossFunction;
		ImgLabeling* YTruth;

		double badPruneFraction; //!< 0 = no pruning, 1 = prune everything

	public:
		OraclePrune();
		OraclePrune(double badPruneFraction);
		OraclePrune(ILossFunction* lossFunction);
		OraclePrune(ILossFunction* lossFunction, double badPruneFraction);
		~OraclePrune();
		
		virtual vector< ImgCandidate > pruneSuccessors(ImgFeatures& X, ImgLabeling& Y, vector< ImgCandidate >& YCandidates, ImgLabeling* YTruth, ILossFunction* lossFunc);

		vector< ImgCandidate > pruneSuccessors(ImgFeatures& X, ImgLabeling& Y, vector< ImgCandidate >& YCandidates, ImgLabeling* YTruth);

		ILossFunction* getLossFunction();
	};
}

#endif