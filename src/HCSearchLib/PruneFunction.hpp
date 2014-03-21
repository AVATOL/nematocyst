#ifndef PRUNEFUNCTION_HPP
#define PRUNEFUNCTION_HPP

#include "FeatureFunction.hpp"
#include "DataStructures.hpp"
#include "MyGraphAlgorithms.hpp"

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
		virtual vector< ImgCandidate > pruneSuccessors(ImgFeatures& X, vector< ImgCandidate >& YCandidates)=0;
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
		
		virtual vector< ImgCandidate > pruneSuccessors(ImgFeatures& X, vector< ImgCandidate >& YCandidates);
	};

	/*!
	 * @brief Pruning function using classifer for good/bad actions.
	 */
	class ClassifierPrune : public IPruneFunction
	{
	public:
		ClassifierPrune();
		ClassifierPrune(IFeatureFunction* featureFunction);
		~ClassifierPrune();
		
		virtual vector< ImgCandidate > pruneSuccessors(ImgFeatures& X, vector< ImgCandidate >& YCandidates);
	};
}

#endif