#ifndef PRUNEFUNCTION_HPP
#define PRUNEFUNCTION_HPP

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
	public:
		virtual ~IPruneFunction() {}
		
		/*!
		 * @brief Prune successors.
		 */
		virtual vector< ImgCandidate > pruneSuccessors(ImgFeatures& X, vector< ImgCandidate >& YCandidates)=0;
	};

	/**************** Prune Functions ****************/

	/*!
	 * @brief Domain knowledge pruning function.
	 *
	 * For each node, flip its label to all possible classes.
	 */
	class DomainKnowledgePrune : public IPruneFunction
	{
	public:
		DomainKnowledgePrune();
		~DomainKnowledgePrune();
		
		virtual vector< ImgCandidate > pruneSuccessors(ImgFeatures& X, vector< ImgCandidate >& YCandidates);
	};
}

#endif