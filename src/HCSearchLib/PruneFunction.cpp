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

	DomainKnowledgePrune::DomainKnowledgePrune()
	{
	}

	DomainKnowledgePrune::~DomainKnowledgePrune()
	{
	}
		
	vector< ImgCandidate > DomainKnowledgePrune::pruneSuccessors(ImgFeatures& X, vector< ImgCandidate >& YCandidates)
	{
		//TODO
		return YCandidates;
	}
}