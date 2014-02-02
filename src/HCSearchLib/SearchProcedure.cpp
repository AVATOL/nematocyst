#include <iostream>
#include "SearchProcedure.hpp"
#include "Globals.hpp"

using namespace std;

namespace HCSearch
{
	/**************** Save Prediction ****************/

	void SavePrediction::saveLabels(ImgLabeling& YPred, string fileName)
	{
		ofstream fh(fileName.c_str());
		if (fh.is_open())
		{
			const int numNodes = YPred.getNumNodes();
			fh << YPred.getLabel(0);
			for (int node = 1; node < numNodes; node++)
			{
				fh << endl << YPred.getLabel(node);
			}
			fh.close();
		}
		else
		{
			cerr << "[Error] cannot open file to write nodes!" << endl;
		}
	}

	void SavePrediction::saveCuts(ImgLabeling& YPred, string fileName)
	{
		if (!YPred.stochasticCutsAvailable)
		{
			cerr << "[Error] no cuts available to write!" << endl;
			return;
		}

		map< int, set<int> > edgeNeighbors = YPred.stochasticCuts;

		// write to file
		ofstream fh(fileName.c_str());
		if (fh.is_open())
		{
			for (map< int, set<int> >::iterator it = edgeNeighbors.begin(); it != edgeNeighbors.end(); ++it)
			{
				int node1 = it->first;
				set<int> neighbors = it->second;
				for (set<int>::iterator it2 = neighbors.begin(); it2 != neighbors.end(); ++it2)
				{
					int node2 = *it2;

					fh << node1+1 << " " << node2+1 << " 1" << endl;
				}
			}

			fh.close();
		}
		else
		{
			cerr << "[Error] cannot open file to write edges!" << endl;
		}
	}

	/**************** Search Procedure ****************/

	ImgLabeling ISearchProcedure::llSearch(ImgFeatures& X, ImgLabeling* YTruth, int timeBound, 
		SearchSpace* searchSpace, SearchMetadata searchMetadata)
	{
		return searchProcedure(LL, X, YTruth, timeBound, 
			searchSpace, NULL, NULL, NULL, searchMetadata);
	}

	ImgLabeling ISearchProcedure::hlSearch(ImgFeatures& X, ImgLabeling* YTruth, int timeBound, 
		SearchSpace* searchSpace, IRankModel* heuristicModel, SearchMetadata searchMetadata)
	{
		return searchProcedure(HL, X, YTruth, timeBound, 
			searchSpace, heuristicModel, NULL, NULL, searchMetadata);
	}

	ImgLabeling ISearchProcedure::lcSearch(ImgFeatures& X, ImgLabeling* YTruth, int timeBound, 
		SearchSpace* searchSpace, IRankModel* costModel, SearchMetadata searchMetadata)
	{
		return searchProcedure(LC, X, YTruth, timeBound, 
			searchSpace, NULL, costModel, NULL, searchMetadata);
	}

	ImgLabeling ISearchProcedure::hcSearch(ImgFeatures& X, int timeBound, SearchSpace* searchSpace, 
		IRankModel* heuristicModel, IRankModel* costModel, SearchMetadata searchMetadata)
	{
		return searchProcedure(HC, X, NULL, timeBound, 
			searchSpace, heuristicModel, costModel, NULL, searchMetadata);
	}

	void ISearchProcedure::learnH(ImgFeatures& X, ImgLabeling* YTruth, int timeBound, SearchSpace* searchSpace, 
		IRankModel* learningModel, SearchMetadata searchMetadata)
	{
		searchProcedure(LEARN_H, X, YTruth, timeBound, 
			searchSpace, NULL, NULL, learningModel, searchMetadata);
	}

	void ISearchProcedure::learnC(ImgFeatures& X, ImgLabeling* YTruth, int timeBound, SearchSpace* searchSpace, 
		IRankModel* heuristicModel, IRankModel* learningModel, SearchMetadata searchMetadata)
	{
		searchProcedure(LEARN_H, X, YTruth, timeBound, 
			searchSpace, heuristicModel, NULL, learningModel, searchMetadata);
	}

	void ISearchProcedure::learnCWithOracleH(ImgFeatures& X, ImgLabeling* YTruth, int timeBound, SearchSpace* searchSpace, 
		IRankModel* learningModel, SearchMetadata searchMetadata)
	{
		searchProcedure(LEARN_H, X, YTruth, timeBound, 
			searchSpace, NULL, NULL, learningModel, searchMetadata);
	}

	void ISearchProcedure::saveAnyTimePrediction(ImgLabeling YPred, int timeBound, SearchMetadata searchMetadata, SearchType searchType)
	{
		if (searchMetadata.saveAnytimePredictions)
		{
			stringstream ssPredictNodes;
			ssPredictNodes << Global::settings->paths->OUTPUT_RESULTS_DIR << "nodes" 
				<< "_" << SearchTypeStrings[searchType] 
				<< "_" << DatasetTypeStrings[searchMetadata.setType] 
				<< "_time" << timeBound 
					<< "_fold" << searchMetadata.iter 
					<< "_" << searchMetadata.exampleName << ".txt";
			SavePrediction::saveLabels(YPred, ssPredictNodes.str());

			if (!YPred.stochasticCutsAvailable)
			{
				YPred.stochasticCuts = YPred.graph.adjList;
				YPred.stochasticCutsAvailable = true;
			}

			stringstream ssPredictEdges;
			ssPredictEdges << Global::settings->paths->OUTPUT_RESULTS_DIR << "edges" 
			<< "_" << SearchTypeStrings[searchType] 
			<< "_" << DatasetTypeStrings[searchMetadata.setType] 
			<< "_time" << timeBound 
				<< "_fold" << searchMetadata.iter 
				<< "_" << searchMetadata.exampleName << ".txt";
			SavePrediction::saveCuts(YPred, ssPredictEdges.str());
		}
	}

	void ISearchProcedure::trainHeuristicRanker(IRankModel* ranker, SearchNodePQ& candidateSet, vector< ISearchNode* > successorSet)
	{
		if (ranker->rankerType() == SVM_RANK)
		{
			vector< RankFeatures > bestFeatures;
			vector< RankFeatures > worstFeatures;

			// best states
			for (vector< ISearchNode* >::iterator it = successorSet.begin(); it != successorSet.end(); ++it)
			{
				ISearchNode* state = *it;
				bestFeatures.push_back(state->getHeuristicFeatures());
			}

			// worst states
			SearchNodePQ tempSet;
			while (!candidateSet.empty())
			{
				ISearchNode* state = candidateSet.top();
				worstFeatures.push_back(state->getHeuristicFeatures());
				tempSet.push(state);
				candidateSet.pop();
			}
			candidateSet = tempSet;

			// train
			SVMRankModel* svmRankModel = dynamic_cast<SVMRankModel*>(ranker);
			svmRankModel->addTrainingExamples(bestFeatures, worstFeatures);
		}
		else if (ranker->rankerType() == ONLINE_RANK)
		{
			//TODO
		}
		else
		{
			cerr << "[Error] unknown ranker type" << endl;
			abort();
		}
	}

	void ISearchProcedure::trainCostRanker(IRankModel* ranker, SearchNodePQ& costSet)
	{
		if (ranker->rankerType() == SVM_RANK)
		{
			vector< RankFeatures > bestFeatures;
			vector< RankFeatures > worstFeatures;

			// get best states
			double bestLoss = costSet.top()->getCost();
			while (!costSet.empty() && costSet.top()->getCost() <= bestLoss)
			{
				ISearchNode* state = costSet.top();
				bestFeatures.push_back(state->getCostFeatures());
				costSet.pop();
				delete state;
			}

			// get worst states
			while (!costSet.empty())
			{
				ISearchNode* state = costSet.top();
				bestFeatures.push_back(state->getCostFeatures());
				costSet.pop();
				delete state;
			}

			// train
			SVMRankModel* svmRankModel = dynamic_cast<SVMRankModel*>(ranker);
			svmRankModel->addTrainingExamples(bestFeatures, worstFeatures);

		}
		else if (ranker->rankerType() == ONLINE_RANK)
		{
			//TODO
		}
		else
		{
			cerr << "[Error] unknown ranker type" << endl;
			abort();
		}
	}

	ImgLabeling IBasicSearchProcedure::searchProcedure(SearchType searchType, ImgFeatures& X, ImgLabeling* YTruth, 
	int timeBound, SearchSpace* searchSpace, IRankModel* heuristicModel, IRankModel* costModel,
	IRankModel* learningModel, SearchMetadata searchMetadata)
	{
		CompareSearchNode compareCost(COST);
		CompareSearchNode compareHeuristic(HEURISTIC);

		// set up priority queues
		// maintain open set for search
		// maintain cost set for returning best cost in the end
		SearchNodePQ costSet(compareCost);
		SearchNodePQ openSet(compareHeuristic);

		// push initial state into queue
		ISearchNode* root = NULL;
		switch (searchType)
		{
			case LL:
				root = new LLSearchNode(&X, YTruth, searchSpace);
				break;
			case HL:
				root = new HLSearchNode(&X, YTruth, searchSpace, heuristicModel);
				break;
			case LC:
				root = new LCSearchNode(&X, YTruth, searchSpace, costModel);
				break;
			case HC:
				root = new HCSearchNode(&X, searchSpace, heuristicModel, costModel);
				break;
			case LEARN_H:
				root = new LearnHSearchNode(&X, YTruth, searchSpace);
				break;
			case LEARN_C:
				root = new LearnCSearchNode(&X, YTruth, searchSpace, heuristicModel);
				break;
			case LEARN_C_ORACLE_H:
				root = new LearnCOracleHSearchNode(&X, YTruth, searchSpace);
				break;
			default:
				cerr << "searchType constant is invalid." << endl;
		}
		openSet.push(root);
		costSet.push(root);

		// while the open set is not empty and the time step is less than the time bound,
		// perform search...
		int timeStep = 0;
		while (!openSet.empty() && timeStep < timeBound)
		{
			cout << endl << "Running " << SearchTypeStrings[searchType] << " search with time step " << timeStep+1 << "/" << timeBound << "..." << endl;

			// save current best if anytime prediction enabled
			saveAnyTimePrediction(costSet.top()->getY(), timeStep, searchMetadata, searchType);

			/***** pick some subset of elements from the open set *****/

			vector< ISearchNode* > subsetOpenSet = selectSubsetOpenSet(openSet);

			/***** expand these elements *****/

			SearchNodePQ candidateSet = expandElements(subsetOpenSet, openSet, costSet);

			/***** choose successors and put them into the open set *****/
			/***** put these expanded elements into the cost set *****/

			vector< ISearchNode* > successorSet = chooseSuccessors(candidateSet, openSet, costSet);

			// NOW: successorSet = best nodes
			// NOW: candidateSet = worst nodes

			/***** use best/worst candidates as training examples for heuristic learning (if applicable) *****/

			if (searchType == LEARN_H)
				trainHeuristicRanker(heuristicModel, candidateSet, successorSet);

			/***** add remaining "worst states" to cost set *****/

			while (!candidateSet.empty())
			{
				ISearchNode* node = candidateSet.top();
				candidateSet.pop();

				costSet.push(node);
			}

			/***** increment time step *****/
			timeStep++;
		}

		/***** search is done, return the lowest cost search node *****/

		if (costSet.empty())
		{
			cerr << "[Error] the cost set is empty, which is not possible!" << endl;
			abort();
		}

		// Get lowest cost node
		ISearchNode* lowestCost = costSet.top();
		ImgLabeling prediction = lowestCost->getY();
		cout << endl << "Finished search. Cost=" << lowestCost->getCost() << endl << endl;

		// use best/worst cost set candidates as training examples for cost learning (if applicable)
		if (searchType == LEARN_C || searchType == LEARN_C_ORACLE_H)
			trainCostRanker(costModel, costSet);

		// clean up cost set
		deleteQueueElements(costSet);

		return prediction;
	}

	bool IBasicSearchProcedure::isDuplicate(ISearchNode* state, SearchNodePQ& pq)
	{
		int size = pq.size();
		bool isDuplicate = false;

		SearchNodePQ temp;

		for (int i = 0; i < size; i++)
		{
			ISearchNode* current = pq.top();
			pq.pop();

			if (!isDuplicate && current->getY().graph.nodesData == state->getY().graph.nodesData)
			{
				isDuplicate = true;
			}

			temp.push(current);
		}

		// reset priority queue passed as argument
		pq = temp;

		return isDuplicate;
	}

	void IBasicSearchProcedure::deleteQueueElements(SearchNodePQ& queue)
	{
		while (!queue.empty())
		{
			ISearchNode* state = queue.top();
			queue.pop();
			delete state;
		}
	}

	/**************** Breadth-First Beam Search Procedure ****************/

	BreadthFirstBeamSearchProcedure::BreadthFirstBeamSearchProcedure()
	{
		this->beamSize = this->DEFAULT_BEAM_SIZE;
	}

	BreadthFirstBeamSearchProcedure::BreadthFirstBeamSearchProcedure(int beamSize)
	{
		this->beamSize = beamSize;
	}

	BreadthFirstBeamSearchProcedure::~BreadthFirstBeamSearchProcedure()
	{
	}

	vector< ISearchProcedure::ISearchNode* > BreadthFirstBeamSearchProcedure::selectSubsetOpenSet(SearchNodePQ& openSet)
	{
		vector< ISearchNode* > subsetOpenSet;
		
		// just add everything (should only be at most B states)
		while (!openSet.empty())
		{
			ISearchNode* current = openSet.top();
			openSet.pop();
			subsetOpenSet.push_back(current);
		}

		return subsetOpenSet;
	}

	ISearchProcedure::SearchNodePQ BreadthFirstBeamSearchProcedure::expandElements(vector< ISearchNode* > subsetOpenSet, SearchNodePQ& openSet, SearchNodePQ& costSet)
	{
		SearchNodePQ candidateSet;
		
		// expand each element
		for (vector< ISearchNode* >::iterator it = subsetOpenSet.begin(); it != subsetOpenSet.end(); ++it)
		{
			ISearchNode* current = *it;
			cout << "Expansion Node: Heuristic=" << current->getHeuristic() << ", Cost=" << current->getCost() << endl;

			vector< ISearchNode* > expansionSet = current->generateSuccessorNodes();

			// only accept expanded element if not a duplicate state
			for (vector< ISearchNode* >::iterator it = expansionSet.begin(); it != expansionSet.end(); ++it)
			{
				ISearchNode* state = *it;
				if (!isDuplicate(state, candidateSet) && !isDuplicate(state, costSet))
				{
					candidateSet.push(state);
				}
			}
		}

		return candidateSet;
	}

	vector< ISearchProcedure::ISearchNode* > BreadthFirstBeamSearchProcedure::chooseSuccessors(SearchNodePQ& candidateSet, SearchNodePQ& openSet, SearchNodePQ& costSet)
	{
		vector< ISearchNode* > bestSet;

		// pick the top B best states
		for (int i = 0; i < this->beamSize; i++)
		{
			if (candidateSet.empty())
				break;

			ISearchNode* state = candidateSet.top();
			candidateSet.pop();
			bestSet.push_back(state);
			openSet.push(state);
			costSet.push(state);
		}

		// side effect: candidateSet will only have the worst states now

		return bestSet;
	}

	/**************** Best-First Beam Search Procedure ****************/

	BestFirstBeamSearchProcedure::BestFirstBeamSearchProcedure()
	{
		this->beamSize = this->DEFAULT_BEAM_SIZE;
	}

	BestFirstBeamSearchProcedure::BestFirstBeamSearchProcedure(int beamSize)
	{
		this->beamSize = beamSize;
	}

	BestFirstBeamSearchProcedure::~BestFirstBeamSearchProcedure()
	{
	}

	vector< ISearchProcedure::ISearchNode* > BestFirstBeamSearchProcedure::selectSubsetOpenSet(SearchNodePQ& openSet)
	{
		vector< ISearchNode* > subsetOpenSet;
		
		// only add top element
		if (!openSet.empty())
		{
			ISearchNode* current = openSet.top();
			openSet.pop();
			subsetOpenSet.push_back(current);
		}

		return subsetOpenSet;
	}

	ISearchProcedure::SearchNodePQ BestFirstBeamSearchProcedure::expandElements(vector< ISearchNode* > subsetOpenSet, SearchNodePQ& openSet, SearchNodePQ& costSet)
	{
		SearchNodePQ candidateSet;
		
		// expand each element
		for (vector< ISearchNode* >::iterator it = subsetOpenSet.begin(); it != subsetOpenSet.end(); ++it)
		{
			ISearchNode* current = *it;
			cout << "Expansion Node: Heuristic=" << current->getHeuristic() << ", Cost=" << current->getCost() << endl;

			vector< ISearchNode* > expansionSet = current->generateSuccessorNodes();

			// only accept expanded element if not a duplicate state
			for (vector< ISearchNode* >::iterator it = expansionSet.begin(); it != expansionSet.end(); ++it)
			{
				ISearchNode* state = *it;
				if (!isDuplicate(state, candidateSet) && !isDuplicate(state, costSet))
				{
					candidateSet.push(state);
				}
			}
		}

		// add remaining openSet to candidate set
		while (!openSet.empty())
		{
			ISearchNode* current = openSet.top();
			openSet.pop();

			if (!isDuplicate(current, candidateSet) && !isDuplicate(current, costSet))
			{
				candidateSet.push(current);
			}
		}
		
		// side effect: openSet now empty

		return candidateSet;
	}

	/**************** Greedy Procedure ****************/

	GreedySearchProcedure::GreedySearchProcedure()
	{
		this->beamSize = 1;
	}

	GreedySearchProcedure::~GreedySearchProcedure()
	{
	}



	/**************** Search Node ****************/

	vector< ISearchProcedure::ISearchNode* > ISearchProcedure::ISearchNode::generateSuccessorNodes()
	{
		vector< ISearchNode* > successors;
		vector< ImgLabeling > YPredSet = this->searchSpace->generateSuccessors(*this->X, this->YPred);
		for (vector< ImgLabeling >::iterator it = YPredSet.begin(); it != YPredSet.end(); it++)
		{
			ImgLabeling YPred = *it;
			switch (getType())
			{
			case LL:
				{
					LLSearchNode* successor = new LLSearchNode(this, YPred);
					successors.push_back(successor);
				}
				break;
			case HL:
				{
					HLSearchNode* successor = new HLSearchNode(this, YPred);
					successors.push_back(successor);
				}
				break;
			case LC:
				{
					LCSearchNode* successor = new LCSearchNode(this, YPred);
					successors.push_back(successor);
				}
				break;
			case HC:
				{
					HCSearchNode* successor = new HCSearchNode(this, YPred);
					successors.push_back(successor);
				}
				break;
			case LEARN_H:
				{
					LearnHSearchNode* successor = new LearnHSearchNode(this, YPred);
					successors.push_back(successor);
				}
				break;
			case LEARN_C:
				{
					LearnCSearchNode* successor = new LearnCSearchNode(this, YPred);
					successors.push_back(successor);
				}
				break;
			case LEARN_C_ORACLE_H:
				{
					LearnCOracleHSearchNode* successor = new LearnCOracleHSearchNode(this, YPred);
					successors.push_back(successor);
				}
				break;
			default:
				cerr << "[Error] not a valid search type for generating successor" << endl;
			}
		}
		return successors;
	}

	RankFeatures ISearchProcedure::ISearchNode::getHeuristicFeatures()
	{
		cerr << "[Error] heuristic features are not defined" << endl;
		return RankFeatures();
	}

	RankFeatures ISearchProcedure::ISearchNode::getCostFeatures()
	{
		cerr << "[Error] cost features are not defined" << endl;
		return RankFeatures();
	}

	ImgLabeling ISearchProcedure::ISearchNode::getY()
	{
		return this->YPred;
	}

	/**************** LL Search Node ****************/

	ISearchProcedure::LLSearchNode::LLSearchNode()
	{
	}

	ISearchProcedure::LLSearchNode::LLSearchNode(ImgFeatures* X, ImgLabeling* YTruth, SearchSpace* searchSpace)
	{
		this->parent = NULL;
		this->X = X;
		this->YPred = searchSpace->getInitialPrediction(*X);
		this->searchSpace = searchSpace;

		this->YTruth = YTruth;
		this->loss = searchSpace->computeLoss(this->YPred, *YTruth);
	}

	ISearchProcedure::LLSearchNode::LLSearchNode(ISearchNode* parent, ImgLabeling YPred)
	{
		LLSearchNode* parentCast = dynamic_cast<LLSearchNode*>(parent);

		this->parent = parentCast;
		this->X = parentCast->X;
		this->YPred = YPred;
		this->searchSpace = parentCast->searchSpace;

		this->YTruth = parentCast->YTruth;
		this->loss = this->searchSpace->computeLoss(this->YPred, *this->YTruth);
	}

	double ISearchProcedure::LLSearchNode::getHeuristic()
	{
		return this->loss;
	}

	double ISearchProcedure::LLSearchNode::getCost()
	{
		return this->loss;
	}

	SearchType ISearchProcedure::LLSearchNode::getType()
	{
		return LL;
	}

	/**************** HL Search Node ****************/

	ISearchProcedure::HLSearchNode::HLSearchNode()
	{
	}

	ISearchProcedure::HLSearchNode::HLSearchNode(ImgFeatures* X, ImgLabeling* YTruth, SearchSpace* searchSpace, IRankModel* heuristicModel)
	{
		this->parent = NULL;
		this->X = X;
		this->YPred = searchSpace->getInitialPrediction(*X);
		this->searchSpace = searchSpace;

		this->heuristicFeatures = searchSpace->computeHeuristicFeatures(*X, this->YPred);
		this->heuristicModel = heuristicModel;
		this->YTruth = YTruth;
		this->heuristic = heuristicModel->rank(this->heuristicFeatures);
		this->loss = searchSpace->computeLoss(this->YPred, *YTruth);
	}

	ISearchProcedure::HLSearchNode::HLSearchNode(ISearchNode* parent, ImgLabeling YPred)
	{
		HLSearchNode* parentCast = dynamic_cast<HLSearchNode*>(parent);

		this->parent = parentCast;
		this->X = parentCast->X;
		this->YPred = YPred;
		this->searchSpace = parentCast->searchSpace;

		this->heuristicFeatures = this->searchSpace->computeHeuristicFeatures(*this->X, this->YPred);
		this->heuristicModel = parentCast->heuristicModel;
		this->YTruth = parentCast->YTruth;
		this->heuristic = this->heuristicModel->rank(this->heuristicFeatures);
		this->loss = this->searchSpace->computeLoss(this->YPred, *this->YTruth);
	}

	RankFeatures ISearchProcedure::HLSearchNode::getHeuristicFeatures()
	{
		return this->heuristicFeatures;
	}

	double ISearchProcedure::HLSearchNode::getHeuristic()
	{
		return this->heuristic;
	}

	double ISearchProcedure::HLSearchNode::getCost()
	{
		return this->loss;
	}

	SearchType ISearchProcedure::HLSearchNode::getType()
	{
		return HL;
	}

	/**************** LC Search Node ****************/

	ISearchProcedure::LCSearchNode::LCSearchNode()
	{
	}

	ISearchProcedure::LCSearchNode::LCSearchNode(ImgFeatures* X, ImgLabeling* YTruth, SearchSpace* searchSpace, IRankModel* costModel)
	{
		this->parent = NULL;
		this->X = X;
		this->YPred = searchSpace->getInitialPrediction(*X);
		this->searchSpace = searchSpace;

		this->costFeatures = searchSpace->computeCostFeatures(*X, this->YPred);
		this->costModel = costModel;
		this->YTruth = YTruth;
		this->cost = costModel->rank(this->costFeatures);
		this->loss = searchSpace->computeLoss(this->YPred, *YTruth);
	}

	ISearchProcedure::LCSearchNode::LCSearchNode(ISearchNode* parent, ImgLabeling YPred)
	{
		LCSearchNode* parentCast = dynamic_cast<LCSearchNode*>(parent);

		this->parent = parentCast;
		this->X = parentCast->X;
		this->YPred = YPred;
		this->searchSpace = parentCast->searchSpace;

		this->costFeatures = this->searchSpace->computeCostFeatures(*this->X, this->YPred);
		this->costModel = parentCast->costModel;
		this->YTruth = parentCast->YTruth;
		this->cost = this->costModel->rank(this->costFeatures);
		this->loss = this->searchSpace->computeLoss(this->YPred, *this->YTruth);
	}

	RankFeatures ISearchProcedure::LCSearchNode::getCostFeatures()
	{
		return this->costFeatures;
	}

	double ISearchProcedure::LCSearchNode::getHeuristic()
	{
		return this->loss;
	}

	double ISearchProcedure::LCSearchNode::getCost()
	{
		return this->cost;
	}

	SearchType ISearchProcedure::LCSearchNode::getType()
	{
		return LC;
	}

	/**************** HC Search Node ****************/

	ISearchProcedure::HCSearchNode::HCSearchNode()
	{
	}

	ISearchProcedure::HCSearchNode::HCSearchNode(ImgFeatures* X, SearchSpace* searchSpace, IRankModel* heuristicModel, IRankModel* costModel)
	{
		this->parent = NULL;
		this->X = X;
		this->YPred = searchSpace->getInitialPrediction(*X);
		this->searchSpace = searchSpace;

		this->heuristicFeatures = searchSpace->computeHeuristicFeatures(*X, this->YPred);
		this->costFeatures = searchSpace->computeCostFeatures(*X, this->YPred);
		this->heuristicModel = heuristicModel;
		this->costModel = costModel;
		this->heuristic = heuristicModel->rank(this->heuristicFeatures);
		this->cost = costModel->rank(this->costFeatures);
	}

	ISearchProcedure::HCSearchNode::HCSearchNode(ISearchNode* parent, ImgLabeling YPred)
	{
		HCSearchNode* parentCast = dynamic_cast<HCSearchNode*>(parent);

		this->parent = parentCast;
		this->X = parentCast->X;
		this->YPred = YPred;
		this->searchSpace = parentCast->searchSpace;

		this->heuristicFeatures = this->searchSpace->computeHeuristicFeatures(*this->X, this->YPred);
		this->costFeatures = this->searchSpace->computeCostFeatures(*this->X, this->YPred);
		this->heuristicModel = parentCast->heuristicModel;
		this->costModel = parentCast->costModel;
		this->heuristic = this->heuristicModel->rank(this->heuristicFeatures);
		this->cost = this->costModel->rank(this->costFeatures);
	}

	RankFeatures ISearchProcedure::HCSearchNode::getHeuristicFeatures()
	{
		return this->heuristicFeatures;
	}

	RankFeatures ISearchProcedure::HCSearchNode::getCostFeatures()
	{
		return this->costFeatures;
	}

	double ISearchProcedure::HCSearchNode::getHeuristic()
	{
		return this->heuristic;
	}

	double ISearchProcedure::HCSearchNode::getCost()
	{
		return this->cost;
	}

	SearchType ISearchProcedure::HCSearchNode::getType()
	{
		return HC;
	}

	/**************** Learn H Search Node ****************/

	ISearchProcedure::LearnHSearchNode::LearnHSearchNode()
	{
	}

	ISearchProcedure::LearnHSearchNode::LearnHSearchNode(ImgFeatures* X, ImgLabeling* YTruth, SearchSpace* searchSpace)
	{
		this->parent = NULL;
		this->X = X;
		this->YPred = searchSpace->getInitialPrediction(*X);
		this->searchSpace= searchSpace;

		this->heuristicFeatures = searchSpace->computeHeuristicFeatures(*X, this->YPred);
		this->YTruth = YTruth;
		this->loss = searchSpace->computeLoss(this->YPred, *YTruth);
	}

	ISearchProcedure::LearnHSearchNode::LearnHSearchNode(ISearchNode* parent, ImgLabeling YPred)
	{
		LearnHSearchNode* parentCast = dynamic_cast<LearnHSearchNode*>(parent);

		this->parent = parentCast;
		this->X = parentCast->X;
		this->YPred = YPred;
		this->searchSpace = parentCast->searchSpace;

		this->heuristicFeatures = this->searchSpace->computeHeuristicFeatures(*this->X, this->YPred);
		this->YTruth = parentCast->YTruth;
		this->loss = this->searchSpace->computeLoss(this->YPred, *this->YTruth);
	}

	RankFeatures ISearchProcedure::LearnHSearchNode::getHeuristicFeatures()
	{
		return this->heuristicFeatures;
	}

	SearchType ISearchProcedure::LearnHSearchNode::getType()
	{
		return LEARN_H;
	}

	/**************** Learn C Search Node ****************/

	ISearchProcedure::LearnCSearchNode::LearnCSearchNode()
	{
	}

	ISearchProcedure::LearnCSearchNode::LearnCSearchNode(ImgFeatures* X, ImgLabeling* YTruth, SearchSpace* searchSpace, IRankModel* heuristicModel)
	{
		this->parent = NULL;
		this->X = X;
		this->YPred = searchSpace->getInitialPrediction(*X);
		this->searchSpace= searchSpace;

		this->heuristicFeatures = searchSpace->computeHeuristicFeatures(*X, this->YPred);
		this->costFeatures = this->searchSpace->computeCostFeatures(*this->X, this->YPred);
		this->heuristicModel = heuristicModel;
		this->YTruth = YTruth;
		this->heuristic = heuristicModel->rank(this->heuristicFeatures);
		this->loss = searchSpace->computeLoss(this->YPred, *YTruth);
	}

	ISearchProcedure::LearnCSearchNode::LearnCSearchNode(ISearchNode* parent, ImgLabeling YPred)
	{
		LearnCSearchNode* parentCast = dynamic_cast<LearnCSearchNode*>(parent);

		this->parent = parentCast;
		this->X = parentCast->X;
		this->YPred = YPred;
		this->searchSpace = parentCast->searchSpace;

		this->heuristicFeatures = this->searchSpace->computeHeuristicFeatures(*this->X, this->YPred);
		this->costFeatures = this->searchSpace->computeCostFeatures(*this->X, this->YPred);
		this->heuristicModel = parentCast->heuristicModel;
		this->YTruth = parentCast->YTruth;
		this->heuristic = this->heuristicModel->rank(this->heuristicFeatures);
		this->loss = this->searchSpace->computeLoss(this->YPred, *this->YTruth);
	}

	RankFeatures ISearchProcedure::LearnCSearchNode::getCostFeatures()
	{
		return this->costFeatures;
	}

	SearchType ISearchProcedure::LearnCSearchNode::getType()
	{
		return LEARN_C;
	}

	/**************** Learn C Given Oracle H Search Node ****************/

	ISearchProcedure::LearnCOracleHSearchNode::LearnCOracleHSearchNode()
	{
	}

	ISearchProcedure::LearnCOracleHSearchNode::LearnCOracleHSearchNode(ImgFeatures* X, ImgLabeling* YTruth, SearchSpace* searchSpace)
	{
		this->parent = NULL;
		this->X = X;
		this->YPred = searchSpace->getInitialPrediction(*X);
		this->searchSpace = searchSpace;

		this->costFeatures = searchSpace->computeCostFeatures(*X, this->YPred);
		this->YTruth = YTruth;
		this->loss = searchSpace->computeLoss(this->YPred, *YTruth);
	}

	ISearchProcedure::LearnCOracleHSearchNode::LearnCOracleHSearchNode(ISearchNode* parent, ImgLabeling YPred)
	{
		LearnCOracleHSearchNode* parentCast = dynamic_cast<LearnCOracleHSearchNode*>(parent);

		this->parent = parentCast;
		this->X = parentCast->X;
		this->YPred = YPred;
		this->searchSpace = parentCast->searchSpace;

		this->costFeatures = this->searchSpace->computeCostFeatures(*this->X, this->YPred);
		this->YTruth = parentCast->YTruth;
		this->loss = this->searchSpace->computeLoss(this->YPred, *this->YTruth);
	}

	RankFeatures ISearchProcedure::LearnCOracleHSearchNode::getCostFeatures()
	{
		return this->costFeatures;
	}

	SearchType ISearchProcedure::LearnCOracleHSearchNode::getType()
	{
		return LEARN_C_ORACLE_H;
	}

	/**************** Compare Search Node ****************/

	bool ISearchProcedure::CompareSearchNode::operator() (ISearchNode*& lhs, ISearchNode*& rhs) const
	{
		if (prioritizeBy == HEURISTIC)
		{
			return lhs->getHeuristic() > rhs->getHeuristic();
		}
		else if (prioritizeBy == COST)
		{
			return lhs->getCost() > rhs->getCost();
		}
		else
		{
			cerr << "prioritizing by something other than heuristic or cost." << endl;
			return false;
		}
	}
}