#include <iostream>
#include <ctime>
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
			LOG(ERROR) << "cannot open file to write nodes!";
		}
	}

	void SavePrediction::saveCuts(ImgLabeling& YPred, string fileName)
	{
		if (!YPred.stochasticCutsAvailable)
		{
			LOG(ERROR) << "no cuts available to write!";
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
			LOG(ERROR) << "cannot open file to write edges!";
		}
	}

	void SavePrediction::saveLabelMask(ImgFeatures& X, ImgLabeling& YPred, string fileName)
	{
		if (!X.segmentsAvailable)
		{
			LOG(ERROR) << "no segments data available to write!";
			return;
		}

		// write to file
		ofstream fh(fileName.c_str());
		if (fh.is_open())
		{
			const int height = X.segments.rows();
			const int width = X.segments.cols();
			for (int row = 0; row < height; row++)
			{
				for (int col = 0; col < width; col++)
				{
					int segmentID = X.segments(row, col);
					int label = YPred.getLabel(segmentID);
					fh << label << " ";
				}
				fh << endl;
			}

			fh.close();
		}
		else
		{
			LOG(ERROR) << "cannot open file to write label mask!";
		}
	}

	void SavePrediction::saveCandidateLosses(vector<double>& losses, string fileName)
	{
		// write to file
		ofstream fh(fileName.c_str());
		if (fh.is_open())
		{
			for (vector<double>::iterator it = losses.begin(); it != losses.end(); ++it)
			{
				double loss = *it;
				fh << loss << endl;
			}

			fh.close();
		}
		else
		{
			LOG(ERROR) << "cannot open file to write candidate losses!";
		}
	}

	/**************** Search Procedure ****************/

	ISearchProcedure::SearchNode* ISearchProcedure::createRootNode(SearchType searchType, ImgFeatures& X, ImgLabeling* YTruth, 
		SearchSpace* searchSpace, IRankModel* heuristicModel, IRankModel* costModel)
	{
		SearchNode* root = NULL;
		switch (searchType)
		{
			case LL:
				root = new SearchNode(&X, YTruth, searchSpace, NULL, NULL, searchType);
				break;
			case HL:
				root = new SearchNode(&X, YTruth, searchSpace, heuristicModel, NULL, searchType);
				break;
			case LC:
				root = new SearchNode(&X, YTruth, searchSpace, NULL, costModel, searchType);
				break;
			case HC:
				root = new SearchNode(&X, NULL, searchSpace, heuristicModel, costModel, searchType);
				break;
			case LEARN_H:
				root = new SearchNode(&X, YTruth, searchSpace, NULL, NULL, searchType);
				break;
			case LEARN_C:
				root = new SearchNode(&X, YTruth, searchSpace, heuristicModel, NULL, searchType);
				break;
			case LEARN_C_ORACLE_H:
				root = new SearchNode(&X, YTruth, searchSpace, NULL, NULL, searchType);
				break;
			default:
				LOG(ERROR) << "searchType constant is invalid.";
		}
		return root;
	}

	ISearchProcedure::SearchMetadata::SearchMetadata()
	{
		this->saveAnytimePredictions = false;
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

	void ISearchProcedure::trainHeuristicRanker(IRankModel* ranker, vector< RankFeatures > bestFeatures, vector< double > bestLosses, 
			vector< RankFeatures > worstFeatures, vector< double > worstLosses)
	{
		// train depending on ranker
		if (ranker->rankerType() == SVM_RANK)
		{
			// train
			SVMRankModel* svmRankModel = dynamic_cast<SVMRankModel*>(ranker);
			svmRankModel->addTrainingExamples(bestFeatures, worstFeatures);
		}
		else if (ranker->rankerType() == VW_RANK)
		{
			// train
			VWRankModel* vwRankModel = dynamic_cast<VWRankModel*>(ranker);
			vwRankModel->addTrainingExamples(bestFeatures, worstFeatures, bestLosses, worstLosses);
		}
		else
		{
			LOG(ERROR) << "unknown ranker type";
			abort();
		}
	}

	void ISearchProcedure::trainCostRanker(IRankModel* ranker, SearchNodeCostPQ& costSet)
	{
		vector< RankFeatures > bestFeatures;
		vector< RankFeatures > worstFeatures;
		vector< double > bestLosses;
		vector< double > worstLosses;

		// get best states
		double overallBestLoss = costSet.top()->getCost();
		while (!costSet.empty() && costSet.top()->getCost() <= overallBestLoss)
		{
			SearchNode* state = costSet.top();
			bestFeatures.push_back(state->getCostFeatures());
			bestLosses.push_back(state->getCost());
			costSet.pop();
			delete state;
		}

		// get worst states
		while (!costSet.empty())
		{
			SearchNode* state = costSet.top();
			worstFeatures.push_back(state->getCostFeatures());
			worstLosses.push_back(state->getCost());
			costSet.pop();
			delete state;
		}

		// train depending on ranker
		if (ranker->rankerType() == SVM_RANK)
		{
			// train
			SVMRankModel* svmRankModel = dynamic_cast<SVMRankModel*>(ranker);
			svmRankModel->addTrainingExamples(bestFeatures, worstFeatures);

		}
		else if (ranker->rankerType() == VW_RANK)
		{
			// train
			VWRankModel* vwRankModel = dynamic_cast<VWRankModel*>(ranker);
			vwRankModel->addTrainingExamples(bestFeatures, worstFeatures, bestLosses, worstLosses);

		}
		else
		{
			LOG(ERROR) << "unknown ranker type";
			abort();
		}
	}

	ImgLabeling IBasicSearchProcedure::performSearch(SearchType searchType, ImgFeatures& X, ImgLabeling* YTruth, 
	int timeBound, SearchSpace* searchSpace, IRankModel* heuristicModel, IRankModel* costModel, 
	IRankModel* pruneModel, SearchMetadata searchMetadata)
	{
		clock_t tic = clock();

		// set up priority queues
		// maintain open set for search
		// maintain cost set for returning best cost in the end
		SearchNodeCostPQ costSet;
		SearchNodeHeuristicPQ openSet;

		// push initial state into queue
		SearchNode* root = createRootNode(searchType, X, YTruth, searchSpace, heuristicModel, costModel);
		openSet.push(root);
		costSet.push(root);

		// while the open set is not empty and the time step is less than the time bound,
		// perform search...
		int timeStep = 0;
		while (!openSet.empty() && timeStep < timeBound)
		{
			LOG() << endl << "Running " << SearchTypeStrings[searchType] << " search with time step " << timeStep+1 << "/" << timeBound << "..." << endl;
			clock_t ticInside = clock();

			// save current best if anytime prediction enabled
			saveAnyTimePrediction(costSet.top()->getY(), timeStep, searchMetadata, searchType);

			/***** pick some subset of elements from the open set *****/

			vector< SearchNode* > subsetOpenSet = selectSubsetOpenSet(openSet);

			/***** expand these elements *****/

			SearchNodeHeuristicPQ candidateSet = expandElements(subsetOpenSet, openSet, costSet, pruneModel, YTruth, searchType, timeStep, timeBound);

			/***** choose successors and put them into the open set *****/
			/***** put these expanded elements into the cost set *****/

			vector< RankFeatures > bestFeatures;
			vector< RankFeatures > worstFeatures;
			vector< double > bestLosses; // heuristic values technically, but definitely losses for learning H
			vector< double > worstLosses;
			chooseSuccessors(searchType, candidateSet, openSet, costSet, bestFeatures, bestLosses, worstFeatures, worstLosses);

			/***** use best/worst candidates as training examples for heuristic learning (if applicable) *****/

			if (searchType == LEARN_H)
				trainHeuristicRanker(heuristicModel, bestFeatures, bestLosses, worstFeatures, worstLosses);

			/***** done with this search step *****/

			clock_t tocInside = clock();
			LOG() << "search step " << timeStep << " total time: " << (double)(tocInside - ticInside)/CLOCKS_PER_SEC << endl;

			/***** increment time step *****/
			timeStep++;
		}

		/***** search is done, return the lowest cost search node *****/

		if (costSet.empty())
		{
			LOG(ERROR) << "the cost set is empty, which is not possible!";
			abort();
		}

		// Get lowest cost node
		SearchNode* lowestCost = costSet.top();
		ImgLabeling prediction = lowestCost->getY();
		LOG() << endl << "Finished search. Cost=" << lowestCost->getCost() << endl;

		// use best/worst cost set candidates as training examples for cost learning (if applicable)
		if (searchType == LEARN_C || searchType == LEARN_C_ORACLE_H)
			trainCostRanker(costModel, costSet);

		// clean up cost set
		//deleteQueueElements(costSet);
		vector<double> candidateLosses;
		while (!costSet.empty())
		{
			SearchNode* state = costSet.top();
			costSet.pop();
			if (YTruth != NULL)
			{
				ImgLabeling YPred = state->getY();
				candidateLosses.push_back(searchSpace->computeLoss(YPred, *YTruth));
			}
			delete state;
		}
		if (YTruth != NULL)
		{
			stringstream ssLosses;
			ssLosses << Global::settings->paths->OUTPUT_RESULTS_DIR << "candidatelosses" 
				<< "_" << SearchTypeStrings[searchType] 
				<< "_" << DatasetTypeStrings[searchMetadata.setType] 
				<< "_time" << timeBound 
					<< "_fold" << searchMetadata.iter 
					<< "_" << searchMetadata.exampleName << ".txt";
			SavePrediction::saveCandidateLosses(candidateLosses, ssLosses.str());
		}

		clock_t toc = clock();
		LOG() << "total search time: " << (double)(toc - tic)/CLOCKS_PER_SEC << endl << endl;

		return prediction;
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

	vector< ISearchProcedure::SearchNode* > BreadthFirstBeamSearchProcedure::selectSubsetOpenSet(SearchNodeHeuristicPQ& openSet)
	{
		vector< SearchNode* > subsetOpenSet;
		
		// just add everything (should only be at most B states)
		while (!openSet.empty())
		{
			SearchNode* current = openSet.top();
			openSet.pop();
			subsetOpenSet.push_back(current);
		}

		return subsetOpenSet;
	}

	ISearchProcedure::SearchNodeHeuristicPQ BreadthFirstBeamSearchProcedure::expandElements(vector< SearchNode* > subsetOpenSet, SearchNodeHeuristicPQ& openSet, SearchNodeCostPQ& costSet,
		IRankModel* pruneModel, ImgLabeling* YTruth, SearchType searchType, int timeStep, int timeBound)
	{
		SearchNodeHeuristicPQ candidateSet;
		
		// expand each element
		for (vector< SearchNode* >::iterator it = subsetOpenSet.begin(); it != subsetOpenSet.end(); ++it)
		{
			SearchNode* current = *it;
			LOG() << "Expansion Node: Heuristic=" << current->getHeuristic() << ", Cost=" << current->getCost() << endl;

			vector< SearchNode* > expansionSet;
			if (searchType == LEARN_PRUNE)
				expansionSet = current->generateSuccessorNodesForPruneLearning(pruneModel, YTruth, timeStep, timeBound);
			else
				expansionSet = current->generateSuccessorNodes(true, timeStep, timeBound, YTruth);

			// only accept expanded element if not a duplicate state
			for (vector< SearchNode* >::iterator it = expansionSet.begin(); it != expansionSet.end(); ++it)
			{
				SearchNode* state = *it;
				if (!isDuplicate(state, candidateSet) && !isDuplicate(state, costSet))
				{
					candidateSet.push(state);
				}
			}
		}

		return candidateSet;
	}

	void BreadthFirstBeamSearchProcedure::chooseSuccessors(SearchType searchType, SearchNodeHeuristicPQ& candidateSet, SearchNodeHeuristicPQ& openSet, SearchNodeCostPQ& costSet, 
			vector< RankFeatures >& bestSet, vector< double >& bestLosses, vector< RankFeatures >& worstSet, vector< double >& worstLosses)
	{
		// pick the top B best states - "good" training examples
		double bestLoss;
		for (int i = 0; i < this->beamSize; i++)
		{
			if (candidateSet.empty())
				break;

			SearchNode* state = candidateSet.top();
			candidateSet.pop();
			if (searchType == LEARN_H)
			{
				bestSet.push_back(state->getHeuristicFeatures());
				bestLosses.push_back(state->getHeuristic());
				bestLoss = state->getHeuristic();
			}
			openSet.push(state);
			costSet.push(state);
		}

		// everything else that is still the same as best loss is not considered bad for training
		while (!candidateSet.empty() && candidateSet.top()->getHeuristic() <= bestLoss)
		{
			SearchNode* state = candidateSet.top();
			candidateSet.pop();
			if (searchType == LEARN_H)
			{
				bestSet.push_back(state->getHeuristicFeatures());
				bestLosses.push_back(state->getHeuristic());
			}
			costSet.push(state);
		}

		// everything else is "bad" for training examples
		while (!candidateSet.empty())
		{
			SearchNode* state = candidateSet.top();
			candidateSet.pop();
			if (searchType == LEARN_H)
			{
				worstSet.push_back(state->getHeuristicFeatures());
				worstLosses.push_back(state->getHeuristic());
			}
			costSet.push(state);
		}

		// side effect: candidateSet is empty now
		// side effect: openSet has the best states now
		// side effect: costSet has all the states from the candidate set
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

	vector< ISearchProcedure::SearchNode* > BestFirstBeamSearchProcedure::selectSubsetOpenSet(SearchNodeHeuristicPQ& openSet)
	{
		vector< SearchNode* > subsetOpenSet;
		
		// only add top element
		if (!openSet.empty())
		{
			SearchNode* current = openSet.top();
			openSet.pop();
			subsetOpenSet.push_back(current);
		}

		return subsetOpenSet;
	}

	ISearchProcedure::SearchNodeHeuristicPQ BestFirstBeamSearchProcedure::expandElements(vector< SearchNode* > subsetOpenSet, SearchNodeHeuristicPQ& openSet, SearchNodeCostPQ& costSet,
		IRankModel* pruneModel, ImgLabeling* YTruth, SearchType searchType, int timeStep, int timeBound)
	{
		SearchNodeHeuristicPQ candidateSet;
		
		// expand each element
		for (vector< SearchNode* >::iterator it = subsetOpenSet.begin(); it != subsetOpenSet.end(); ++it)
		{
			SearchNode* current = *it;
			LOG() << "Expansion Node: Heuristic=" << current->getHeuristic() << ", Cost=" << current->getCost() << endl;

			vector< SearchNode* > expansionSet;
			if (searchType == LEARN_PRUNE)
				expansionSet = current->generateSuccessorNodesForPruneLearning(pruneModel, YTruth, timeStep, timeBound);
			else
				expansionSet = current->generateSuccessorNodes(true, timeStep, timeBound, YTruth);

			// only accept expanded element if not a duplicate state
			for (vector< SearchNode* >::iterator it = expansionSet.begin(); it != expansionSet.end(); ++it)
			{
				SearchNode* state = *it;
				if (!isDuplicate(state, candidateSet) && !isDuplicate(state, costSet))
				{
					candidateSet.push(state);
				}
			}
		}

		// add remaining openSet to candidate set
		while (!openSet.empty())
		{
			SearchNode* current = openSet.top();
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

	ISearchProcedure::SearchNode::SearchNode()
	{
	}

	ISearchProcedure::SearchNode::SearchNode(ImgFeatures* X, ImgLabeling* YTruth, SearchSpace* searchSpace, 
			IRankModel* heuristicModel, IRankModel* costModel, SearchType searchType)
	{
		if (X == NULL || searchSpace == NULL)
		{
			LOG(ERROR) << "invalid parameters for search node initialization.";
			abort();
		}

		this->parent = NULL;
		this->searchSpace = searchSpace;
		this->searchType = searchType;
		this->X = X;
		this->YTruth = YTruth;
		this->heuristicModel = heuristicModel;
		this->costModel = costModel;

		this->YPred = this->searchSpace->getInitialPrediction(*this->X);

		constructorHelper();
	}

	ISearchProcedure::SearchNode::SearchNode(SearchNode* parent, ImgLabeling YPred)
	{
		if (parent == NULL)
		{
			LOG(ERROR) << "invalid parameters for search node generation.";
			abort();
		}

		this->parent = parent;
		this->searchSpace = this->parent->searchSpace;
		this->searchType = this->parent->searchType;
		this->X = this->parent->X;
		this->YTruth = this->parent->YTruth;
		this->heuristicModel = this->parent->heuristicModel;
		this->costModel = this->parent->costModel;

		this->YPred = YPred;

		constructorHelper();
	}

	void ISearchProcedure::SearchNode::constructorHelper()
	{
		double loss;
		if (this->YTruth != NULL)
		{
			loss = this->searchSpace->computeLoss(this->YPred, *this->YTruth);
		}

		switch (this->searchType)
		{
			case LL:
			{
				this->heuristic = loss;
				this->cost = loss;
				break;
			}
			case HL:
			{
				this->heuristicFeatures = this->searchSpace->computeHeuristicFeatures(*this->X, this->YPred);
				this->heuristic = this->heuristicModel->rank(this->heuristicFeatures);
				this->cost = loss;
				break;
			}
			case LC:
			{
				this->heuristic = loss;
				this->costFeatures = this->searchSpace->computeCostFeatures(*this->X, this->YPred);
				this->cost = this->costModel->rank(this->costFeatures);
				break;
			}
			case HC:
			{
				this->heuristicFeatures = this->searchSpace->computeHeuristicFeatures(*this->X, this->YPred);
				this->heuristic = this->heuristicModel->rank(this->heuristicFeatures);
				this->costFeatures = this->searchSpace->computeCostFeatures(*this->X, this->YPred);
				this->cost = this->costModel->rank(this->costFeatures);
				break;
			}
			case LEARN_H:
			{
				this->heuristicFeatures = this->searchSpace->computeHeuristicFeatures(*this->X, this->YPred);
				this->heuristic = loss;
				this->cost = loss;
				break;
			}
			case LEARN_C:
			{
				this->heuristicFeatures = this->searchSpace->computeHeuristicFeatures(*this->X, this->YPred);
				this->costFeatures = this->searchSpace->computeCostFeatures(*this->X, this->YPred);
				this->heuristic = this->heuristicModel->rank(this->heuristicFeatures);
				this->cost = loss;
				break;
			}
			case LEARN_C_ORACLE_H:
			{
				this->costFeatures = this->searchSpace->computeCostFeatures(*this->X, this->YPred);
				this->heuristic = loss;
				this->cost = loss;
				break;
			}
			default:
				LOG(ERROR) << "searchType constant is invalid.";
		}
	}

	vector< ISearchProcedure::SearchNode* > ISearchProcedure::SearchNode::generateSuccessorNodes(bool prune, int timeStep, int timeBound, ImgLabeling* YTruth)
	{
		vector< SearchNode* > successors;

		// generate successors
		vector< ImgCandidate > YPredSet = this->searchSpace->generateSuccessors(*this->X, this->YPred, timeStep, timeBound);

		// prune successors
		if (prune)
			YPredSet = this->searchSpace->pruneSuccessors(*this->X, this->YPred, YPredSet, YTruth, this->searchSpace->getLossFunction());

		for (vector< ImgCandidate >::iterator it = YPredSet.begin(); it != YPredSet.end(); it++)
		{
			ImgCandidate YCandidate = *it;
			ImgLabeling YCandPred = YCandidate.labeling;
			SearchNode* successor = new SearchNode(this, YPred);
			successors.push_back(successor);
		}
		return successors;
	}

	vector< ISearchProcedure::SearchNode* > ISearchProcedure::SearchNode::generateSuccessorNodesForPruneLearning(IRankModel* learningModel, 
		ImgLabeling* YTruth, int timeStep, int timeBound)
	{
		vector< SearchNode* > successors;

		double prevLoss = this->searchSpace->computeLoss(this->YPred, *YTruth);
		RankFeatures prevPruneFeatures = this->searchSpace->computePruneFeatures(*this->X, *YTruth, set<int>());

		// generate successors
		vector< ImgCandidate > YPredSet = this->searchSpace->generateSuccessors(*this->X, this->YPred, timeStep, timeBound);

		// collect training examples
		for (vector< ImgCandidate >::iterator it = YPredSet.begin(); it != YPredSet.end(); it++)
		{
			ImgCandidate YCandidate = *it;
			ImgLabeling YCandPred = YCandidate.labeling;
			
			// collect training examples
			double candLoss = this->searchSpace->computeLoss(YCandPred, *YTruth);
			set<int> action = YCandidate.action;
			RankFeatures pruneFeatures = this->searchSpace->computePruneFeatures(*this->X, YCandPred, action);

			if (learningModel->rankerType() == SVM_RANK)
			{
				SVMRankModel* svmModel = dynamic_cast<SVMRankModel*>(learningModel);
				if (candLoss <= prevLoss)
					svmModel->addTrainingExample(pruneFeatures, prevPruneFeatures);
				else
					svmModel->addTrainingExample(prevPruneFeatures, pruneFeatures);

			}
			else if (learningModel->rankerType() == VW_RANK)
			{
				VWRankModel* vwModel = dynamic_cast<VWRankModel*>(learningModel);
				if (candLoss <= prevLoss)
					vwModel->addTrainingExample(pruneFeatures, prevPruneFeatures, candLoss, prevLoss);
				else
					vwModel->addTrainingExample(prevPruneFeatures, pruneFeatures, prevLoss, candLoss);

			}
			else
			{
				LOG(ERROR) << "unknown ranker for prune training positive example";
				abort();
			}

			// generate examples
			SearchNode* successor = new SearchNode(this, YCandPred);
			successors.push_back(successor);
		}
		return successors;
	}

	RankFeatures ISearchProcedure::SearchNode::getHeuristicFeatures()
	{
		if (getType() == LL || getType() == LC || getType() == LEARN_C)
		{
			LOG(ERROR) << "heuristic features are not defined";
			return RankFeatures();
		}

		return this->heuristicFeatures;
	}

	RankFeatures ISearchProcedure::SearchNode::getCostFeatures()
	{
		if (getType() == LL || getType() == HL || getType() == LEARN_H || getType() == LEARN_C_ORACLE_H)
		{
			LOG(ERROR) << "cost features are not defined";
			return RankFeatures();
		}

		return this->costFeatures;
	}

	double ISearchProcedure::SearchNode::getHeuristic()
	{
		return this->heuristic;
	}

	double ISearchProcedure::SearchNode::getCost()
	{
		return this->cost;
	}

	ImgLabeling ISearchProcedure::SearchNode::getY()
	{
		return this->YPred;
	}

	SearchType ISearchProcedure::SearchNode::getType()
	{
		return this->searchType;
	}

	/**************** Compare Search Node ****************/

	bool ISearchProcedure::CompareByHeuristic::operator() (SearchNode*& lhs, SearchNode*& rhs) const
	{
		return lhs->getHeuristic() > rhs->getHeuristic();
	}

	bool ISearchProcedure::CompareByCost::operator() (SearchNode*& lhs, SearchNode*& rhs) const
	{
		return lhs->getCost() > rhs->getCost();
	}
}