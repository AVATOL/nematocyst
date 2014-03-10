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

	//TODO: get this value from command line
	const int ISearchProcedure::PRUNE_MAX_NUM_CANDIDATES = 100;

	ISearchProcedure::SearchMetadata::SearchMetadata()
	{
		this->saveAnytimePredictions = false;
	}

	ImgLabeling ISearchProcedure::llSearch(ImgFeatures& X, ImgLabeling* YTruth, int timeBound, 
		SearchSpace* searchSpace, SearchMetadata searchMetadata)
	{
		return searchProcedure(LL, X, YTruth, timeBound, 
			searchSpace, NULL, NULL, searchMetadata);
	}

	ImgLabeling ISearchProcedure::hlSearch(ImgFeatures& X, ImgLabeling* YTruth, int timeBound, 
		SearchSpace* searchSpace, IRankModel* heuristicModel, SearchMetadata searchMetadata)
	{
		return searchProcedure(HL, X, YTruth, timeBound, 
			searchSpace, heuristicModel, NULL, searchMetadata);
	}

	ImgLabeling ISearchProcedure::lcSearch(ImgFeatures& X, ImgLabeling* YTruth, int timeBound, 
		SearchSpace* searchSpace, IRankModel* costModel, SearchMetadata searchMetadata)
	{
		return searchProcedure(LC, X, YTruth, timeBound, 
			searchSpace, NULL, costModel, searchMetadata);
	}

	ImgLabeling ISearchProcedure::hcSearch(ImgFeatures& X, int timeBound, SearchSpace* searchSpace, 
		IRankModel* heuristicModel, IRankModel* costModel, SearchMetadata searchMetadata)
	{
		return searchProcedure(HC, X, NULL, timeBound, 
			searchSpace, heuristicModel, costModel, searchMetadata);
	}

	ImgLabeling ISearchProcedure::hcSearch(ImgFeatures& X, ImgLabeling* YTruth, int timeBound, SearchSpace* searchSpace, 
		IRankModel* heuristicModel, IRankModel* costModel, SearchMetadata searchMetadata)
	{
		return searchProcedure(HC, X, YTruth, timeBound, 
			searchSpace, heuristicModel, costModel, searchMetadata);
	}

	ImgLabeling ISearchProcedure::rlSearch(ImgFeatures& X, ImgLabeling* YTruth, int timeBound, 
		SearchSpace* searchSpace, SearchMetadata searchMetadata)
	{
		return searchProcedure(RL, X, YTruth, timeBound, 
			searchSpace, NULL, NULL, searchMetadata);
	}

	ImgLabeling ISearchProcedure::rcSearch(ImgFeatures& X, int timeBound, 
		SearchSpace* searchSpace, IRankModel* costModel, SearchMetadata searchMetadata)
	{
		return searchProcedure(RC, X, NULL, timeBound, 
			searchSpace, NULL, costModel, searchMetadata);
	}

	void ISearchProcedure::learnH(ImgFeatures& X, ImgLabeling* YTruth, int timeBound, SearchSpace* searchSpace, 
		IRankModel* learningModel, SearchMetadata searchMetadata)
	{
		searchProcedure(LEARN_H, X, YTruth, timeBound, 
			searchSpace, learningModel, NULL, searchMetadata);
	}

	void ISearchProcedure::learnC(ImgFeatures& X, ImgLabeling* YTruth, int timeBound, SearchSpace* searchSpace, 
		IRankModel* heuristicModel, IRankModel* learningModel, SearchMetadata searchMetadata)
	{
		searchProcedure(LEARN_C, X, YTruth, timeBound, 
			searchSpace, heuristicModel, learningModel, searchMetadata);
	}

	void ISearchProcedure::learnCWithOracleH(ImgFeatures& X, ImgLabeling* YTruth, int timeBound, SearchSpace* searchSpace, 
		IRankModel* learningModel, SearchMetadata searchMetadata)
	{
		searchProcedure(LEARN_C_ORACLE_H, X, YTruth, timeBound, 
			searchSpace, NULL, learningModel, searchMetadata);
	}

	void ISearchProcedure::learnCWithRandomH(ImgFeatures& X, ImgLabeling* YTruth, int timeBound, SearchSpace* searchSpace, 
		IRankModel* learningModel, SearchMetadata searchMetadata)
	{
		searchProcedure(LEARN_C_RANDOM_H, X, YTruth, timeBound, 
			searchSpace, NULL, learningModel, searchMetadata);
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
		else if (ranker->rankerType() == ONLINE_RANK)
		{
			OnlineRankModel* onlineRankModel = dynamic_cast<OnlineRankModel*>(ranker);

			// find the best scoring output in the best set according to the current heuristic model
			RankFeatures bestHeuristicFeature;
			double bestScore;
			double bestLoss;

			const int numBestFeatures = bestFeatures.size();
			for (int i = 0; i < numBestFeatures; i++)
			{
				RankFeatures feature = bestFeatures[i];
				double score = onlineRankModel->rank(feature);
				if (i == 0 || score <= bestScore)
				{
					bestHeuristicFeature = feature;
					bestScore = score;
					bestLoss = bestLosses[i];
				}
			}

			// perform update
			const int numWorstFeatures = worstFeatures.size();
			for (int i = 0; i < numWorstFeatures; i++)
			{
				RankFeatures worseFeature = worstFeatures[i];
				double score = onlineRankModel->rank(worseFeature);
				double bestScore = onlineRankModel->rank(bestHeuristicFeature);

				if (score >= bestScore)
				{
					double delta = worstLosses[i] - bestLoss;
					VectorXd featureDiff = bestHeuristicFeature.data - worseFeature.data;
					onlineRankModel->performOnlineUpdate(delta, featureDiff);
				}
			}
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
			ISearchNode* state = costSet.top();
			bestFeatures.push_back(state->getCostFeatures());
			bestLosses.push_back(state->getCost());
			costSet.pop();
			delete state;
		}

		// get worst states
		while (!costSet.empty())
		{
			ISearchNode* state = costSet.top();
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
		else if (ranker->rankerType() == ONLINE_RANK)
		{
			OnlineRankModel* onlineRankModel = dynamic_cast<OnlineRankModel*>(ranker);

			// find the best scoring output overall according to the current cost model
			double bestScore;
			bool fromWorstSet = false;

			const int numBestFeatures = bestFeatures.size();
			for (int i = 0; i < numBestFeatures; i++)
			{
				RankFeatures feature = bestFeatures[i];
				double score = onlineRankModel->rank(feature);
				if (i == 0 || score <= bestScore)
				{
					bestScore = score;
				}
			}
			const int numWorstFeatures = worstFeatures.size();
			for (int i = 0; i < numWorstFeatures; i++)
			{
				RankFeatures feature = worstFeatures[i];
				double score = onlineRankModel->rank(feature);
				if (score <= bestScore)
				{
					bestScore = score;
					fromWorstSet = true;
				}
			}

			// perform update if necessary
			if (fromWorstSet)
			{
				// find best scoring output in the best set according to current weights
				RankFeatures bestCostFeature;
				double bestScore;
				double bestLoss;

				const int numBestFeatures = bestFeatures.size();
				for (int i = 0; i < numBestFeatures; i++)
				{
					RankFeatures feature = bestFeatures[i];
					double score = onlineRankModel->rank(feature);
					if (i == 0 || score < bestScore)
					{
						bestCostFeature = feature;
						bestScore = score;
						bestLoss = bestLosses[i];
					}
				}

				// perform update
				for (int i = 0; i < numWorstFeatures; i++)
				{
					RankFeatures worseFeature = worstFeatures[i];
					double score = onlineRankModel->rank(worseFeature);
					double bestScore = onlineRankModel->rank(bestCostFeature);

					if (score >= bestScore)
					{
						double delta = worstLosses[i] - bestLoss;
						VectorXd featureDiff = bestCostFeature.data - worseFeature.data;
						onlineRankModel->performOnlineUpdate(delta, featureDiff);
					}
				}
			}
		}
		else
		{
			LOG(ERROR) << "unknown ranker type";
			abort();
		}
	}

	ImgLabeling IBasicSearchProcedure::searchProcedure(SearchType searchType, ImgFeatures& X, ImgLabeling* YTruth, 
	int timeBound, SearchSpace* searchSpace, IRankModel* heuristicModel, IRankModel* costModel, SearchMetadata searchMetadata)
	{
		clock_t tic = clock();

		// set up priority queues
		// maintain open set for search
		// maintain cost set for returning best cost in the end
		SearchNodeCostPQ costSet;
		SearchNodeHeuristicPQ openSet;

		// push initial state into queue
		ISearchNode* root = createRootNode(searchType, X, YTruth, searchSpace, heuristicModel, costModel);
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

			vector< ISearchNode* > subsetOpenSet = selectSubsetOpenSet(openSet);

			/***** expand these elements *****/

			SearchNodeHeuristicPQ candidateSet = expandElements(subsetOpenSet, openSet, costSet);

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
		ISearchNode* lowestCost = costSet.top();
		ImgLabeling prediction = lowestCost->getY();
		LOG() << endl << "Finished search. Cost=" << lowestCost->getCost() << endl;

		// use best/worst cost set candidates as training examples for cost learning (if applicable)
		if (searchType == LEARN_C || searchType == LEARN_C_ORACLE_H || searchType == LEARN_C_RANDOM_H)
			trainCostRanker(costModel, costSet);

		// clean up cost set
		//deleteQueueElements(costSet);
		vector<double> candidateLosses;
		while (!costSet.empty())
		{
			ISearchNode* state = costSet.top();
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

	ISearchProcedure::ISearchNode* IBasicSearchProcedure::createRootNode(SearchType searchType, ImgFeatures& X, ImgLabeling* YTruth, 
		SearchSpace* searchSpace, IRankModel* heuristicModel, IRankModel* costModel)
	{
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
			case RL:
				root = new RLSearchNode(&X, YTruth, searchSpace);
				break;
			case RC:
				root = new RCSearchNode(&X, searchSpace, costModel);
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
			case LEARN_C_RANDOM_H:
				root = new LearnCRandomHSearchNode(&X, YTruth, searchSpace);
				break;
			default:
				LOG(ERROR) << "searchType constant is invalid.";
		}
		return root;
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

	vector< ISearchProcedure::ISearchNode* > BreadthFirstBeamSearchProcedure::selectSubsetOpenSet(SearchNodeHeuristicPQ& openSet)
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

	ISearchProcedure::SearchNodeHeuristicPQ BreadthFirstBeamSearchProcedure::expandElements(vector< ISearchNode* > subsetOpenSet, SearchNodeHeuristicPQ& openSet, SearchNodeCostPQ& costSet)
	{
		SearchNodeHeuristicPQ candidateSet;
		
		// expand each element
		for (vector< ISearchNode* >::iterator it = subsetOpenSet.begin(); it != subsetOpenSet.end(); ++it)
		{
			SearchNodeHeuristicPQ candidateSetTemp;

			ISearchNode* current = *it;
			LOG() << "Expansion Node: Heuristic=" << current->getHeuristic() << ", Cost=" << current->getCost() << endl;

			vector< ISearchNode* > expansionSet = current->generateSuccessorNodes();

			// only accept expanded element if not a duplicate state
			for (vector< ISearchNode* >::iterator it = expansionSet.begin(); it != expansionSet.end(); ++it)
			{
				ISearchNode* state = *it;
				if (!isDuplicate(state, candidateSet) && !isDuplicate(state, costSet))
				{
					candidateSetTemp.push(state);
				}
			}

			// prune
			int numCandidates = candidateSetTemp.size();
			for (int i = 0; i < PRUNE_MAX_NUM_CANDIDATES; i++)
			{
				if (candidateSetTemp.empty())
					break;

				ISearchNode* state = candidateSetTemp.top();
				candidateSet.push(state);
				candidateSetTemp.pop();
			}
			int numPrunedCandidates = candidateSetTemp.size();
			while (!candidateSetTemp.empty())
			{
				ISearchNode* state = candidateSetTemp.top();
				candidateSetTemp.pop();
				delete state;
			}
			LOG() << "num candidates=" << numCandidates << endl;
			LOG() << "\tnum pruned candidates=" << numPrunedCandidates << endl;
		}

		return candidateSet;
	}

	void BreadthFirstBeamSearchProcedure::chooseSuccessors(SearchType searchType, SearchNodeHeuristicPQ& candidateSet, SearchNodeHeuristicPQ& openSet, SearchNodeCostPQ& costSet, 
			vector< RankFeatures >& bestSet, vector< double >& bestLosses, vector< RankFeatures >& worstSet, vector< double >& worstLosses)
	{
		// pick the top B best states - "good" training examples
		for (int i = 0; i < this->beamSize; i++)
		{
			if (candidateSet.empty())
				break;

			ISearchNode* state = candidateSet.top();
			candidateSet.pop();
			if (searchType == LEARN_H)
			{
				bestSet.push_back(state->getHeuristicFeatures());
				bestLosses.push_back(state->getHeuristic());
			}
			openSet.push(state);
			costSet.push(state);
		}

		// everything else is "bad" for training examples
		while (!candidateSet.empty())
		{
			ISearchNode* state = candidateSet.top();
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

	vector< ISearchProcedure::ISearchNode* > BestFirstBeamSearchProcedure::selectSubsetOpenSet(SearchNodeHeuristicPQ& openSet)
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

	ISearchProcedure::SearchNodeHeuristicPQ BestFirstBeamSearchProcedure::expandElements(vector< ISearchNode* > subsetOpenSet, SearchNodeHeuristicPQ& openSet, SearchNodeCostPQ& costSet)
	{
		SearchNodeHeuristicPQ candidateSet;
		
		// expand each element
		for (vector< ISearchNode* >::iterator it = subsetOpenSet.begin(); it != subsetOpenSet.end(); ++it)
		{
			SearchNodeHeuristicPQ candidateSetTemp;

			ISearchNode* current = *it;
			LOG() << "Expansion Node: Heuristic=" << current->getHeuristic() << ", Cost=" << current->getCost() << endl;

			vector< ISearchNode* > expansionSet = current->generateSuccessorNodes();

			// only accept expanded element if not a duplicate state
			for (vector< ISearchNode* >::iterator it = expansionSet.begin(); it != expansionSet.end(); ++it)
			{
				ISearchNode* state = *it;
				if (!isDuplicate(state, candidateSet) && !isDuplicate(state, costSet))
				{
					candidateSetTemp.push(state);
				}
			}

			// prune
			int numCandidates = candidateSetTemp.size();
			for (int i = 0; i < PRUNE_MAX_NUM_CANDIDATES; i++)
			{
				if (candidateSetTemp.empty())
					break;

				ISearchNode* state = candidateSetTemp.top();
				candidateSet.push(state);
				candidateSetTemp.pop();
			}
			int numPrunedCandidates = candidateSetTemp.size();
			while (!candidateSetTemp.empty())
			{
				ISearchNode* state = candidateSetTemp.top();
				candidateSetTemp.pop();
				delete state;
			}
			LOG() << "num candidates=" << numCandidates << endl;
			LOG() << "\tnum pruned candidates=" << numPrunedCandidates << endl;
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
			case RL:
				{
					RLSearchNode* successor = new RLSearchNode(this, YPred);
					successors.push_back(successor);
				}
				break;
			case RC:
				{
					RCSearchNode* successor = new RCSearchNode(this, YPred);
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
			case LEARN_C_RANDOM_H:
				{
					LearnCRandomHSearchNode* successor = new LearnCRandomHSearchNode(this, YPred);
					successors.push_back(successor);
				}
				break;
			default:
				LOG(ERROR) << "not a valid search type for generating successor";
			}
		}
		return successors;
	}

	RankFeatures ISearchProcedure::ISearchNode::getHeuristicFeatures()
	{
		LOG(ERROR) << "heuristic features are not defined";
		return RankFeatures();
	}

	RankFeatures ISearchProcedure::ISearchNode::getCostFeatures()
	{
		LOG(ERROR) << "cost features are not defined";
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

	/**************** RL Search Node ****************/

	ISearchProcedure::RLSearchNode::RLSearchNode()
	{
	}

	ISearchProcedure::RLSearchNode::RLSearchNode(ImgFeatures* X, ImgLabeling* YTruth, SearchSpace* searchSpace)
	{
		this->parent = NULL;
		this->X = X;
		this->YPred = searchSpace->getInitialPrediction(*X);
		this->searchSpace = searchSpace;

		this->YTruth = YTruth;
		this->heuristic = Rand::unifDist();
		this->loss = searchSpace->computeLoss(this->YPred, *YTruth);
	}

	ISearchProcedure::RLSearchNode::RLSearchNode(ISearchNode* parent, ImgLabeling YPred)
	{
		RLSearchNode* parentCast = dynamic_cast<RLSearchNode*>(parent);

		this->parent = parentCast;
		this->X = parentCast->X;
		this->YPred = YPred;
		this->searchSpace = parentCast->searchSpace;

		this->YTruth = parentCast->YTruth;
		this->heuristic = Rand::unifDist();
		this->loss = this->searchSpace->computeLoss(this->YPred, *this->YTruth);
	}

	double ISearchProcedure::RLSearchNode::getHeuristic()
	{
		return this->heuristic;
	}

	double ISearchProcedure::RLSearchNode::getCost()
	{
		return this->loss;
	}

	SearchType ISearchProcedure::RLSearchNode::getType()
	{
		return RL;
	}

	/**************** RC Search Node ****************/

	ISearchProcedure::RCSearchNode::RCSearchNode()
	{
	}

	ISearchProcedure::RCSearchNode::RCSearchNode(ImgFeatures* X, SearchSpace* searchSpace, IRankModel* costModel)
	{
		this->parent = NULL;
		this->X = X;
		this->YPred = searchSpace->getInitialPrediction(*X);
		this->searchSpace = searchSpace;

		this->costFeatures = searchSpace->computeCostFeatures(*X, this->YPred);
		this->costModel = costModel;
		this->cost = costModel->rank(this->costFeatures);
		this->heuristic = Rand::unifDist();
	}

	ISearchProcedure::RCSearchNode::RCSearchNode(ISearchNode* parent, ImgLabeling YPred)
	{
		RCSearchNode* parentCast = dynamic_cast<RCSearchNode*>(parent);

		this->parent = parentCast;
		this->X = parentCast->X;
		this->YPred = YPred;
		this->searchSpace = parentCast->searchSpace;

		this->costFeatures = this->searchSpace->computeCostFeatures(*this->X, this->YPred);
		this->costModel = parentCast->costModel;
		this->cost = this->costModel->rank(this->costFeatures);
		this->heuristic = Rand::unifDist();
	}

	RankFeatures ISearchProcedure::RCSearchNode::getCostFeatures()
	{
		return this->costFeatures;
	}

	double ISearchProcedure::RCSearchNode::getHeuristic()
	{
		return this->heuristic;
	}

	double ISearchProcedure::RCSearchNode::getCost()
	{
		return this->cost;
	}

	SearchType ISearchProcedure::RCSearchNode::getType()
	{
		return RC;
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

	/**************** Learn C Given Random H Search Node ****************/

	ISearchProcedure::LearnCRandomHSearchNode::LearnCRandomHSearchNode()
	{
	}

	ISearchProcedure::LearnCRandomHSearchNode::LearnCRandomHSearchNode(ImgFeatures* X, ImgLabeling* YTruth, SearchSpace* searchSpace)
	{
		this->parent = NULL;
		this->X = X;
		this->YPred = searchSpace->getInitialPrediction(*X);
		this->searchSpace = searchSpace;

		this->costFeatures = searchSpace->computeCostFeatures(*X, this->YPred);
		this->YTruth = YTruth;
		this->loss = searchSpace->computeLoss(this->YPred, *YTruth);
		this->heuristic = Rand::unifDist();
	}

	ISearchProcedure::LearnCRandomHSearchNode::LearnCRandomHSearchNode(ISearchNode* parent, ImgLabeling YPred)
	{
		LearnCRandomHSearchNode* parentCast = dynamic_cast<LearnCRandomHSearchNode*>(parent);

		this->parent = parentCast;
		this->X = parentCast->X;
		this->YPred = YPred;
		this->searchSpace = parentCast->searchSpace;

		this->costFeatures = this->searchSpace->computeCostFeatures(*this->X, this->YPred);
		this->YTruth = parentCast->YTruth;
		this->loss = this->searchSpace->computeLoss(this->YPred, *this->YTruth);
		this->heuristic = Rand::unifDist();
	}

	RankFeatures ISearchProcedure::LearnCRandomHSearchNode::getCostFeatures()
	{
		return this->costFeatures;
	}

	SearchType ISearchProcedure::LearnCRandomHSearchNode::getType()
	{
		return LEARN_C_RANDOM_H;
	}

	/**************** Compare Search Node ****************/

	bool ISearchProcedure::CompareByHeuristic::operator() (ISearchNode*& lhs, ISearchNode*& rhs) const
	{
		return lhs->getHeuristic() > rhs->getHeuristic();
	}

	bool ISearchProcedure::CompareByCost::operator() (ISearchNode*& lhs, ISearchNode*& rhs) const
	{
		return lhs->getCost() > rhs->getCost();
	}
}