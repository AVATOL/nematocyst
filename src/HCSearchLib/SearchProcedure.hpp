#ifndef SEARCHPROCEDURE_HPP
#define SEARCHPROCEDURE_HPP

#include <vector>
#include "DataStructures.hpp"
#include "SearchSpace.hpp"

namespace HCSearch
{
	/**************** Save Prediction ****************/

	/*!
	 * @brief Convenience functions to save predictions.
	 */
	class SavePrediction
	{
	public:
		/*!
		 * Save the labels of the labeling.
		 */
		static void saveLabels(ImgLabeling& YPred, string fileName);

		/*!
		 * Save the stochastic cuts of the labeling.
		 */
		static void saveCuts(ImgLabeling& YPred, string fileName);

		/*!
		 * Save the labeling as a label mask using the segments data.
		 */
		static void saveLabelMask(ImgFeatures& X, ImgLabeling& YPred, string fileName);

		static void saveCandidateLosses(vector<double>& losses, string fileName);
	};

	class Training
	{
	public:
		static IRankModel* initializeLearning(RankerType rankerType, SearchType searchType);

		static void restartLearning(IRankModel* learningModel, SearchType searchType);

		static void finishLearning(IRankModel* learningModel, SearchType searchType);
	};

	/*!
	 * @defgroup SearchProcedure Search Procedure
	 * @brief Provides an interface for setting up a search procedure.
	 * @{
	 */

	/**************** Search Procedure ****************/

	/*!
	 * @brief Search procedure abstract class. 
	 * Implement a search procedure by extending this class.
	 */
	class ISearchProcedure
	{
	public:
		// Meta data for an instance of search
		struct SearchMetadata
		{
			// save anytime results during search if true
			bool saveAnytimePredictions;

			// train, validation or testing
			DatasetType setType;

			// image example name
			string exampleName;

			// stochastic iteration
			int iter;

		public:
			SearchMetadata();
		};
	
	protected:
		class SearchNode;
		class CompareByHeuristic;
		class CompareByCost;

		typedef priority_queue<SearchNode*, vector<SearchNode*>, CompareByHeuristic> SearchNodeHeuristicPQ;
		typedef priority_queue<SearchNode*, vector<SearchNode*>, CompareByCost> SearchNodeCostPQ;

	public:
		virtual ~ISearchProcedure() {}

		/*!
		 * @brief Search procedure implemented by extending class.
		 * 
		 * Accepts features X and a model (and groudtruth Y if applicable) and performs search.
		 */
		virtual ImgLabeling performSearch(SearchType searchType, ImgFeatures& X, ImgLabeling* YTruth, 
			int timeBound, SearchSpace* searchSpace, IRankModel* heuristicModel, IRankModel* costModel, 
			IRankModel* pruneModel, SearchMetadata searchMetadata)=0;

	protected:
		/*!
		 * @brief Create the root node of the search tree.
		 */
		SearchNode* createRootNode(SearchType searchType, ImgFeatures& X, ImgLabeling* YTruth, 
			SearchSpace* searchSpace, IRankModel* heuristicModel, IRankModel* costModel);

		void saveAnyTimePrediction(ImgLabeling YPred, int timeBound, SearchMetadata searchMetadata, SearchType searchType);
		void trainRanker(IRankModel* ranker, vector< RankFeatures > bestFeatures, vector< double > bestLosses, 
			vector< RankFeatures > worstFeatures, vector< double > worstLosses);
		void trainCostRanker(IRankModel* ranker, SearchNodeCostPQ& costSet);
	};

	/*!
	 * @brief Basic search procedure abstract definition.
	 * Implements a generic search procedure, where you must define the virtual "stubs."
	 */
	class IBasicSearchProcedure : public ISearchProcedure
	{
	public:
		virtual ImgLabeling performSearch(SearchType searchType, ImgFeatures& X, ImgLabeling* YTruth, 
			int timeBound, SearchSpace* searchSpace, IRankModel* heuristicModel, IRankModel* costModel, 
			IRankModel* pruneModel, SearchMetadata searchMetadata);

	protected:
		/*!
		 * @brief Stub for selecting a subset of the open set for processing.
		 */
		virtual vector< SearchNode* > selectSubsetOpenSet(SearchNodeHeuristicPQ& openSet)=0;

		/*!
		 * @brief Stub for expanding the elements.
		 * 
		 * openSet may be modified. costSet is used for duplicate checking.
		 */
		virtual SearchNodeHeuristicPQ expandElements(vector< SearchNode* > subsetOpenSet, SearchNodeHeuristicPQ& openSet, SearchNodeCostPQ& costSet,
			IRankModel* pruneModel, ImgLabeling* YTruth, SearchType searchType, int timeStep, int timeBound)=0;

		/*!
		 * @brief Stub for choosing successors among the expanded.
		 * 
		 * Returns the successors and adds the successors to the openSet and costSet.
		 * 
		 * @post candidateSet becomes empty, openSet contains best states, costSet contains best and worst states
		 */
		virtual void chooseSuccessors(SearchType searchType, SearchNodeHeuristicPQ& candidateSet, SearchNodeHeuristicPQ& openSet, SearchNodeCostPQ& costSet, 
			vector< RankFeatures >& bestSet, vector< double >& bestLosses, vector< RankFeatures >& worstSet, vector< double >& worstLosses)=0;

		/*!
		 * @brief Checks if the state is duplicate among the states in the priority queue.
		 */
		template <class T>
		bool isDuplicate(SearchNode* state, T& pq);

		/*!
		 * @brief Empty priority queue and delete all elements.
		 */
		template <class T>
		void deleteQueueElements(T& queue);
	};

	/**************** Beam Search Procedure ****************/

	/*!
	 * @brief Beam search procedure abstract definition.
	 */
	class IBeamSearchProcedure : public IBasicSearchProcedure
	{
	protected:
		static const int DEFAULT_BEAM_SIZE = 1;

		int beamSize; //!< Beam size
	};

	/**************** Breadth-First Beam Search Procedure ****************/

	/*!
	 * @brief Breadth-first beam search procedure.
	 */
	class BreadthFirstBeamSearchProcedure : public IBeamSearchProcedure
	{
	public:
		BreadthFirstBeamSearchProcedure();
		BreadthFirstBeamSearchProcedure(int beamSize);
		~BreadthFirstBeamSearchProcedure();

		virtual vector< SearchNode* > selectSubsetOpenSet(SearchNodeHeuristicPQ& openSet);
		virtual SearchNodeHeuristicPQ expandElements(vector< SearchNode* > subsetOpenSet, SearchNodeHeuristicPQ& openSet, SearchNodeCostPQ& costSet,
			IRankModel* pruneModel, ImgLabeling* YTruth, SearchType searchType, int timeStep, int timeBound);
		virtual void chooseSuccessors(SearchType searchType, SearchNodeHeuristicPQ& candidateSet, SearchNodeHeuristicPQ& openSet, SearchNodeCostPQ& costSet, 
			vector< RankFeatures >& bestSet, vector< double >& bestLosses, vector< RankFeatures >& worstSet, vector< double >& worstLosses);
	};

	/**************** Best-First Beam Search Procedure ****************/

	/*!
	 * @brief Best-first beam search procedure.
	 */
	class BestFirstBeamSearchProcedure : public BreadthFirstBeamSearchProcedure
	{
	public:
		BestFirstBeamSearchProcedure();
		BestFirstBeamSearchProcedure(int beamSize);
		~BestFirstBeamSearchProcedure();

		virtual vector< SearchNode* > selectSubsetOpenSet(SearchNodeHeuristicPQ& openSet);
		virtual SearchNodeHeuristicPQ expandElements(vector< SearchNode* > subsetOpenSet, SearchNodeHeuristicPQ& openSet, SearchNodeCostPQ& costSet, 
			IRankModel* pruneModel, ImgLabeling* YTruth, SearchType searchType, int timeStep, int timeBound);
	};

	/**************** Greedy Procedure ****************/

	/*!
	 * @brief Greedy search procedure.
	 */
	class GreedySearchProcedure : public ISearchProcedure
	{
	protected:
		typedef vector< SearchNode* > SearchNodeList;

	public:
		GreedySearchProcedure();
		~GreedySearchProcedure();

		virtual ImgLabeling performSearch(SearchType searchType, ImgFeatures& X, ImgLabeling* YTruth, 
			int timeBound, SearchSpace* searchSpace, IRankModel* heuristicModel, IRankModel* costModel, 
			IRankModel* pruneModel, SearchMetadata searchMetadata);

	protected:
		/*!
		 * @brief Stub for expanding the element. 
		 * 
		 * (costSet used for checking duplicates.)
		 * 
		 * @post bestHeuristicNode and bestCostNode updated with the best node, 
		 */
		SearchNodeList expandElements(SearchNode*& bestHeuristicNode, SearchNode*& bestCostNode, SearchNodeList& costSet, 
			IRankModel* pruneModel, ImgLabeling* YTruth, SearchType searchType, int timeStep, int timeBound, int& numOutputs);

		/*!
		 * @brief Stub for choosing successors among the expanded.
		 * 
		 * Sorts the list of search nodes into best and worst sets
		 */
		void sortNodes(SearchType searchType, SearchNodeList& candidateSet, 
			vector< RankFeatures >& bestSet, vector< double >& bestLosses, vector< RankFeatures >& worstSet, vector< double >& worstLosses);

		/*!
		 * @brief Checks if the state is duplicate among the states in the priority queue.
		 */
		bool isDuplicate(SearchNode* state, SearchNodeList& list);
	};

	/*! @} */



	/**************** Search Node ****************/

	class ISearchProcedure::SearchNode
	{
	protected:
		SearchNode* parent; //!< Pointer to parent node
		SearchSpace* searchSpace; //!< Pointer to search space
		SearchType searchType; //!< Search type

		ImgFeatures* X; //!< Structured features of node
		ImgLabeling YPred; //<! Structured labeling of node
		ImgLabeling* YTruth; //!< Pointer to groundtruth labeling

		RankFeatures heuristicFeatures; //!< Heuristic features
		IRankModel* heuristicModel; //!< Heuristic model
		double heuristic; //!< Heuristic value

		RankFeatures costFeatures; //!< Cost features
		IRankModel* costModel; //!< Cost model
		double cost; //!< Cost value

	public:
		SearchNode(); // do not use default constructor

		/*!
		 * Node initialization constructor.
		 */
		SearchNode(ImgFeatures* X, ImgLabeling* YTruth, SearchSpace* searchSpace, 
			IRankModel* heuristicModel, IRankModel* costModel, SearchType searchType);

		/*!
		 * Node generation constructor.
		 */
		SearchNode(SearchNode* parent, ImgLabeling YPred);

		/*!
		 * Generate successor nodes.
		 */
		vector< SearchNode* > generateSuccessorNodes(bool prune, int timeStep, int timeBound, ImgLabeling* YTruth);

		vector< SearchNode* > generateSuccessorNodesForPruneLearning(IRankModel* learningModel, 
			ImgLabeling* YTruth, int timeStep, int timeBound);

		/*!
		 * Get the heuristic features of the node. 
		 * May not be defined depending on search type.
		 */
		virtual RankFeatures getHeuristicFeatures();

		/*!
		 * Get the cost features of the node. 
		 * May not be defined depending on search type.
		 */
		virtual RankFeatures getCostFeatures();

		/*!
		 * Get the heuristic value. 
		 * (From heuristic weights or loss function depending on search type.)
		 */
		virtual double getHeuristic();

		/*!
		 * Get the cost value. 
		 * (From cost weights or loss function depending on search type.)
		 */
		virtual double getCost();

		/*!
		 * Get the labeling of the node.
		 */
		ImgLabeling getY();

	protected:
		/*!
		 * Return type of search node.
		 */
		SearchType getType();

	private:
		void constructorHelper();
	};

	/**************** Compare Search Node ****************/

	class ISearchProcedure::CompareByHeuristic
	{
	public:
		bool operator() (SearchNode*& lhs, SearchNode*& rhs) const;
	};

	class ISearchProcedure::CompareByCost
	{
	public:
		bool operator() (SearchNode*& lhs, SearchNode*& rhs) const;
	};

	/**************** Template definitions ****************/

	template <class T>
	bool IBasicSearchProcedure::isDuplicate(SearchNode* state, T& pq)
	{
		int size = pq.size();
		bool isDuplicate = false;

		T temp;

		for (int i = 0; i < size; i++)
		{
			SearchNode* current = pq.top();
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

	template <class T>
	void IBasicSearchProcedure::deleteQueueElements(T& queue)
	{
		while (!queue.empty())
		{
			SearchNode* state = queue.top();
			queue.pop();
			delete state;
		}
	}
}

#endif