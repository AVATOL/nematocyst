#ifndef SEARCHSPACE_HPP
#define SEARCHSPACE_HPP

#include <set>
#include <queue>
#include "../../external/Eigen/Eigen/Dense"
#include "DataStructures.hpp"
#include "MyGraphAlgorithms.hpp"

using namespace Eigen;
using namespace std;

namespace HCSearch
{
	/*!
	 * @defgroup SearchSpace Search Space
	 * @brief Provides interfaces for setting up the search space.
	 * @{
	 */

	/**************** Search Space Function Abstract Definitions ****************/

	/*!
	 * @brief Feature function interface.
	 */
	class IFeatureFunction
	{
	public:
		virtual ~IFeatureFunction() {}

		/*!
		 * @brief Compute features.
		 */
		virtual RankFeatures computeFeatures(ImgFeatures& X, ImgLabeling& Y)=0;

		/*!
		 * @brief Get dimension of computed feature vector given structured features and labeling.
		 */
		virtual int featureSize(ImgFeatures& X, ImgLabeling& Y);
	};

	/*!
	 * @brief Initial prediction function interface
	 */
	class IInitialPredictionFunction
	{
	public:
		virtual ~IInitialPredictionFunction() {}

		/*!
		 * @brief Get initial prediction.
		 */
		virtual ImgLabeling getInitialPrediction(ImgFeatures& X)=0;
	};

	/*!
	 * @brief Successor function interface
	 */
	class ISuccessorFunction
	{
	protected:
		static const int MAX_NUM_SUCCESSOR_CANDIDATES;
		int maxNumSuccessorCandidates;

	public:
		virtual ~ISuccessorFunction() {}
		
		/*!
		 * @brief Generate successors.
		 */
		virtual vector< ImgLabeling > generateSuccessors(ImgFeatures& X, ImgLabeling& YPred)=0;
	};

	/*!
	 * @brief Loss function interface
	 */
	class ILossFunction
	{
	public:
		virtual ~ILossFunction() {}

		/*!
		 * @brief Compute loss.
		 */
		virtual double computeLoss(ImgLabeling& YPred, const ImgLabeling& YTruth)=0;
	};

	/**************** Feature Functions ****************/

	/*!
	 * @brief Standard CRF features with unary and pairwise potentials.
	 */
	class StandardFeatures : public IFeatureFunction
	{
	public:
		StandardFeatures();
		~StandardFeatures();

		virtual RankFeatures computeFeatures(ImgFeatures& X, ImgLabeling& Y);
		virtual int featureSize(ImgFeatures& X, ImgLabeling& Y);

	protected:
		/*!
		 * @brief Compute unary term.
		 */
		virtual VectorXd computeUnaryTerm(ImgFeatures& X, ImgLabeling& Y);
		
		/*!
		 * @brief Compute pairwise term.
		 */
		virtual VectorXd computePairwiseTerm(ImgFeatures& X, ImgLabeling& Y);

		/*!
		 * @brief Compute pairwise features.
		 */
		virtual VectorXd computePairwiseFeatures(VectorXd& nodeFeatures1, VectorXd& nodeFeatures2, 
			int nodeLabel1, int nodeLabel2, int& classIndex);
	};

	/*!
	 * @brief Dense version of standard CRF features with unary and pairwise potentials.
	 */
	class DenseStandardFeatures : public StandardFeatures
	{
	public:
		DenseStandardFeatures();
		~DenseStandardFeatures();

	protected:
		virtual VectorXd computePairwiseTerm(ImgFeatures& X, ImgLabeling& Y);
	};

	/*!
	 * @brief Dense CRF features with unary and pairwise potentials.
	 */
	class DenseCRFFeatures : public IFeatureFunction
	{
	public:
		DenseCRFFeatures();
		~DenseCRFFeatures();

		virtual RankFeatures computeFeatures(ImgFeatures& X, ImgLabeling& Y);
		virtual int featureSize(ImgFeatures& X, ImgLabeling& Y);

	protected:
		virtual VectorXd computeUnaryTerm(ImgFeatures& X, ImgLabeling& Y);
		virtual VectorXd computePairwiseTerm(ImgFeatures& X, ImgLabeling& Y);
		virtual VectorXd computePairwiseFeatures(VectorXd& nodeFeatures1, VectorXd& nodeFeatures2, 
			double nodeLocationX1, double nodeLocationY1, double nodeLocationX2, double nodeLocationY2, 
			int nodeLabel1, int nodeLabel2, int& classIndex);
	};

	/*!
	 * @brief Unary confidences only features.
	 */
	class UnaryConfidencesFeatures : public IFeatureFunction
	{
	public:
		UnaryConfidencesFeatures();
		~UnaryConfidencesFeatures();

		virtual RankFeatures computeFeatures(ImgFeatures& X, ImgLabeling& Y);
		virtual int featureSize(ImgFeatures& X, ImgLabeling& Y);

	protected:
		virtual VectorXd computeUnaryTerm(ImgFeatures& X, ImgLabeling& Y);
	};

	/*!
	 * @brief Interface for standard CRF features plus a global potential.
	 */
	class IGlobalFeatures : public StandardFeatures
	{
	protected:
		/*!
		 * Codebook for bag-of-words.
		 */
		struct CodeBook
		{
			/*!
			 * Codebook data
			 * rows = codewords, cols = histogram features
			 */
			MatrixXd data;
		};

		/*!
		 * Dictionary for global potential.
		 */
		CodeBook dictionary;

	public:
		virtual RankFeatures computeFeatures(ImgFeatures& X, ImgLabeling& Y)=0;

	protected:
		/*!
		 * @brief Compute global term.
		 */
		virtual VectorXd computeGlobalTerm(ImgFeatures& X, ImgLabeling& Y)=0;

		void loadDictionary(string fileName);
	};

	/*!
	 * @brief Standard CRF features with a sum global potential.
	 */
	class SumGlobalFeatures : public IGlobalFeatures
	{
	public:
		/*!
		 * @brief Construct without loading dictionary.
		 */
		SumGlobalFeatures();

		/*!
		 * @brief Construct and load dictionary.
		 */
		SumGlobalFeatures(string fileName);
		
		~SumGlobalFeatures();

		virtual RankFeatures computeFeatures(ImgFeatures& X, ImgLabeling& Y);
		virtual int featureSize(ImgFeatures& X, ImgLabeling& Y);

	protected:
		virtual VectorXd computeGlobalTerm(ImgFeatures& X, ImgLabeling& Y);
	};

	/*!
	 * @brief Standard CRF features with a max global potential.
	 */
	class MaxGlobalFeatures : public IGlobalFeatures
	{
	public:
		/*!
		 * @brief Construct without loading dictionary.
		 */
		MaxGlobalFeatures();

		/*!
		 * @brief Construct and load dictionary.
		 */
		MaxGlobalFeatures(string fileName);

		~MaxGlobalFeatures();

		virtual RankFeatures computeFeatures(ImgFeatures& X, ImgLabeling& Y);
		virtual int featureSize(ImgFeatures& X, ImgLabeling& Y);

	protected:
		virtual VectorXd computeGlobalTerm(ImgFeatures& X, ImgLabeling& Y);
	};

	/**************** Initial Prediction Functions ****************/

	/*!
	 * @brief IID logistic regression initial prediction function.
	 */
	class LogRegInit : public IInitialPredictionFunction
	{
		static const double DEFAULT_C;
		static const double BINARY_CONFIDENCE_THRESHOLD;

	public:
		LogRegInit();
		~LogRegInit();

		virtual ImgLabeling getInitialPrediction(ImgFeatures& X);

		/*!
		 * @brief Train logistic regression model.
		 */
		void trainClassifier();

	protected:
		static void imgfeatures2liblinear(ImgFeatures& X, string filename);
		static void liblinear2imglabeling(ImgLabeling& Y, string filename);
		bool hasForegroundNeighbors(ImgLabeling& Y, int node);

		/*!
		 * @brief Eliminate 1-islands.
		 */
		void eliminateIslands(ImgLabeling& Y);
	};

	/**************** Successor Functions ****************/

	/*!
	 * @brief Deterministic flipbit successor function.
	 *
	 * For each node, flip its label to all possible classes.
	 */
	class FlipbitSuccessor : public ISuccessorFunction
	{
	protected:
		static const int NUM_TOP_LABELS_KEEP;
		static const double BINARY_CONFIDENCE_THRESHOLD;

	public:
		FlipbitSuccessor();
		FlipbitSuccessor(int maxNumSuccessorCandidates);
		~FlipbitSuccessor();
		
		virtual vector< ImgLabeling > generateSuccessors(ImgFeatures& X, ImgLabeling& YPred);
	};

	/*!
	 * @brief Deterministic flipbit successor function using neighbor labels.
	 *
	 * For each node, flip its label to a label of a neighboring node.
	 */
	class FlipbitNeighborSuccessor : public FlipbitSuccessor
	{
	public:
		FlipbitNeighborSuccessor();
		FlipbitNeighborSuccessor(int maxNumSuccessorCandidates);
		~FlipbitNeighborSuccessor();
		
		virtual vector< ImgLabeling > generateSuccessors(ImgFeatures& X, ImgLabeling& YPred);
	};

	/*!
	 * @brief Stochastic successor function.
	 * 
	 * Stochastically cut edges to form subgraphs. 
	 * For each subgraph, flip its label to all possible classes.
	 */
	class StochasticSuccessor : public ISuccessorFunction
	{
	protected:
		static const double TOP_CONFIDENCES_PROPORTION;
		static const double DEFAULT_T_PARM;
		double cutParam; //!< temperature parameter
		bool cutEdgesIndependently; //!< cut independently if true, cut by state otherwise

	public:
		StochasticSuccessor();
		StochasticSuccessor(bool cutEdgesIndependently, double cutParam, int maxNumSuccessorCandidates);
		~StochasticSuccessor();

		virtual vector< ImgLabeling > generateSuccessors(ImgFeatures& X, ImgLabeling& YPred);

	protected:
		virtual MyGraphAlgorithms::SubgraphSet* cutEdges(ImgFeatures& X, ImgLabeling& YPred, double threshold, double T);
		virtual vector< ImgLabeling > createCandidates(ImgLabeling& YPred, MyGraphAlgorithms::SubgraphSet* subgraphs);
		virtual void getLabels(set<int>& candidateLabelsSet, MyGraphAlgorithms::ConnectedComponent* cc);

		void getAllLabels(set<int>& candidateLabelsSet, MyGraphAlgorithms::ConnectedComponent* cc);
		void getNeighborLabels(set<int>& candidateLabelsSet, MyGraphAlgorithms::ConnectedComponent* cc);
		void getConfidencesNeighborLabels(set<int>& candidateLabelsSet, MyGraphAlgorithms::ConnectedComponent* cc);

		static double computeKL(const VectorXd& p, const VectorXd& q);
	};

	/*!
	 * @brief Stochastic successor function using neighbor labels.
	 * 
	 * Stochastically cut edges to form subgraphs. 
	 * For each subgraph, flip its label to a label of a neighboring node.
	 */
	class StochasticNeighborSuccessor : public StochasticSuccessor
	{
	public:
		StochasticNeighborSuccessor();
		StochasticNeighborSuccessor(bool cutEdgesIndependently, double cutParam, int maxNumSuccessorCandidates);
		~StochasticNeighborSuccessor();

	protected:
		virtual void getLabels(set<int>& candidateLabelsSet, MyGraphAlgorithms::ConnectedComponent* cc);
	};

	/*!
	 * @brief Stochastic successor function using confident labels and neighbor labels.
	 * 
	 * Stochastically cut edges to form subgraphs. 
	 * For each subgraph, flip its label to a label of a confident or neighboring node.
	 */
	class StochasticConfidencesNeighborSuccessor : public StochasticSuccessor
	{
	public:
		StochasticConfidencesNeighborSuccessor();
		StochasticConfidencesNeighborSuccessor(bool cutEdgesIndependently, double cutParam, int maxNumSuccessorCandidates);
		~StochasticConfidencesNeighborSuccessor();

	protected:
		virtual void getLabels(set<int>& candidateLabelsSet, MyGraphAlgorithms::ConnectedComponent* cc);
	};

	/*!
	 * @brief Cut schedule successor function.
	 * 
	 * Schedule to find the best cut for forming subgraphs. 
	 * For each subgraph, flip its label to all possible classes.
	 */
	class CutScheduleSuccessor : public StochasticSuccessor
	{
	protected:
		static const int NUM_GOOD_SUBGRAPHS_THRESHOLD;
		static const double FINAL_THRESHOLD;
		static const double THRESHOLD_INCREMENT;

	public:
		CutScheduleSuccessor();
		CutScheduleSuccessor(double cutParam, int maxNumSuccessorCandidates);
		~CutScheduleSuccessor();

		virtual vector< ImgLabeling > generateSuccessors(ImgFeatures& X, ImgLabeling& YPred);

	protected:
		virtual MyGraphAlgorithms::SubgraphSet* cutEdges(ImgFeatures& X, ImgLabeling& YPred, double threshold, double T);
		virtual void getLabels(set<int>& candidateLabelsSet, MyGraphAlgorithms::ConnectedComponent* cc);
	};

	/*!
	 * @brief Cut schedule successor function using neighbor labels.
	 * 
	 * Schedule to find the best cut for forming subgraphs. 
	 * For each subgraph, flip its label to a label of a neighboring node.
	 */
	class CutScheduleNeighborSuccessor : public CutScheduleSuccessor
	{
	public:
		CutScheduleNeighborSuccessor();
		CutScheduleNeighborSuccessor(double cutParam, int maxNumSuccessorCandidates);
		~CutScheduleNeighborSuccessor();

	protected:
		virtual void getLabels(set<int>& candidateLabelsSet, MyGraphAlgorithms::ConnectedComponent* cc);
	};

	/*!
	 * @brief Cut schedule successor function using confident labels and neighbor labels.
	 * 
	 * Schedule to find the best cut for forming subgraphs. 
	 * For each subgraph, flip its label to a label of a confident or neighboring node.
	 */
	class CutScheduleConfidencesNeighborSuccessor : public CutScheduleSuccessor
	{
	public:
		CutScheduleConfidencesNeighborSuccessor();
		CutScheduleConfidencesNeighborSuccessor(double cutParam, int maxNumSuccessorCandidates);
		~CutScheduleConfidencesNeighborSuccessor();

	protected:
		virtual void getLabels(set<int>& candidateLabelsSet, MyGraphAlgorithms::ConnectedComponent* cc);
	};

	/**************** Loss Functions ****************/

	/*!
	 * @brief Hamming loss function.
	 */
	class HammingLoss : public ILossFunction
	{
	public:
		HammingLoss();
		~HammingLoss();

		virtual double computeLoss(ImgLabeling& YPred, const ImgLabeling& YTruth);
	};

	/*!
	 * @brief Pixel Hamming loss function.
	 */
	class PixelHammingLoss : public ILossFunction
	{
	public:
		PixelHammingLoss();
		~PixelHammingLoss();

		virtual double computeLoss(ImgLabeling& YPred, const ImgLabeling& YTruth);
	};

	/**************** Search Space ****************/

	/*!
	 * @brief Defines a search space. 
	 *
	 * SearchSpace is a container that holds the heuristic function, cost function, 
	 * initial prediction function, successor function and loss function, 
	 * all necessary to define a search space.
	 */
	class SearchSpace
	{
	protected:
		/*!
		 * Heuristic feature function.
		 */
		IFeatureFunction* heuristicFeatureFunction;

		/*!
		 * Cost feature function.
		 */
		IFeatureFunction* costFeatureFunction;

		/*!
		 * Initial prediction function.
		 */
		IInitialPredictionFunction* initialPredictionFunction;

		/*!
		 * Successor function.
		 */
		ISuccessorFunction* successorFunction;

		/*!
		 * Loss function.
		 */
		ILossFunction* lossFunction;

	public:
		SearchSpace();

		/*!
		 * @brief Construct a SearchSpace from already defined functions. Only use this constructor.
		 *
		 * Note that the destructor will destroy the objects passed into this constructor!
		 *
		 * @param[in] heuristicFeatureFunction Heuristic feature function
		 * @param[in] costFeatureFunction Cost feature function
		 * @param[in] initialPredictionFunction Initial prediction function
		 * @param[in] successorFunction Successor function
		 * @param[in] lossFunction Loss function
		 */
		SearchSpace(IFeatureFunction* heuristicFeatureFunction, IFeatureFunction* costFeatureFunction,
			IInitialPredictionFunction* initialPredictionFunction, ISuccessorFunction* successorFunction,
			ILossFunction* lossFunction);

		/*!
		 * Note that the destructor will destroy the objects passed into the constructor!
		 */
		~SearchSpace();

		/*!
		 * @brief Compute heuristic features from image features and current labeling.
		 * @param[in] X Structured image features
		 * @param[in] Y Structured output labeling
		 * @return Heuristic features for ranking
		 */
		RankFeatures computeHeuristicFeatures(ImgFeatures& X, ImgLabeling& Y);

		/*!
		 * @brief Compute cost features from image features and current labeling.
		 * @param[in] X Structured image features
		 * @param[in] Y Structured output labeling
		 * @return Cost features for ranking
		 */
		RankFeatures computeCostFeatures(ImgFeatures& X, ImgLabeling& Y);

		/*!
		 * @brief Get the initial labeling from image features.
		 * @param[in] X Structured image features
		 * @return Predicted structured output labeling
		 */
		ImgLabeling getInitialPrediction(ImgFeatures& X);

		/*!
		 * @brief Generate a list of successors from a current labeling.
		 * @param[in] X Structured image features
		 * @param[in] YPred Current structured output labeling
		 * @return List of successors, which are structured output labelings
		 */
		vector< ImgLabeling > generateSuccessors(ImgFeatures& X, ImgLabeling& YPred);

		/*!
		 * @brief Compute the loss between a predicted labeling and its groundtruth labeling.
		 * @param[in] YPred Predicted structured output labeling
		 * @param[in] YTruth Groundtruth structured output labeling
		 * @return Loss value
		 */
		double computeLoss(ImgLabeling& YPred, const ImgLabeling& YTruth);
	};

	/*! @} */
}

#endif