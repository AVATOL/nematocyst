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
	class CompareConfidences;
	typedef MyPrimitives::Pair<double, int> ConfidenceIndexPair_t;
	typedef priority_queue<ConfidenceIndexPair_t, vector< ConfidenceIndexPair_t >, CompareConfidences> mypq_confidences;

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
		static VectorXd computeUnaryTerm(ImgFeatures& X, ImgLabeling& Y);
		
		/*!
		 * @brief Compute pairwise term.
		 */
		static VectorXd computePairwiseTerm(ImgFeatures& X, ImgLabeling& Y);

		/*!
		 * @brief Compute pairwise features.
		 */
		static VectorXd computePairwiseFeatures(VectorXd& nodeFeatures1, VectorXd& nodeFeatures2, 
			int nodeLabel1, int nodeLabel2, int& classIndex);
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
		/*!
		 * @brief Construct without training classifier.
		 */
		LogRegInit();

		/*!
		 * @brief Construct and train classifier.
		 */
		LogRegInit(string fileName);
		
		~LogRegInit();

		virtual ImgLabeling getInitialPrediction(ImgFeatures& X);

		/*!
		 * @brief Train logistic regression model.
		 */
		void trainClassifier(string fileName);

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
	 * For each node, flip its label to a label of a neighboring node.
	 */
	class FlipbitSuccessor : public ISuccessorFunction
	{
		static const int NUM_TOP_LABELS_KEEP;
		static const double BINARY_CONFIDENCE_THRESHOLD;

	public:
		FlipbitSuccessor();
		~FlipbitSuccessor();
		
		virtual vector< ImgLabeling > generateSuccessors(ImgFeatures& X, ImgLabeling& YPred);
	};

	/*!
	 * @brief Stochastic successor function.
	 */
	class StochasticSuccessor : public ISuccessorFunction
	{
		static const double DEFAULT_T_PARM;
		double cutParam; //!< temperature parameter
		bool cutEdgesIndependently; //!< cut independently if true, cut by state otherwise

	public:
		StochasticSuccessor();
		StochasticSuccessor(double cutParam);
		~StochasticSuccessor();

		virtual vector< ImgLabeling > generateSuccessors(ImgFeatures& X, ImgLabeling& YPred);

	protected:
		MyGraphAlgorithms::SubgraphSet* cutEdges(ImgFeatures& X, ImgLabeling& YPred, double threshold, double T);
		vector< ImgLabeling > createCandidates(ImgLabeling& YPred, MyGraphAlgorithms::SubgraphSet* subgraphs);
		static double computeKL(const VectorXd& p, const VectorXd& q);
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

	/*
	 * @brief Compare class for SearchNode.
	 */
	class CompareConfidences
	{
	public:
		bool operator() (ConfidenceIndexPair_t& lhs, ConfidenceIndexPair_t& rhs) const
		{
			return lhs.first < rhs.first;
		}
	};
}

#endif