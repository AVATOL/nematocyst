#ifndef SEARCHSPACE_HPP
#define SEARCHSPACE_HPP

#include <vector>
#include "DataStructures.hpp"
#include "FeatureFunction.hpp"
#include "InitialStateFunction.hpp"
#include "SuccessorFunction.hpp"
#include "PruneFunction.hpp"
#include "LossFunction.hpp"

using namespace Eigen;
using namespace std;

namespace HCSearch
{
	/*!
	 * @defgroup SearchSpace Search Space
	 * @brief Provides interfaces for setting up the search space.
	 * @{
	 */

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
		 * Prune function.
		 */
		IPruneFunction* pruneFunction;

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
			IPruneFunction* pruneFunction, ILossFunction* lossFunction);

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
		 * @brief Compute prune features from image features and current labeling.
		 * @param[in] X Structured image features
		 * @param[in] Y Structured output labeling
		 * @return Prune features for classification
		 */
		RankFeatures computePruneFeatures(ImgFeatures& X, ImgLabeling& Y, set<int> action);

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
		vector< ImgCandidate > generateSuccessors(ImgFeatures& X, ImgLabeling& YPred, int timeStep, int timeBound);

		/*!
		 * @brief Generate a list of successors from a current labeling.
		 * @param[in] X Structured image features
		 * @param[in] YPred Current structured output labeling
		 * @return List of successors, which are structured output labelings
		 */
		vector< ImgCandidate > pruneSuccessors(ImgFeatures& X, ImgLabeling& YPred, vector< ImgCandidate >& YCandidates, ImgLabeling* YTruth, ILossFunction* lossFunc);

		/*!
		 * @brief Compute the loss between a predicted labeling and its groundtruth labeling.
		 * @param[in] YPred Predicted structured output labeling
		 * @param[in] YTruth Groundtruth structured output labeling
		 * @return Loss value
		 */
		double computeLoss(ImgLabeling& YPred, const ImgLabeling& YTruth);

		IPruneFunction* getPruneFunction();
		IInitialPredictionFunction* getInitialPredictionFunction();
		ILossFunction* getLossFunction();
	};

	/*! @} */
}

#endif