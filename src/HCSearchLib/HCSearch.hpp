#ifndef HCSEARCH_HPP
#define HCSEARCH_HPP

#include <vector>

#include "Settings.hpp"
#include "MyLogger.hpp"
#include "Globals.hpp"
#include "DataStructures.hpp"
#include "SearchSpace.hpp"
#include "SearchProcedure.hpp"

using namespace std;

/*!
 * @mainpage Documentation
 * @tableofcontents
 * 
 * @section intro_sec Introduction
 * 
 * This document describes the design and usage of the <b>%HC-Search</b> framework for computer vision.
 * 
 * Check out the API Overview for a basic overview of the API functions available, or look at the Modules section to go more in depth.
 * 
 * @section overview_sec HC-Search Overview
 * 
 * This section provides a brief overview about the HC-Search framework.
 * 
 * @section install_sec Downloading and Installing
 * 
 * <b>%HC-Search</b> depends on a few dependencies, but they should be relatively easy to install. 
 * The following are detailed installation instructions for Windows, Linux and Mac OS X:
 *		- @ref windows_install
 *		- @ref linux_install
 *		- @ref mac_install
 *
 * Stable releases can be found on GitHub: https://github.com/mqtlam/hcsearchcv/releases
 * 
 * The source code and project can also be found on GitHub: https://github.com/mqtlam/hcsearchcv
 * 
 * @section quickstart_sec Quick Start
 * 
 * This section shows how to quickly get started by walking through a small demo program.
 * 
 * @section api_sec API Overview
 * 
 * @ref api_overview "Check out the API Overview."
 * 
 * @section external_sec External Documentation
 *
 * - Eigen matrix library (<http://eigen.tuxfamily.org>)
 * - SVM-Rank (<http://www.cs.cornell.edu/people/tj/svm_light/svm_rank.html>)
 * - LIBLINEAR (<http://www.csie.ntu.edu.tw/~cjlin/liblinear/>)
 * - LIBSVM (<http://www.csie.ntu.edu.tw/~cjlin/libsvm/>)
 * - C++ STL (<http://www.sgi.com/tech/stl>)
 * 
 */

/*!
 * @page api_overview API Overview
 * @brief Provides an overview of the API functions and how to put them together.
 * @tableofcontents
 * 
 * The following are steps to loading the dataset, learning models and performing inference:
 * 
 * @section summary_sec Summary
 * 
 * 1. Must call HCSearch::Setup::initialize() and HCSearch::Setup::configure() before doing anything. 
 * Load dataset using HCSearch::Dataset::loadDataset() and load models using HCSearch::Model::loadModel() if necessary. (Jump to @ref setup_sec and @ref util_sec)
 * 
 * 2. Define a search space using HCSearch::SearchSpace and search procedure using HCSearch::ISearchProcedure. (Jump to @ref searchspace_sec and @ref searchprocedure_sec)
 *
 * 3. Perform learning using HCSearch::Learning and inference using HCSearch::Inference. (Jump to @ref learn_sec and @ref infer_sec)
 * 
 * 4. Clean up using HCSearch::Dataset::unloadDataset() and HCSearch::Setup::finalize(). (Jump to @ref setup_sec)
 * 
 * @section setup_sec Setup and Teardown
 * 
 * 1. At the beginning of the program, you must call HCSearch::Setup::initialize().
 *
 * 2. After calling HCSearch::Setup::initialize() and before calling any HCSearch-related methods, 
 * you must call HCSearch::Setup::configure() to set up the input and output paths.
 * 
 * 3. You must call HCSearch::Setup::finalize() to release resources right before the program exits.
 * 
 * @section util_sec Dataset and Model Utilities
 * 
 * 1. Use HCSearch::Dataset::loadDataset() to load in the preprocessed dataset for the HC-Search program. 
 * Right before exiting the program, use HCSearch::Dataset::unloadDataset() to clean up dataset resources.
 * 
 * 2. To save a heuristic or cost model, use the HCSearch::Model::saveModel(). 
 * To load a model for next time, use the HCSearch::Model::loadModel().
 * 
 * @section searchspace_sec Search Space
 * 
 * Define a search space by defining a heuristic feature function, cost feature function, 
 * initial prediction function, successor function and loss function, then construct an HCSearch::SearchSpace object:
 *		1. Define a feature function by extending HCSearch::IFeatureFunction and constructing that object.
 *				- To use standard CRF features, simply construct an object from the provided HCSearch::StandardFeatures class.
 *		2. Define a initial prediction function by extending HCSearch::IInitialPredictionFunction and constructing that object.
 *				- To use IID logistic regression, simply construct an object from the provided HCSearch::LogRegInit class.
 *		3. Define a successor function by extending HCSearch::ISuccessorFunction and constructing that object.
 *				- To use flipbit successor, simply construct an object from the provided HCSearch::FlipbitSuccessor class.
 *				- To use stochastic successor, simply construct an object from the provided HCSearch::StochasticSuccessor class.
 *		4. Define a loss function by extending HCSearch::ILossFunction and constructing that object.
 *				- To use Hamming loss, simply construct an object from the provided HCSearch::HammingLoss class.
 *		5. Pass these function objects into the HCSearch::SearchSpace constructor.
 * 
 * @section searchprocedure_sec Search Procedure
 * 
 * Define a search procedure by extending HCSearch::ISearchProcedure and constructing that object.
 *		- To use greedy search, simply construct an object from the provided HCSearch::GreedySearchProcedure class.
 *		- To use breadth first beam search, simply construct an object from the provided HCSearch::BreadthFirstBeamSearchProcedure class.
 * 
 * @section learn_sec Learning
 * 
 * Before learning you must define the search space and search procedure. Use a function from Learning to perform learning:
 *		- To learn a heuristic function, use HCSearch::Learning::learnH().
 *		- To learn a cost function given the learned heuristic function, use HCSearch::Learning::learnC().
 *		- To learn a cost function given an oracle heuristic (loss), use HCSearch::Learning::learnCWithOracleH().
 * 
 * @section infer_sec Inference
 * 
 * Before inference you must define the search space and search procedure. Use a function from Inference to perform inference:
 *		- To run LL-search, use HCSearch::Inference::runLLSearch().
 *		- To run HL-search, use HCSearch::Inference::runHLSearch(). Requires heuristic model.
 *		- To run LC-search, use HCSearch::Inference::runLCSearch(). Requires cost model trained from oracle H.
 *		- To run HC-search, use HCSearch::Inference::runHCSearch(). Requires heuristic model and cost model trained from the heuristic.
 */

 /*!
 * @page windows_install Windows Installation Instructions
 * @brief Provides installation instructions for Windows.
 * @tableofcontents
 * 
 * The following are Windows installation instructions.
 */

  /*!
 * @page linux_install Linux Installation Instructions
 * @brief Provides installation instructions for Linux.
 * @tableofcontents
 * 
 * The following are Linux installation instructions.
 */

  /*!
 * @page mac_install Mac OS X Installation Instructions
 * @brief Provides installation instructions for Mac OS X.
 * @tableofcontents
 * 
 * The following are Mac OS X installation instructions.
 */

/////////////////////////////////////////////////////////
/// This file contains the API functions for HC-Search
/////////////////////////////////////////////////////////

namespace HCSearch
{
	/**************** Initialize/Finalize ****************/
	
	/*!
	 * @defgroup Setup Setup
	 * @brief Required calls for setting up and tearing down. 
	 * 
	 * 1. Must call Setup::initialize() before running any HC-Search functions, 
	 * including from the Setup, Learning and Inference modules. 
	 * If using MPI, Setup::initialize() must be the first call in main() of the program!
	 *
	 * 2. Must call Setup::configure() sometime after Setup::initialize() and sometime before 
	 * other HC-Search functions. This sets up the proper paths for the dataset and output.
	 *
	 * 3. Must call Setup::finalize() before exiting the program to clean up 
	 * resources. This is especially important if using MPI, otherwise a run-time 
	 * error will be thrown.
	 * @{
	 */

	/*!
	 * @brief Required interfaces for setting up and tearing down. 
	 * 
	 * These are necessary to set up global variables and optionally MPI stuff.
	 */
	class Setup
	{
	public:
		/*!
		 * Initialize HC-Search state. 
		 * This must be the first call in main() of the program! 
		 */
		static void initialize(int argc, char* argv[]);

		/*!
		 * Configures settings. 
		 * Must be called sometime after Setup::initialize() 
		 * and sometime before calling any HCSearch-related functions.
		 */
		static void configure(string datasetPath, string outputPath);

		/*!
		 * Must be called right before exiting program! 
		 * Cleans up resources and finalizes MPI if used.
		 */
		static void finalize();
		
	private:
		static void initializeHelper();
		static void finalizeHelper();

#ifdef USE_MPI
		/*!
		 * Forces all processes to sync up wherever the function is called.
		 */
		static void synchronize_MPI();
#endif
	};

	/*! @}*/

	/**************** Dataset ****************/

	/*!
	 * @defgroup Dataset Dataset Utilities
	 * @brief Provide utilities to load/unload datasets.
	 *
	 * - Use the Dataset::loadDataset() 
	 * function to read in a preprocessed dataset. 
	 *
	 * - Use the Dataset::unloadDataset() 
	 * function to clean up dataset resources properly before exiting the program.
	 * @{
	 */

	/*!
	 * @brief Interfaces for loading/unloading datasets.
	 */
	class Dataset
	{
	public:
		/*!
		 * Load preprocessed dataset. 
		 * Must call Utility::configure() first to set up dataset path.
		 * @param[out] XTrain Vector of structured features for training
		 * @param[out] YTrain Vector of structured labelings for training
		 * @param[out] XValidation Vector of structured features for validation
		 * @param[out] YValidation Vector of structured labelings for validation
		 * @param[out] XTest Vector of structured features for test
		 * @param[out] YTest Vector of structured labelings for test
		 */
		static void loadDataset(vector< ImgFeatures* >& XTrain, vector< ImgLabeling* >& YTrain, 
			vector< ImgFeatures* >& XValidation, vector< ImgLabeling* >& YValidation, 
			vector< ImgFeatures* >& XTest, vector< ImgLabeling* >& YTest);

		/*!
		 * Clean up dataset objects: delete and empty.
		 * @param[out] XTrain Vector of structured features for training
		 * @param[out] YTrain Vector of structured labelings for training
		 * @param[out] XValidation Vector of structured features for validation
		 * @param[out] YValidation Vector of structured labelings for validation
		 * @param[out] XTest Vector of structured features for test
		 * @param[out] YTest Vector of structured labelings for test
		 */
		static void unloadDataset(vector< ImgFeatures* >& XTrain, vector< ImgLabeling* >& YTrain, 
			vector< ImgFeatures* >& XValidation, vector< ImgLabeling* >& YValidation, 
			vector< ImgFeatures* >& XTest, vector< ImgLabeling* >& YTest);
	};

	/*! @} */

	/**************** Model ****************/

	/*!
	 * @defgroup Model Model Utilities
	 * @brief Provide utilities to load/save learned models.
	 *
	 * - Use the Utility::loadModel() function to load a model file.
	 *
	 * - Use the Utility::saveModel() to save a model to file.
	 *
	 * @{
	 */

	/*!
	 * @brief Interfaces for loading/saving learned models.
	 */
	class Model
	{
	public:
		/*!
		 * Load single model from file. 
		 * Can be learned H, C, or C from oracle H.
		 * @param[in] fileName File name of model to read from
		 * @param[in] rankerType Type of ranking model
		 * @return Return a RankModel
		 */
		static IRankModel* loadModel(string fileName, RankerType rankerType);

		/*!
		 * Save model to file.
		 * @param[in] model Rank model to save
		 * @param[in] fileName File name of model to save to
		 * @param[in] rankerType Type of ranking model
		 */
		static void saveModel(IRankModel* model, string fileName, RankerType rankerType);
	};

	/*! @} */

	/**************** Learning ****************/
	
	/*!
	 * @defgroup Learning Learning
	 * @brief Provides an interface for learning heuristic, cost with learned heuristic and cost with oracle heuristic.
	 *
	 * - To learn a heuristic function given a search space definition and search procedure, 
	 * use Learning::learnH().
	 *
	 * - To learn a cost function given a search space definition, search procedure and <em>learned heuristic function</em>, 
	 * use Learning::learnC().
	 *
	 * - To learn a cost function given a search space definition, search procedure and <em>oracle heuristic (loss) function</em>,
	 * use Learning::learnCWithOracleH().
	 * @{
	 */

	/*!
	 * @brief Interfaces for learning heuristic, cost with learned heuristic and cost with oracle heuristic.
     */
	class Learning
    {
    public:
		/*!
		 * Learn heuristic function. 
		 * Given training data, validation data, time bound, search space and procedure, 
		 * learn a heuristic model and return it.
		 * @param[in] XTrain Vector of structured features for training
		 * @param[in] YTrain Vector of structured labelings for training
		 * @param[in] XValidation Vector of structured features for validation
		 * @param[in] YValidation Vector of structured labelings for validation
		 * @param[in] timeBound  Time bound for learning
		 * @param[in] searchSpace Search space definition
		 * @param[in] searchProcedure Search procedure
		 * @param[in] SearchMetadata Meta information
		 * @return Returns the learned heuristic model
		 */
		static IRankModel* learnH(vector< ImgFeatures* >& XTrain, vector< ImgLabeling* >& YTrain, 
			vector< ImgFeatures* >& XValidation, vector< ImgLabeling* >& YValidation, 
			int timeBound, SearchSpace* searchSpace, ISearchProcedure* searchProcedure);

		/*!
		 * Learn cost function given learned heuristic function. 
		 * Given training data, validation data, time bound, search space, procedure and learned heuristic, 
		 * learn a cost model and return it.
		 * @param[in] XTrain Vector of structured features for training
		 * @param[in] YTrain Vector of structured labelings for training
		 * @param[in] XValidation Vector of structured features for validation
		 * @param[in] YValidation Vector of structured labelings for validation
		 * @param[in] heuristicModel Learned heuristic model
		 * @param[in] timeBound  Time bound for learning
		 * @param[in] searchSpace Search space definition
		 * @param[in] searchProcedure Search procedure
		 * @param[in] SearchMetadata Meta information
		 * @return Returns the learned cost model
		 */
		static IRankModel* learnC(vector< ImgFeatures* >& XTrain, vector< ImgLabeling* >& YTrain, 
			vector< ImgFeatures* >& XValidation, vector< ImgLabeling* >& YValidation, 
			IRankModel* heuristicModel, int timeBound, SearchSpace* searchSpace, ISearchProcedure* searchProcedure);

		/*!
		 * Learn cost function given oracle heuristic function.
		 * Given training data, validation data, time bound, search space and procedure, 
		 * learn a cost model using an oracle H and return it.
		 * @param[in] XTrain Vector of structured features for training
		 * @param[in] YTrain Vector of structured labelings for training
		 * @param[in] XValidation Vector of structured features for validation
		 * @param[in] YValidation Vector of structured labelings for validation
		 * @param[in] timeBound  Time bound for learning
		 * @param[in] searchSpace Search space definition
		 * @param[in] searchProcedure Search procedure
		 * @param[in] SearchMetadata Meta information
		 * @return Returns the learned cost (using oracle H) model
		 */
		static IRankModel* learnCWithOracleH(vector< ImgFeatures* >& XTrain, vector< ImgLabeling* >& YTrain, 
			vector< ImgFeatures* >& XValidation, vector< ImgLabeling* >& YValidation, 
			int timeBound, SearchSpace* searchSpace, ISearchProcedure* searchProcedure);
    };

	/*! @} */

	/**************** Inference ****************/

	/*!
	 * @defgroup Inference Inference
	 * @brief Provides an interface to run inference: LL-search, HL-search, LC-search or HC-search.
	 *
	 * - To run search with the loss function (oracle) as both the heuristic and cost functions, 
	 * use Inference::runLLSearch().
	 *
	 * - To run search with a learned heuristic for the heuristic function and loss function (oracle) as the cost function, 
	 * use Inference::runHLSearch().
	 *
	 * - To run search with a loss function as the heuristic function (oracle) and learned cost function (using the oracle heuristic) as the cost function, 
	 * use Inference::runLCSearch().
	 *
	 * - To run search with a learned heuristic for the heuristic function and learned cost function (using the learned heuristic) as the cost function, 
	 * use Inference::runHCSearch().
	 * @{
	 */

	/*!
	 * @brief Interfaces to run inference: LL-search, HL-search, LC-search or HC-search.
     */
	class Inference
    {
    public:
		/*!
		 * Run LL-search (oracle heuristic, oracle cost).
		 * @param[in] X Input structured features
		 * @param[in] YTruth Groundtruth structured labeling
		 * @param[in] timeBound Time bound for learning
		 * @param[in] searchSpace Search space definition
		 * @param[in] searchProcedure Search procedure
		 * @param[in] SearchMetadata Meta information
		 * @return Inference labeling
		 */
		static ImgLabeling runLLSearch(ImgFeatures* X, ImgLabeling* YTruth, 
			int timeBound, SearchSpace* searchSpace, ISearchProcedure* searchProcedure, 
			ISearchProcedure::SearchMetadata searchMetadata);

		/*!
		 * Run HL-search (learned heuristic, oracle cost).
		 * @param[in] X Input structured features
		 * @param[in] YTruth Groundtruth structured labeling
		 * @param[in] timeBound Time bound for learning
		 * @param[in] searchSpace Search space definition
		 * @param[in] searchProcedure Search procedure
		 * @param[in] heuristicModel Learned heuristic model
		 * @param[in] SearchMetadata Meta information
		 * @return Inference labeling
		 */
		static ImgLabeling runHLSearch(ImgFeatures* X, ImgLabeling* YTruth, 
			int timeBound, SearchSpace* searchSpace, ISearchProcedure* searchProcedure,
			IRankModel* heuristicModel, ISearchProcedure::SearchMetadata searchMetadata);

		/*!
		 * Run LC-search (oracle heuristic, learned cost).
		 * @param[in] X Input structured features
		 * @param[in] YTruth Groundtruth structured labeling
		 * @param[in] timeBound Time bound for learning
		 * @param[in] searchSpace Search space definition
		 * @param[in] searchProcedure Search procedure
		 * @param[in] costOracleHModel Learned cost (using oracle H) model
		 * @param[in] SearchMetadata Meta information
		 * @return Inference labeling
		 */
		static ImgLabeling runLCSearch(ImgFeatures* X, ImgLabeling* YTruth, 
			int timeBound, SearchSpace* searchSpace, ISearchProcedure* searchProcedure,
			IRankModel* costOracleHModel, ISearchProcedure::SearchMetadata searchMetadata);

		/*!
		 * Run HC-search (learned heuristic, learned cost).
		 * @param[in] X Input structured features
		 * @param[in] timeBound Time bound for learning
		 * @param[in] searchSpace Search space definition
		 * @param[in] searchProcedure Search procedure
		 * @param[in] heuristicModel Learned heuristic model
		 * @param[in] costModel Learned cost model
		 * @param[in] SearchMetadata Meta information
		 * @return Inference labeling
		 */
		static ImgLabeling runHCSearch(ImgFeatures* X, int timeBound, 
			SearchSpace* searchSpace, ISearchProcedure* searchProcedure,
			IRankModel* heuristicModel, IRankModel* costModel, 
			ISearchProcedure::SearchMetadata searchMetadata);
    };

	/*! @} */

	/**************** Miscellaneous ****************/

	/*!
	 * @brief Abort program cleanly. Especially if using MPI.
	 */
	void abort();

	/*!
	 * @brief Abort program cleanly with error code. Especially if using MPI.
	 */
	void abort(int errcode);
}

#endif