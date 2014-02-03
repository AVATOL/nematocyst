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
 * <b>%HC-Search</b> is a learning and inference framework for structured prediction. This is an implementation of <b>%HC-Search</b> primarily geared toward scene labeling for computer vision, but may be adapted to other computer vision tasks.
 *
 * The <b>%HC-Search</b> framework allows you to define a search space and search procedure to perform structured prediction. <b>%HC-Search</b> learns a heuristic function to uncover high quality candidates and a cost function to select the best candidate.
 *
 * A search space consists of a heuristic feature function, cost feature function, initial state function, successor function and loss function. A search procedure can be anything that you define, including the commonly used greedy search and beam search. Our implementation allows you to define all of these components.
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
 * @section quickstart_sec Quick Start (Application)
 * 
 * This section shows how to quickly get started by walking through a small demo program. Let $ROOT denote the root directory containing src/.
 * 
 * @subsection setup_subsec Setup
 *
 * 1. Create a folder $ROOT/DataRaw/SomeDataset/. Create additional folders in it: Annotations/, Images/ and Splits/
 * 2. Put the images (.jpg) in the Images/ folder and the groundtruth masks (.jpg) in the Annotations/ folder. Corresponding images and groundtruth files must have the same file name!
 * 3. Create Train.txt, Validation.txt and Test.txt in Splits/. In each file, list the file name in the Images folder (without .jpg extension) that belong in each split you want.
 *
 * @subsection preprocess_subsec Preprocessing
 * 
 * 1. Create folder $ROOT/DataPreprocessed/.
 * 2. Open MATLAB, make sure VLFeat is set up properly (run vl_setup), and run the following command in Matlab: 
 * `preprocess('$ROOT/DataRaw/SomeDataset/Images', '$ROOT/DataRaw/SomeDataset/Annotations', '$ROOT/DataRaw/SomeDataset/Splits', '$ROOT/DataPreprocessed/SomeDataset' )`
 * This should create files and folders in the $ROOT/DataPreprocessed/SomeDataset/ folder.
 *
 * @subsection hcsearch_subsec HC-Search
 *
 * Run the following command from the command line: `./HCSearch $ROOT/DataPreprocessed/SomeDataset $ROOT/Results 5 --learn --infer`
 *
 * This learns a heuristic and cost function and then runs LL/HL/LC/HL search with time bound 5. This should create files and folders in $ROOT/Results/
 * 
 * @subsection postprocess_subsec Postprocessing
 * 
 * In development.
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
 * - VLFeat (<http://www.vlfeat.org/>)
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
 * @page preprocess Preprocessing Modules
 * @brief How to use the MATLAB preprocessing modules to extract features and prepare data for HC-Search.
 * @tableofcontents
 * 
 * The purpose of the preprocessing modules is to convert images and groundtruth into a format for the HC-Search program.
 * 
 * Note: Make sure VLFeat is set up properly before using a preprocessing module. Check out the installation pages.
 * 
 * Let $ROOT denote the root directory containing src/.
 *
 * The main preprocessing module is $ROOT/preprocess/preprocess.m 
 * It accepts three arguments, the path to the images, groundtruth and training/validation/test splits.
 *
 * - Put the images (.jpg) in the images folder and the groundtruth masks (.jp) in the groundtruth folder. Corresponding images and groundtruth files must have the same file name!
 * - Create Train.txt, Validation.txt and Test.txt in the splits folder. In each file, list the file name in the Images folder (without .jpg extension) that belong in each split you want.
 *
 * Preprocess function definition reference:
 * 
 *		function [ allData ] = preprocess( imagesPath, labelsPath, splitsPath, outputPath )
 *		%PREPROCESS Preprocesses folders of images and groundtruth labels into a
 *		%data format for HCSearch to work. Performs feature extraction.
 *		%
 *		%This implementation creates a regular grid of HOG/SIFT patches.
 *		%Therefore only need grayscale images.
 *		%
 *		%One can change the features (e.g. add color) by editing this file.
 *		%
 *		%   imagesPath: folder path to images folder of *.jpg images
 *		%                   e.g. 'DataRaw/SomeDataset/Images'
 *		%   labelsPath: folder path to groundtruth folder of *.jpg label masks
 *		%                   e.g. 'DataRaw/SomeDataset/Groundtruth'
 *		%   splitsPath: folder path that contains Train.txt,
 *		%               Validation.txt, Test.txt
 *		%                   e.g. 'DataRaw/SomeDataset/Splits'
 *		%   outputPath: folder path to output preprocessed data
 *		%                   e.g. 'DataPreprocessed/SomeDataset'
 *		%
 *		%   allData:    data structure containing all preprocessed data
 */

 /*!
 * @page postprocessing Postprocessing Modules
 * @brief How to use the MATLAB postprocessing modules for evaluation and visualization.
 * @tableofcontents
 * 
 * The following are instructions to use the postprocessing modules.
 * 
 * In development.
 */

 /*!
 * @page windows_install Windows Installation Instructions
 * @brief Provides installation instructions for Windows.
 * @tableofcontents
 * 
 * The following are Windows installation instructions.
 * 
 * @section overview_sec Overview
 *
 * Let $ROOT denote the root directory containing src/.
 * 
 * 1. Before compiling the source or running the binary executable, 
 * dependencies must be installed in the external/ directory.
 * 
 *	- Eigen matrix libary
 *		1. Download from http://eigen.tuxfamily.org
 *		2. Unpack to $ROOT/external/Eigen/
 *	- SVM-Rank
 *		1. http://www.cs.cornell.edu/people/tj/svm_light/svm_rank.html
 *		2. Unpack to $ROOT/external/svm_rank/
 *	- LIBLINEAR
 *		1. Download from http://www.csie.ntu.edu.tw/~cjlin/liblinear/
 *		2. Unpack to $ROOT/external/liblinear/
 *	- LIBSVM (optional - if you use SVM for initial state in HC-Search)
 *		1. Download from http://www.csie.ntu.edu.tw/~cjlin/libsvm/
 *		2. Unpack to $ROOT/external/libsvm/
 *	- VLFeat (optional - if you use preprocessing modules)
 *		1. Download from http://www.vlfeat.org/
 *		2. Unpack to $ROOT/external/vlfeat/
 * 
 * 2. The provided binary executable `HCSearch.exe` should now work (make sure it is in $ROOT/external/). 
 * If you prefer to compile from source...
 *
 *	1. Open $ROOT/src/HCSearch.sln in Microsoft Visual Studio 2012 or later.
 *	2. Build the solution. Make sure it is on Release.
 *	3. Move $ROOT/src/Release/HCSearch.exe to $ROOT/HCSearch.exe.
 *
 * 3. To set up VLFeat for the preprocessing modules, follow these instructions:
 *	1. Launch MATLAB. Run the following command in MATLAB: `run('$ROOT/external/vlfeat/toolbox/vl_setsup')` or wherever it is installed.
 *		- Alternatively, add the line to your startup.m file to automatically run this everytime MATLAB starts.
 *	2. Verify is set up properly by running the following command in MATLAB: `vl_version`
 * 
 * @section mpi_sec Installing with MPI (Optional)
 * 
 * To take advantage of parallelism, you need to install MPICH2. 
 * Check out the following website for downloading: 
 * http://www.mpich.org/downloads/
 * 
 * To build from source, make sure the USE_MPI preprocessor flag is defined. 
 * In Microsoft Visual Studio 2012, open up the property pages for each project and edit Preprocessor Definitions.
 *
 * To run, you need to prepend the command with `mpiexec -n NUM_PROCESSORS `. For instance, to run 
 * with 4 parallel processes, run `mpiexec -n 4 ./HCSearch inputDir outputDir timeBound`.
 */

  /*!
 * @page linux_install Linux Installation Instructions
 * @brief Provides installation instructions for Linux.
 * @tableofcontents
 * 
 * The following are Linux installation instructions.
 * 
 * @section overview_sec Overview
 *
 * Let $ROOT denote the root directory containing src/.
 * 
 * 1. Before compiling the source or running the binary executable, 
 * dependencies must be installed in the external/ directory.
 * 
 *	- Eigen matrix libary
 *		1. Download from http://eigen.tuxfamily.org
 *		2. Unpack to $ROOT/external/Eigen/
 *	- SVM-Rank
 *		1. Download from http://www.cs.cornell.edu/people/tj/svm_light/svm_rank.html.
 *		Make sure to download the source code version.
 *		2. Unpack to $ROOT/external/svm_rank/
 *		3. Compile by running `make` in $ROOT/external/svm_rank/
 *	- LIBLINEAR
 *		1. Download from http://www.csie.ntu.edu.tw/~cjlin/liblinear/
 *		2. Unpack to $ROOT/external/liblinear/
 *		3. Compile by running `make` in $ROOT/external/liblinear/
 *	- LIBSVM (optional - if you use SVM for initial state in HC-Search)
 *		1. Download from http://www.csie.ntu.edu.tw/~cjlin/libsvm/
 *		2. Unpack to $ROOT/external/libsvm/
 *		3. Compile by running `make` in $ROOT/external/libsvm/
 *	- VLFeat (optional - if you use preprocessing modules)
 *		1. Download from http://www.vlfeat.org/
 *		2. Unpack to $ROOT/external/vlfeat/
 * 
 * 2. Compile from source by running `make` in $ROOT/.
 *
 * 3. To set up VLFeat for the preprocessing modules, follow these instructions:
 *	1. Launch MATLAB. Run the following command in MATLAB: `run('$ROOT/external/vlfeat/toolbox/vl_setsup')` or wherever it is installed.
 *		- Alternatively, add the line to your startup.m file to automatically run this everytime MATLAB starts.
 *	2. Verify is set up properly by running the following command in MATLAB: `vl_version`
 * 
 * @section mpi_sec Installing with MPI (Optional)
 * 
 * To take advantage of parallelism, you need to install MPICH2. 
 * On Ubuntu and Debian-like systems, you can run the following command:
 * 
 * `sudo apt-get install mpich2`
 * 
 * Alternatively, check out the following website: 
 * http://www.mpich.org/downloads/
 * 
 * To build from source, run `make mpi` instead of `make`.
 * 
 * To run, you need to prepend the command with `mpiexec -n NUM_PROCESSORS `. For instance, to run 
 * with 4 parallel processes, run `mpiexec -n 4 ./HCSearch inputDir outputDir timeBound`.
 */

  /*!
 * @page mac_install Mac OS X Installation Instructions
 * @brief Provides installation instructions for Mac OS X.
 * @tableofcontents
 * 
 * The following are Mac OS X installation instructions.
 *
 * In development.
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
		static void setClasses();
		static set<int> parseList(string str);

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

		/*!
		 * Computes the range of tasks for rank to perform. Used for scheduling parallel processes.
		 * @param[in] rank Rank
		 * @param[in] numTasks Number of tasks
		 * @param[in] numProcesses Number of processes
		 * @param[out] start starting range
		 * @param[out] end ending range (exclusive)
		 */
		static void computeTaskRange(int rank, int numTasks, int numProcesses, int& start, int& end);

	private:
		/*!
		 * Read split file (Train.txt, Validation.txt, Test.txt).
		 * @param[in] filename Path to split file
		 * @return Return a vector of file name strings
		 */
		static vector<string> readSplitsFile(string filename);

		/*!
		 * Read a meta file for a particular image.
		 * @param[in] filename Path to meta file
		 * @param[out] numNodes Number of nodes in the image
		 * @param[out] numFeatures Number of features for each node in the image
		 */
		static void readMetaFile(string filename, int& numNodes, int& numFeatures);

		/*!
		 * Read a nodes file for a particular image.
		 * @param[in] filename Path to nodes file
		 * @param[out] labels Groundtruth labels per node
		 * @param[out] features Image features per node
		 */
		static void readNodesFile(string filename, VectorXi& labels, MatrixXd& features);

		/*!
		 * Read an edges file for a particular image.
		 * @param[in] filename Path to edges file
		 * @param[out] edges Adjacency list for graph edges of the image
		 */
		static void readEdgesFile(string filename, AdjList_t& edges);
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
		 * @param[in] rankerType Rank learner type
		 * @param[in] numIter Number of iterations for each training image
		 * @return Returns the learned heuristic model
		 */
		static IRankModel* learnH(vector< ImgFeatures* >& XTrain, vector< ImgLabeling* >& YTrain, 
			vector< ImgFeatures* >& XValidation, vector< ImgLabeling* >& YValidation, 
			int timeBound, SearchSpace* searchSpace, ISearchProcedure* searchProcedure, RankerType rankerType, int numIter);

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
		 * @param[in] rankerType Rank learner type
		 * @param[in] numIter Number of iterations for each training image
		 * @return Returns the learned cost model
		 */
		static IRankModel* learnC(vector< ImgFeatures* >& XTrain, vector< ImgLabeling* >& YTrain, 
			vector< ImgFeatures* >& XValidation, vector< ImgLabeling* >& YValidation, 
			IRankModel* heuristicModel, int timeBound, SearchSpace* searchSpace, ISearchProcedure* searchProcedure, RankerType rankerType, int numIter);

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
		 * @param[in] rankerType Rank learner type
		 * @param[in] numIter Number of iterations for each training image
		 * @return Returns the learned cost (using oracle H) model
		 */
		static IRankModel* learnCWithOracleH(vector< ImgFeatures* >& XTrain, vector< ImgLabeling* >& YTrain, 
			vector< ImgFeatures* >& XValidation, vector< ImgLabeling* >& YValidation, 
			int timeBound, SearchSpace* searchSpace, ISearchProcedure* searchProcedure, RankerType rankerType, int numIter);

	private:
		static IRankModel* initializeLearning(RankerType rankerType, SearchType searchType);

		static void finishLearning(IRankModel* learningModel, SearchType searchType);
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
}

#endif