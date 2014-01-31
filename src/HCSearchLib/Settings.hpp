#ifndef SETTINGS_H
#define SETTINGS_H

/**************** Enable/Disable MPI ****************/

// comment out below if not using MPI
//#define USE_MPI

/**************** End Enable/Disable MPI ****************/

#include <string>

#ifdef USE_MPI
#include <mpi.h>
#endif

#include "MyPrimitives.hpp"

using namespace std;

namespace HCSearch
{
	/**************** Class Map ****************/

	// Stores class labels
	// Also stores mapping to class index, as used by confidences
	class ClassMap
	{
	protected:
		static const int DEFAULT_BACKGROUND_LABEL;
		
		MyPrimitives::Bimap<int> allClasses; // class index <-> class label
		map<int, bool> backgroundClasses; // label -> is background? true/false

		int numBackground; // num of background classes (cached)
		int backgroundLabel; // background label
		bool backgroundExists; // if background class even exists

	public:
		ClassMap();
		~ClassMap();

		int numClasses();
		int numBackgroundClasses();

		int getClassIndex(int classLabel);
		int getClassLabel(int classIndex);

		bool classIndexIsBackground(int classIndex);
		bool classLabelIsBackground(int classLabel);

		set<int> getLabels();
		set<int> getBackgroundLabels();
		set<int> getForegroundLabels();

		int getBackgroundLabel();
	};

	/**************** Directory/File Paths Class ****************/

	/*!
	 * @brief Store directory/file paths.
	 */
	class Paths
	{
	public:
		// misc

		string DIR_SEP;

		// basic directories

		string BASE_PATH;
		string EXTERNAL_DIR;
		string DATA_DIR;
		string EXPERIMENT_DIR;

		// external directories

		string LIBLINEAR_DIR;
		string LIBSVM_DIR;
		string SVMRANK_DIR;
		string RANKLIB_DIR;

		// data directories

		string DATA_ALLDATA_DIR;
		string DATA_INITFUNC_DIR;
		string DATA_METADATA_FILE;
		string DATA_CODEBOOK_FILE;
		string DATA_INITFUNC_TRAINING_FILE;
		string DATA_PAIRWISE_TRAINING_FILE;

		// data directories: file name bases

		string DATA_NODES_FILE_BASE;
		string DATA_EDGES_FILE_BASE;
		string DATA_SEGMENTS_FILE_BASE;

		// experiment directories

		string EXPERIMENT_LOGS_DIR;
		string EXPERIMENT_MODELS_DIR;
		string EXPERIMENT_RESULTS_DIR;
		string EXPERIMENT_TEMP_DIR;

		string EXPERIMENT_HEURISTIC_FEATURES_FILE;
		string EXPERIMENT_COST_H_FEATURES_FILE;
		string EXPERIMENT_COST_L_FEATURES_FILE;

		string EXPERIMENT_HEURISTIC_ONLINE_WEIGHTS_FILE;
		string EXPERIMENT_COST_H_ONLINE_WEIGHTS_FILE;
		string EXPERIMENT_COST_L_ONLINE_WEIGHTS_FILE;

		string EXPERIMENT_HEURISTIC_MODEL_FILE;
		string EXPERIMENT_COST_H_MODEL_FILE;
		string EXPERIMENT_COST_L_MODEL_FILE;

		string EXPERIMENT_LOG_FILE;

		string EXPERIMENT_INITFUNC_MODEL_FILE;
		string EXPERIMENT_INITFUNC_FEATURES_FILE;
		string EXPERIMENT_INITFUNC_PREDICT_FILE;

		string EXPERIMENT_PAIRWISE_MODEL_FILE;
		string EXPERIMENT_PAIRWISE_FEATURES_FILE;
		string EXPERIMENT_PAIRWISE_PREDICT_FILE;

		string EXPERIMENT_RANKLIB_FEATURES_FILE;
		string EXPERIMENT_RANKLIB_SCORE_FILE;

	public:
		Paths();
		~Paths();
	};

	/**************** Commands Class ****************/

	/*!
	 * @brief Store commands.
	 */
	class Commands
	{
	protected:
		Paths* paths;

	public:
		string SYSTEM_COPY_CMD;
		string SYSTEM_MKDIR_CMD;
		string SYSTEM_RM_CMD;
		
		string LIBLINEAR_PREDICT_CMD;
		string LIBLINEAR_TRAIN_CMD;
		
		string LIBSVM_PREDICT_CMD;
		string LIBSVM_TRAIN_CMD;

		string SVMRANK_LEARN_CMD;

		string RANKLIB_CMD;

	public:
		Commands();
		Commands(Paths* paths);
		~Commands();
	};

	/**************** Settings Class ****************/

	/*!
	 * @brief Store settings, configuration options 
	 * and session variables.
	 */
	class Settings
	{
	public:
		bool initialized;

		/**************** Configuration Options ****************/

		/*!
		 * @brief Use online passive-aggressive rank learning if true 
		 * otherwise use SVM-Rank learner.
		 */
		bool USE_ONLINE_LEARNING;

		/*!
		 * @brief Number of iterations to train online learner for each image.
		 */
		int ONLINE_LEARNING_NUM_ITERATIONS;

		/*!
		 * @brief Save inference at each iteration of search if true
		 * otherwise only save inference at final iteration (time bound).
		 */
		bool SAVE_ANYTIME;

		/*!
		 * @brief Save features to file if true for offline learning algorithms.
		 */
		bool OFFLINE_SAVE_FEATURES;

		/*!
		 * @brief Use DAgger to improve heuristic learning.
		 */
		bool USE_DAGGER;

		/**************** Experiment Settings ****************/

		/*!
		 * Classes data.
		 */
		ClassMap CLASSES;

		/**************** MPI-related ****************/

		/*!
		 * Rank of this process. 
		 * If MPI is disabled, then value is 0.
		 */
		int RANK;

		/*!
		 * Number of processes.
		 * If MPI is disabled, then value is 1.
		 */
		int NUM_PROCESSES;

#ifdef USE_MPI
		/*!
		 * MPI status object.
		 */
		MPI_Status* MPI_STATUS;
#endif

		/**************** Other Configuration Constants ****************/

		/*!
		 * Directories and file names.
		 */
		Paths* paths;

		/*!
		 * Commands
		 */
		Commands* cmds;

	public:
		/*!
		 * Initialize settings to defaults. 
		 * Must call Settings::refresh() 
		 * (after getting the data and output paths) to 
		 * fully initialize settings.
		 */
		Settings();

		// Clean up
		~Settings();

		/*!
		 * Refresh settings directories based on the data, output folder and rank.
		 */
		void refresh(string dataDir, string experimentDir);

	private:
		string updateRankIDHelper(string path, string fileName, int rank);
		void refreshDataDirectories(string dataDir);
		void refreshExperimentDirectories(string experimentDir);
		void refreshRankIDFiles(int rankID);
	};
}

#endif