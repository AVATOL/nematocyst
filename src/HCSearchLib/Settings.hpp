#ifndef SETTINGS_H
#define SETTINGS_H

#include <string>

// Define USE_MPI in Makefile or Visual Studio project
// It is defined by default in the Visual Studio project
// It is defined when using "make mpi" and not when using "make"
#ifdef USE_MPI
#include <mpi.h>
#endif

#include "MyPrimitives.hpp"

using namespace std;

namespace HCSearch
{
	/**************** Constants ****************/

	enum RankerType { SVM_RANK, VW_RANK };
	const extern string RankerTypeStrings[];
	const extern bool RankerTypeSaveable[];

	/**************** Class Map ****************/

	// Stores class labels
	// Also stores mapping to class index, as used by confidences
	class ClassMap
	{
	protected:
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

		bool backgroundClassExists();
		int getBackgroundLabel();

		void addClass(int classIndex, int classLabel, bool isBackground);
		void setBackgroundLabel(int classLabel);
	};

	/**************** Run-time Statistics ****************/

	/*!
	 * @brief Store run time statistics.
	 */
	class RunTimeStats
	{
	private:
		int cumSumSuccessors;
		int numSumSuccessors;

	public:
		RunTimeStats();
		~RunTimeStats();

		void addSuccessorCount(int count);
		double getSuccessorAverage();
		void resetSuccessorCount();
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
		string INPUT_DIR;
		string OUTPUT_DIR;

		// external directories

		string LIBLINEAR_DIR;
		string LIBSVM_DIR;
		string SVMRANK_DIR;
		string VOWPALWABBIT_DIR;

		// input directories

		string INPUT_NODES_DIR;
		string INPUT_NODE_LOCATIONS_DIR;
		string INPUT_EDGES_DIR;
		string INPUT_EDGE_FEATURES_DIR;
		string INPUT_META_DIR;
		string INPUT_SEGMENTS_DIR;
		string INPUT_SPLITS_DIR;
		string INPUT_SPLITS_FOLDER_NAME;
		string INPUT_NODES_FOLDER_NAME;
		string INPUT_EDGES_FOLDER_NAME;
		string INPUT_EDGE_FEATURES_FOLDER_NAME;
		string INPUT_INITIAL_STATES_DIR;

		string INPUT_SPLITS_TRAIN_FILE;
		string INPUT_SPLITS_VALIDATION_FILE;
		string INPUT_SPLITS_TEST_FILE;

		string INPUT_SPLITS_TRAIN_FILE_BASE;
		string INPUT_SPLITS_VALIDATION_FILE_BASE;
		string INPUT_SPLITS_TEST_FILE_BASE;

		string INPUT_METADATA_FILE;
		string INPUT_CODEBOOK_FILE;
		string INPUT_INITFUNC_TRAINING_FILE;

		// output directories

		string OUTPUT_LOGS_DIR;
		string OUTPUT_MODELS_DIR;
		string OUTPUT_RESULTS_DIR;
		string OUTPUT_TEMP_DIR;

		string OUTPUT_LOGS_FOLDER_NAME;
		string OUTPUT_MODELS_FOLDER_NAME;
		string OUTPUT_RESULTS_FOLDER_NAME;
		string OUTPUT_TEMP_FOLDER_NAME;

		string OUTPUT_HEURISTIC_FEATURES_FILE;
		string OUTPUT_COST_H_FEATURES_FILE;
		string OUTPUT_COST_ORACLE_H_FEATURES_FILE;
		string OUTPUT_PRUNE_FEATURES_FILE;

		string OUTPUT_HEURISTIC_FEATURES_FILE_BASE;
		string OUTPUT_COST_H_FEATURES_FILE_BASE;
		string OUTPUT_COST_ORACLE_H_FEATURES_FILE_BASE;
		string OUTPUT_PRUNE_FEATURES_FILE_BASE;

		string OUTPUT_ARCHIVED_HEURISTIC_FEATURES_FILE;
		string OUTPUT_ARCHIVED_COST_H_FEATURES_FILE;
		string OUTPUT_ARCHIVED_COST_ORACLE_H_FEATURES_FILE;
		string OUTPUT_ARCHIVED_PRUNE_FEATURES_FILE;

		string OUTPUT_HEURISTIC_MODEL_FILE;
		string OUTPUT_COST_H_MODEL_FILE;
		string OUTPUT_COST_ORACLE_H_MODEL_FILE;
		string OUTPUT_PRUNE_MODEL_FILE;

		string OUTPUT_LOG_FILE;

		string OUTPUT_INITFUNC_MODEL_FILE;
		string OUTPUT_INITFUNC_FEATURES_FILE;
		string OUTPUT_INITFUNC_PREDICT_FILE;

		string OUTPUT_MUTEX_FILE;

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

		string VOWPALWABBIT_TRAIN_CMD;

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
		 * @brief Use DAgger to improve heuristic learning.
		 * NOT IMPLEMENTED
		 */
		bool USE_DAGGER;

		/*!
		 * @brief Check for duplicates when searching.
		 */
		bool CHECK_FOR_DUPLICATES;

		/*!
		 * @brief Use edge weights from external file.
		 */
		bool USE_EDGE_WEIGHTS;

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

		/*!
		 * Run-time statistics.
		 */
		RunTimeStats* stats;

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
		void refresh(string dataDir, string experimentDir, string basePath);

		string updateRankIDHelper(string path, string fileName, int rank);
	private:
		void refreshDataDirectories(string dataDir);
		void refreshExperimentDirectories(string experimentDir);
		void refreshRankIDFiles(int rankID);
		void refreshExternalDirectories(string basePath);
	};
}

#endif