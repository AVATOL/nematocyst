#include <iostream>
#include <sstream>
#include <cstdlib>
#include "Settings.hpp"
#include "MyLogger.hpp"

namespace HCSearch
{
	/**************** Constants ****************/

	const string RankerTypeStrings[] = {"svm-rank", "vw"};
	const bool RankerTypeSaveable[] = {true, true};

	/**************** Class Map ****************/

	ClassMap::ClassMap()
	{
		this->numBackground = 0;
		this->backgroundExists = false;
	}

	ClassMap::~ClassMap()
	{
	}

	int ClassMap::numClasses()
	{
		return this->allClasses.size();
	}

	int ClassMap::numBackgroundClasses()
	{
		return this->numBackground;
	}

	int ClassMap::getClassIndex(int classLabel)
	{
		if (!this->allClasses.iexists(classLabel))
		{
			LOG(ERROR) << "class label does not exist in mapping: " << classLabel;
			exit(1);
		}

		return this->allClasses.ilookup(classLabel);
	}

	int ClassMap::getClassLabel(int classIndex)
	{
		if (!this->allClasses.exists(classIndex))
		{
			LOG(ERROR) << "class index does not exist in mapping: " << classIndex;
			exit(1);
		}

		return this->allClasses.lookup(classIndex);
	}

	bool ClassMap::classIndexIsBackground(int classIndex)
	{
		return this->backgroundClasses[getClassLabel(classIndex)];
	}

	bool ClassMap::classLabelIsBackground(int classLabel)
	{
		return this->backgroundClasses[classLabel];
	}

	set<int> ClassMap::getLabels()
	{
		return this->allClasses.ikeyset();
	}

	set<int> ClassMap::getBackgroundLabels()
	{
		set<int> allLabels = getLabels();
		set<int> backgrounds;
		for (set<int>::iterator it = allLabels.begin(); it != allLabels.end(); ++it)
		{
			int label = *it;
			if (backgroundClasses[label])
				backgrounds.insert(label);
		}
		return backgrounds;
	}

	set<int> ClassMap::getForegroundLabels()
	{
		set<int> allLabels = getLabels();
		set<int> foregrounds;
		for (set<int>::iterator it = allLabels.begin(); it != allLabels.end(); ++it)
		{
			int label = *it;
			if (!backgroundClasses[label])
				foregrounds.insert(label);
		}
		return foregrounds;
	}

	bool ClassMap::backgroundClassExists()
	{
		return this->backgroundExists;
	}

	int ClassMap::getBackgroundLabel()
	{
		if (backgroundClassExists())
		{
			return this->backgroundLabel;
		}
		else
		{
			LOG(ERROR) << "background does not exist in this problem!";
			exit(1);
			return 0;
		}
	}

	void ClassMap::addClass(int classIndex, int classLabel, bool isBackground)
	{
		if (!this->allClasses.exists(classIndex) && !this->allClasses.iexists(classLabel))
		{
			this->allClasses.insert(classIndex, classLabel);
			this->backgroundClasses[classLabel] = isBackground;
			if (isBackground)
			{
				this->numBackground++;
			}
		}
		else
		{
			LOG(ERROR) << "class already exists!";
		}
	}

	void ClassMap::setBackgroundLabel(int classLabel)
	{
		this->backgroundLabel = classLabel;
		this->backgroundExists = true;
	}

	/**************** Run-time Statistics ****************/

	RunTimeStats::RunTimeStats()
	{
		resetSuccessorCount();
	}

	RunTimeStats::~RunTimeStats()
	{
	}

	void RunTimeStats::addSuccessorCount(int count)
	{
		this->cumSumSuccessors += count;
		this->numSumSuccessors++;
	}

	double RunTimeStats::getSuccessorAverage()
	{
		if (numSumSuccessors == 0)
		{
			return -1;
		}

		return 1.0*cumSumSuccessors/numSumSuccessors;
	}

	void RunTimeStats::resetSuccessorCount()
	{
		this->cumSumSuccessors = 0;
		this->numSumSuccessors = 0;
	}


	/**************** Directory/File Paths Class ****************/

	Paths::Paths()
	{
		// misc
#ifdef USE_WINDOWS
		DIR_SEP = "\\";
#else
		DIR_SEP = "/";
#endif

		// basic directories
		BASE_PATH = ""; // must end in DIR_SEP if not empty
		EXTERNAL_DIR = BASE_PATH + "external" + DIR_SEP;

		// external directories
		LIBLINEAR_DIR = EXTERNAL_DIR + "liblinear" + DIR_SEP;
		LIBSVM_DIR = EXTERNAL_DIR + "libsvm" + DIR_SEP;
		SVMRANK_DIR = EXTERNAL_DIR + "svm_rank" + DIR_SEP;
#ifdef USE_WINDOWS
		//VOWPALWABBIT_DIR = EXTERNAL_DIR + "vowpal_wabbit" + DIR_SEP;
		VOWPALWABBIT_DIR = EXTERNAL_DIR; // external/vw.exe provided
#else
		VOWPALWABBIT_DIR = EXTERNAL_DIR + "vowpal_wabbit" + DIR_SEP + "vowpalwabbit" + DIR_SEP;
#endif

		// input directories

		INPUT_SPLITS_FOLDER_NAME = "splits";
		INPUT_NODES_FOLDER_NAME = "nodes";
		INPUT_EDGES_FOLDER_NAME = "edges";
		INPUT_EDGE_FEATURES_FOLDER_NAME = "edgefeatures";

		INPUT_SPLITS_TRAIN_FILE_BASE = "Train.txt";
		INPUT_SPLITS_VALIDATION_FILE_BASE = "Validation.txt";
		INPUT_SPLITS_TEST_FILE_BASE = "Test.txt";

		// output directories

		OUTPUT_LOGS_FOLDER_NAME = "logs";
		OUTPUT_MODELS_FOLDER_NAME = "models";
		OUTPUT_RESULTS_FOLDER_NAME = "results";
		OUTPUT_TEMP_FOLDER_NAME = "temp";

		OUTPUT_HEURISTIC_FEATURES_FILE_BASE = "heuristic_features";
		OUTPUT_COST_H_FEATURES_FILE_BASE = "cost_H_features";
		OUTPUT_COST_ORACLE_H_FEATURES_FILE_BASE = "cost_oracleH_features";
		OUTPUT_PRUNE_FEATURES_FILE_BASE = "prune_features";
	}

	Paths::~Paths()
	{
	}

	void Settings::refreshExternalDirectories(string basePath)
	{
		// basic directories
		this->paths->BASE_PATH = basePath; // must end in DIR_SEP if not empty
		this->paths->EXTERNAL_DIR = this->paths->BASE_PATH + "external" + this->paths->DIR_SEP;

		// external directories
		this->paths->LIBLINEAR_DIR = this->paths->EXTERNAL_DIR + "liblinear" + this->paths->DIR_SEP;
		this->paths->LIBSVM_DIR = this->paths->EXTERNAL_DIR + "libsvm" + this->paths->DIR_SEP;
		this->paths->SVMRANK_DIR = this->paths->EXTERNAL_DIR + "svm_rank" + this->paths->DIR_SEP;
#ifdef USE_WINDOWS
		//this->paths->VOWPALWABBIT_DIR = this->paths->EXTERNAL_DIR + "vowpal_wabbit" + this->paths->DIR_SEP;
		this->paths->VOWPALWABBIT_DIR = this->paths->EXTERNAL_DIR; // external/vw.exe provided
#else
		this->paths->VOWPALWABBIT_DIR = this->paths->EXTERNAL_DIR + "vowpal_wabbit" + this->paths->DIR_SEP + "vowpalwabbit" + this->paths->DIR_SEP;
#endif
	}

	void Settings::refreshDataDirectories(string dataDir)
	{
		this->paths->INPUT_DIR = this->paths->BASE_PATH + dataDir + this->paths->DIR_SEP;

		// data directories
		this->paths->INPUT_NODES_DIR = this->paths->INPUT_DIR + this->paths->INPUT_NODES_FOLDER_NAME + this->paths->DIR_SEP;
		this->paths->INPUT_NODE_LOCATIONS_DIR = this->paths->INPUT_DIR + "nodelocations" + this->paths->DIR_SEP;
		this->paths->INPUT_EDGES_DIR = this->paths->INPUT_DIR + this->paths->INPUT_EDGES_FOLDER_NAME + this->paths->DIR_SEP;
		this->paths->INPUT_EDGE_FEATURES_DIR = this->paths->INPUT_DIR + this->paths->INPUT_EDGE_FEATURES_FOLDER_NAME + this->paths->DIR_SEP;
		this->paths->INPUT_META_DIR = this->paths->INPUT_DIR + "meta" + this->paths->DIR_SEP;
		this->paths->INPUT_SEGMENTS_DIR = this->paths->INPUT_DIR + "segments" + this->paths->DIR_SEP;
		this->paths->INPUT_SPLITS_DIR = this->paths->INPUT_DIR + this->paths->INPUT_SPLITS_FOLDER_NAME + this->paths->DIR_SEP;
		this->paths->INPUT_INITIAL_STATES_DIR = this->paths->INPUT_DIR + "initstate" + this->paths->DIR_SEP;

		this->paths->INPUT_SPLITS_TRAIN_FILE = this->paths->INPUT_SPLITS_DIR + this->paths->INPUT_SPLITS_TRAIN_FILE_BASE;
		this->paths->INPUT_SPLITS_VALIDATION_FILE = this->paths->INPUT_SPLITS_DIR + this->paths->INPUT_SPLITS_VALIDATION_FILE_BASE;
		this->paths->INPUT_SPLITS_TEST_FILE = this->paths->INPUT_SPLITS_DIR + this->paths->INPUT_SPLITS_TEST_FILE_BASE;

		this->paths->INPUT_METADATA_FILE = this->paths->INPUT_DIR + "metadata.txt";
		this->paths->INPUT_CODEBOOK_FILE = this->paths->INPUT_DIR + "codebook.txt";
		this->paths->INPUT_INITFUNC_TRAINING_FILE = this->paths->INPUT_DIR + "initfunc_training.txt";

		this->paths->OUTPUT_INITFUNC_MODEL_FILE = this->paths->INPUT_DIR + "initfunc_model.txt";
	}

	void Settings::refreshExperimentDirectories(string experimentDir)
	{
		this->paths->OUTPUT_DIR = this->paths->BASE_PATH + experimentDir + this->paths->DIR_SEP;

		// experiment directories
		this->paths->OUTPUT_LOGS_DIR = this->paths->OUTPUT_DIR + this->paths->OUTPUT_LOGS_FOLDER_NAME + this->paths->DIR_SEP;
		this->paths->OUTPUT_MODELS_DIR = this->paths->OUTPUT_DIR + this->paths->OUTPUT_MODELS_FOLDER_NAME + this->paths->DIR_SEP;
		this->paths->OUTPUT_RESULTS_DIR = this->paths->OUTPUT_DIR + this->paths->OUTPUT_RESULTS_FOLDER_NAME + this->paths->DIR_SEP;
		this->paths->OUTPUT_TEMP_DIR = this->paths->OUTPUT_DIR + this->paths->OUTPUT_TEMP_FOLDER_NAME + this->paths->DIR_SEP;

		this->paths->OUTPUT_HEURISTIC_MODEL_FILE = this->paths->OUTPUT_MODELS_DIR + "model_heuristic.txt";
		this->paths->OUTPUT_COST_H_MODEL_FILE = this->paths->OUTPUT_MODELS_DIR + "model_cost.txt";
		this->paths->OUTPUT_COST_ORACLE_H_MODEL_FILE = this->paths->OUTPUT_MODELS_DIR + "model_cost_oracleH.txt";
		this->paths->OUTPUT_PRUNE_MODEL_FILE = this->paths->OUTPUT_MODELS_DIR + "model_prune.txt";

		this->paths->OUTPUT_ARCHIVED_HEURISTIC_FEATURES_FILE = this->paths->OUTPUT_MODELS_DIR + "features_heuristic.txt";
		this->paths->OUTPUT_ARCHIVED_COST_H_FEATURES_FILE = this->paths->OUTPUT_MODELS_DIR + "features_cost.txt";
		this->paths->OUTPUT_ARCHIVED_COST_ORACLE_H_FEATURES_FILE = this->paths->OUTPUT_MODELS_DIR + "features_cost_oracleH.txt";
		this->paths->OUTPUT_ARCHIVED_PRUNE_FEATURES_FILE = this->paths->OUTPUT_MODELS_DIR + "features_prune.txt";

		this->paths->OUTPUT_MUTEX_FILE = this->paths->OUTPUT_MODELS_DIR + "model_mutex.txt";
	}

	void Settings::refreshRankIDFiles(int rankID)
	{
		this->RANK = rankID;

		this->paths->OUTPUT_LOG_FILE = updateRankIDHelper(this->paths->OUTPUT_LOGS_DIR, "log", rankID);

		this->paths->OUTPUT_INITFUNC_FEATURES_FILE = updateRankIDHelper(this->paths->OUTPUT_TEMP_DIR, "init_func_features", rankID);
		this->paths->OUTPUT_INITFUNC_PREDICT_FILE = updateRankIDHelper(this->paths->OUTPUT_TEMP_DIR, "init_func_predict", rankID);

		this->paths->OUTPUT_HEURISTIC_FEATURES_FILE = updateRankIDHelper(this->paths->OUTPUT_TEMP_DIR, this->paths->OUTPUT_HEURISTIC_FEATURES_FILE_BASE, rankID);
		this->paths->OUTPUT_COST_H_FEATURES_FILE = updateRankIDHelper(this->paths->OUTPUT_TEMP_DIR, this->paths->OUTPUT_COST_H_FEATURES_FILE_BASE, rankID);
		this->paths->OUTPUT_COST_ORACLE_H_FEATURES_FILE = updateRankIDHelper(this->paths->OUTPUT_TEMP_DIR, this->paths->OUTPUT_COST_ORACLE_H_FEATURES_FILE_BASE, rankID);
		this->paths->OUTPUT_PRUNE_FEATURES_FILE = updateRankIDHelper(this->paths->OUTPUT_TEMP_DIR, this->paths->OUTPUT_PRUNE_FEATURES_FILE_BASE, rankID);
	}

	string Settings::updateRankIDHelper(string path, string fileName, int rank)
	{
		ostringstream oss;
		oss << path << fileName << "_mpi_" << rank << ".txt";
		return oss.str();
	}

	/**************** Commands Class ****************/

	Commands::Commands()
	{
		this->paths = NULL;
	}

	Commands::Commands(Paths* paths)
	{
		this->paths = paths;

#ifdef USE_WINDOWS
		SYSTEM_COPY_CMD = "copy";
		SYSTEM_MKDIR_CMD = "mkdir";
		SYSTEM_RM_CMD = "del";

		LIBLINEAR_PREDICT_CMD = paths->LIBLINEAR_DIR + "windows" + paths->DIR_SEP + "predict";
		LIBLINEAR_TRAIN_CMD = paths->LIBLINEAR_DIR + "windows" + paths->DIR_SEP + "train";
	
		LIBSVM_PREDICT_CMD = paths->LIBSVM_DIR + "windows" + paths->DIR_SEP + "svm-predict";
		LIBSVM_TRAIN_CMD = paths->LIBSVM_DIR + "windows" + paths->DIR_SEP + "svm-train";

		VOWPALWABBIT_TRAIN_CMD = paths->VOWPALWABBIT_DIR + "vw";

		SVMRANK_LEARN_CMD = paths->SVMRANK_DIR + "svm_rank_learn";
#else
		SYSTEM_COPY_CMD = "cp";
		SYSTEM_MKDIR_CMD = "mkdir -p";
		SYSTEM_RM_CMD = "rm -f";

		LIBLINEAR_PREDICT_CMD = paths->LIBLINEAR_DIR + "predict";
		LIBLINEAR_TRAIN_CMD = paths->LIBLINEAR_DIR + "train";
	
		LIBSVM_PREDICT_CMD = paths->LIBSVM_DIR + "svm-predict";
		LIBSVM_TRAIN_CMD = paths->LIBSVM_DIR + "svm-train";

		VOWPALWABBIT_TRAIN_CMD = paths->VOWPALWABBIT_DIR + "vw";

		SVMRANK_LEARN_CMD = paths->SVMRANK_DIR + "svm_rank_learn";
#endif
	}

	Commands::~Commands()
	{
		this->paths = NULL;
	}

	/**************** Settings Class ****************/

	Settings::Settings()
	{
		initialized = false;

		/**************** Configuration Options ****************/

		USE_DAGGER = false;
		CHECK_FOR_DUPLICATES = false;
		USE_EDGE_WEIGHTS = false;

		/**************** Experiment Settings ****************/

		CLASSES = ClassMap();

		/**************** MPI-related ****************/

		RANK = 0;
		NUM_PROCESSES = 1;
#ifdef USE_MPI
		MPI_STATUS = NULL;
#endif

		/**************** Other Configuration Constants ****************/

		paths = new Paths();
		cmds = new Commands(paths);
		stats = new RunTimeStats();
	}

	Settings::~Settings()
	{
		delete cmds;
		delete paths;
		delete stats;
	}

	void Settings::refresh(string dataDir, string experimentDir, string basePath)
	{
		refreshExternalDirectories(basePath);
		refreshDataDirectories(dataDir);
		refreshExperimentDirectories(experimentDir);
		refreshRankIDFiles(this->RANK);
		initialized = true;
	}
}