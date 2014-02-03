#include <iostream>
#include <sstream>
#include <cstdlib>
#include "Settings.hpp"

namespace HCSearch
{
	/**************** Constants ****************/

	const string RankerTypeStrings[] = {"svm-rank", "online"};

	/**************** Class Map ****************/

	ClassMap::ClassMap()
	{
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
			cerr << "[Error] class label does not exist in mapping: " << classLabel << endl;
			exit(1);
		}

		return this->allClasses.ilookup(classLabel);
	}

	int ClassMap::getClassLabel(int classIndex)
	{
		if (!this->allClasses.iexists(classIndex))
		{
			cerr << "[Error] class index does not exist in mapping: " << classIndex << endl;
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

	int ClassMap::getBackgroundLabel()
	{
		if (backgroundExists)
		{
			return this->backgroundLabel;
		}
		else
		{
			cerr << "[Error] background does not exist in this problem!" << endl;
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
			cerr << "[Error] class already exists!" << endl;
		}
	}

	void ClassMap::setBackgroundLabel(int classLabel)
	{
		this->backgroundLabel = classLabel;
		this->backgroundExists = true;
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

		// input directories

		INPUT_SPLITS_FOLDER_NAME = "splits";
	}

	Paths::~Paths()
	{
	}

	void Settings::refreshDataDirectories(string dataDir)
	{
		this->paths->INPUT_DIR = this->paths->BASE_PATH + dataDir + this->paths->DIR_SEP;

		// data directories
		this->paths->INPUT_NODES_DIR = this->paths->INPUT_DIR + "nodes" + this->paths->DIR_SEP;
		this->paths->INPUT_EDGES_DIR = this->paths->INPUT_DIR + "edges" + this->paths->DIR_SEP;
		this->paths->INPUT_META_DIR = this->paths->INPUT_DIR + "meta" + this->paths->DIR_SEP;
		this->paths->INPUT_SEGMENTS_DIR = this->paths->INPUT_DIR + "segments" + this->paths->DIR_SEP;
		this->paths->INPUT_SPLITS_DIR = this->paths->INPUT_DIR + this->paths->INPUT_SPLITS_FOLDER_NAME + this->paths->DIR_SEP;

		this->paths->INPUT_SPLITS_TRAIN_FILE = this->paths->INPUT_SPLITS_DIR + "Train.txt";
		this->paths->INPUT_SPLITS_VALIDATION_FILE = this->paths->INPUT_SPLITS_DIR + "Validation.txt";
		this->paths->INPUT_SPLITS_TEST_FILE = this->paths->INPUT_SPLITS_DIR + "Test.txt";

		this->paths->INPUT_METADATA_FILE = this->paths->INPUT_DIR + "metadata.txt";
		this->paths->INPUT_CODEBOOK_FILE = this->paths->INPUT_DIR + "codebook.txt";
		this->paths->INPUT_INITFUNC_TRAINING_FILE = this->paths->INPUT_DIR + "initfunc_training.txt";
	}

	void Settings::refreshExperimentDirectories(string experimentDir)
	{
		this->paths->OUTPUT_DIR = this->paths->BASE_PATH + experimentDir + this->paths->DIR_SEP;

		// experiment directories
		this->paths->OUTPUT_LOGS_DIR = this->paths->OUTPUT_DIR + "logs" + this->paths->DIR_SEP;
		this->paths->OUTPUT_MODELS_DIR = this->paths->OUTPUT_DIR + "models" + this->paths->DIR_SEP;
		this->paths->OUTPUT_RESULTS_DIR = this->paths->OUTPUT_DIR + "results" + this->paths->DIR_SEP;
		this->paths->OUTPUT_TEMP_DIR = this->paths->OUTPUT_DIR + "temp" + this->paths->DIR_SEP;

		this->paths->OUTPUT_INITFUNC_MODEL_FILE = this->paths->OUTPUT_TEMP_DIR + "init_func_model.txt";

		this->paths->OUTPUT_HEURISTIC_MODEL_FILE = this->paths->OUTPUT_MODELS_DIR + "model_heuristic.txt";
		this->paths->OUTPUT_COST_H_MODEL_FILE = this->paths->OUTPUT_MODELS_DIR + "model_cost.txt";
		this->paths->OUTPUT_COST_ORACLE_H_MODEL_FILE = this->paths->OUTPUT_MODELS_DIR + "model_cost_oracleH.txt";

		this->paths->OUTPUT_ARCHIVED_HEURISTIC_FEATURES_FILE = this->paths->OUTPUT_MODELS_DIR + "features_heuristic.txt";
		this->paths->OUTPUT_ARCHIVED_COST_H_FEATURES_FILE = this->paths->OUTPUT_MODELS_DIR + "features_cost.txt";
		this->paths->OUTPUT_ARCHIVED_COST_ORACLE_H_FEATURES_FILE = this->paths->OUTPUT_MODELS_DIR + "features_cost_oracleH.txt";
	}

	void Settings::refreshRankIDFiles(int rankID)
	{
		this->RANK = rankID;

		this->paths->OUTPUT_LOG_FILE = updateRankIDHelper(this->paths->OUTPUT_LOGS_DIR, "log", rankID);

		this->paths->OUTPUT_INITFUNC_FEATURES_FILE = updateRankIDHelper(this->paths->OUTPUT_TEMP_DIR, "init_func_features", rankID);
		this->paths->OUTPUT_INITFUNC_PREDICT_FILE = updateRankIDHelper(this->paths->OUTPUT_TEMP_DIR, "init_func_predict", rankID);

		this->paths->OUTPUT_HEURISTIC_FEATURES_FILE = updateRankIDHelper(this->paths->OUTPUT_TEMP_DIR, "heuristic_features", rankID);
		this->paths->OUTPUT_COST_H_FEATURES_FILE = updateRankIDHelper(this->paths->OUTPUT_TEMP_DIR, "cost_H_features", rankID);
		this->paths->OUTPUT_COST_ORACLE_H_FEATURES_FILE = updateRankIDHelper(this->paths->OUTPUT_TEMP_DIR, "cost_oracleH_features", rankID);

		this->paths->OUTPUT_HEURISTIC_ONLINE_WEIGHTS_FILE = updateRankIDHelper(this->paths->OUTPUT_TEMP_DIR, "heuristic_online_weights", rankID);
		this->paths->OUTPUT_COST_H_ONLINE_WEIGHTS_FILE = updateRankIDHelper(this->paths->OUTPUT_TEMP_DIR, "cost_H_online_weights", rankID);
		this->paths->OUTPUT_COST_ORACLE_H_ONLINE_WEIGHTS_FILE = updateRankIDHelper(this->paths->OUTPUT_TEMP_DIR, "cost_oracleH_model.txt", rankID);
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

		SVMRANK_LEARN_CMD = paths->SVMRANK_DIR + "svm_rank_learn";
#else
		SYSTEM_COPY_CMD = "cp";
		SYSTEM_MKDIR_CMD = "mkdir -p";
		SYSTEM_RM_CMD = "rm -f";

		LIBLINEAR_PREDICT_CMD = paths->LIBLINEAR_DIR + "predict";
		LIBLINEAR_TRAIN_CMD = paths->LIBLINEAR_DIR + "train";
	
		LIBSVM_PREDICT_CMD = paths->LIBSVM_DIR + "svm-predict";
		LIBSVM_TRAIN_CMD = paths->LIBSVM_DIR + "svm-train";

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
	}

	Settings::~Settings()
	{
		delete cmds;
		cmds = NULL;

		delete paths;
		paths = NULL;
	}

	void Settings::refresh(string dataDir, string experimentDir)
	{
		refreshDataDirectories(dataDir);
		refreshExperimentDirectories(experimentDir);
		refreshRankIDFiles(this->RANK);
		initialized = true;
	}
}