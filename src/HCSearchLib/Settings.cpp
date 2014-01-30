#include <iostream>
#include <sstream>
#include "Settings.hpp"

namespace HCSearch
{
	/**************** Class Map ****************/

	const int ClassMap::DEFAULT_BACKGROUND_LABEL = -1;

	ClassMap::ClassMap()
	{
		//TODO
		this->allClasses = MyPrimitives::Bimap<int>();
		this->allClasses.insert(0, 1);
		this->allClasses.insert(1, 0);
		this->allClasses.insert(2, -1);
		this->backgroundClasses[1] = false;
		this->backgroundClasses[0] = false;
		this->backgroundClasses[-1] = true;
		this->backgroundExists = true;
		this->numBackground = 1;

		this->backgroundLabel = DEFAULT_BACKGROUND_LABEL;
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

	/**************** Directory/File Paths Class ****************/

	Paths::Paths()
	{
		// misc
		DIR_SEP = "\\";

		// basic directories
		BASE_PATH = ""; // must end in DIR_SEP if not empty
		EXTERNAL_DIR = BASE_PATH + "external" + DIR_SEP;
		DATA_DIR;
		EXPERIMENT_DIR;

		// external directories
		LIBLINEAR_DIR = EXTERNAL_DIR + "liblinear" + DIR_SEP;
		LIBSVM_DIR = EXTERNAL_DIR + "libsvm" + DIR_SEP;
		SVMRANK_DIR = EXTERNAL_DIR + "svm_rank" + DIR_SEP;
		RANKLIB_DIR + EXTERNAL_DIR;

		// data directories
		DATA_ALLDATA_DIR;
		DATA_INITFUNC_DIR;
		DATA_METADATA_FILE;
		DATA_CODEBOOK_FILE;
		DATA_INITFUNC_TRAINING_FILE;
		DATA_PAIRWISE_TRAINING_FILE;

		// data directories: file name bases
		DATA_NODES_FILE_BASE = "nodes";
		DATA_EDGES_FILE_BASE = "edges";
		DATA_SEGMENTS_FILE_BASE = "segments";

		// experiment directories
		EXPERIMENT_LOGS_DIR;
		EXPERIMENT_MODELS_DIR;
		EXPERIMENT_RESULTS_DIR;
		EXPERIMENT_TEMP_DIR;

		EXPERIMENT_HEURISTIC_FEATURES_FILE;
		EXPERIMENT_COST_H_FEATURES_FILE;
		EXPERIMENT_COST_L_FEATURES_FILE;

		EXPERIMENT_HEURISTIC_ONLINE_WEIGHTS_FILE;
		EXPERIMENT_COST_H_ONLINE_WEIGHTS_FILE;
		EXPERIMENT_COST_L_ONLINE_WEIGHTS_FILE;

		EXPERIMENT_HEURISTIC_MODEL_FILE;
		EXPERIMENT_COST_H_MODEL_FILE;
		EXPERIMENT_COST_L_MODEL_FILE;

		EXPERIMENT_LOG_FILE;

		EXPERIMENT_INITFUNC_MODEL_FILE;
		EXPERIMENT_INITFUNC_FEATURES_FILE;
		EXPERIMENT_INITFUNC_PREDICT_FILE;

		EXPERIMENT_PAIRWISE_MODEL_FILE;
		EXPERIMENT_PAIRWISE_FEATURES_FILE;
		EXPERIMENT_PAIRWISE_PREDICT_FILE;

		EXPERIMENT_RANKLIB_FEATURES_FILE;
		EXPERIMENT_RANKLIB_SCORE_FILE;
	}

	Paths::~Paths()
	{
	}

	void Settings::refreshDataDirectories(string dataDir)
	{
		this->paths->DATA_DIR = this->paths->BASE_PATH + dataDir + this->paths->DIR_SEP;

		// data directories
		this->paths->DATA_ALLDATA_DIR = this->paths->DATA_DIR + "allData" + this->paths->DIR_SEP;
		this->paths->DATA_INITFUNC_DIR = this->paths->DATA_DIR + "initfunc" + this->paths->DIR_SEP;
		this->paths->DATA_METADATA_FILE = this->paths->DATA_DIR + "metadata.txt";
		this->paths->DATA_CODEBOOK_FILE = this->paths->DATA_DIR + "codebook.txt";
		this->paths->DATA_INITFUNC_TRAINING_FILE = this->paths->DATA_DIR + "initfunc_training.txt";
		this->paths->DATA_PAIRWISE_TRAINING_FILE = this->paths->DATA_DIR + "pairwise_training.txt";
	}

	void Settings::refreshExperimentDirectories(string experimentDir)
	{
		this->EXPERIMENT_NAME = experimentDir;

		this->paths->EXPERIMENT_DIR = this->paths->BASE_PATH + experimentDir + this->paths->DIR_SEP;

		// experiment directories
		this->paths->EXPERIMENT_LOGS_DIR = this->paths->EXPERIMENT_DIR + "logs" + this->paths->DIR_SEP;
		this->paths->EXPERIMENT_MODELS_DIR = this->paths->EXPERIMENT_DIR + "models" + this->paths->DIR_SEP;
		this->paths->EXPERIMENT_RESULTS_DIR = this->paths->EXPERIMENT_DIR + "results" + this->paths->DIR_SEP;
		this->paths->EXPERIMENT_TEMP_DIR = this->paths->EXPERIMENT_DIR + "temp" + this->paths->DIR_SEP;

		this->paths->EXPERIMENT_INITFUNC_MODEL_FILE = this->paths->EXPERIMENT_TEMP_DIR + "initfunc_model.txt";
		this->paths->EXPERIMENT_PAIRWISE_MODEL_FILE = this->paths->EXPERIMENT_TEMP_DIR + "pairwise_model.txt";

		this->paths->EXPERIMENT_HEURISTIC_MODEL_FILE = this->paths->EXPERIMENT_MODELS_DIR + "heuristic_model.txt";
		this->paths->EXPERIMENT_COST_H_MODEL_FILE = this->paths->EXPERIMENT_MODELS_DIR + "cost_H_model.txt";
		this->paths->EXPERIMENT_COST_L_MODEL_FILE = this->paths->EXPERIMENT_MODELS_DIR + "cost_L_model.txt";
	}

	void Settings::refreshRankIDFiles(int rankID)
	{
		this->RANK = rankID;

		this->paths->EXPERIMENT_LOG_FILE = updateRankIDHelper(this->paths->EXPERIMENT_LOGS_DIR, "log", rankID);

		this->paths->EXPERIMENT_INITFUNC_FEATURES_FILE = updateRankIDHelper(this->paths->EXPERIMENT_TEMP_DIR, "initfunc_features", rankID);
		this->paths->EXPERIMENT_INITFUNC_PREDICT_FILE = updateRankIDHelper(this->paths->EXPERIMENT_TEMP_DIR, "initfunc_predict", rankID);

		this->paths->EXPERIMENT_PAIRWISE_FEATURES_FILE = updateRankIDHelper(this->paths->EXPERIMENT_TEMP_DIR, "pairwise_features", rankID);
		this->paths->EXPERIMENT_PAIRWISE_PREDICT_FILE = updateRankIDHelper(this->paths->EXPERIMENT_TEMP_DIR, "pairwise_predict", rankID);

		this->paths->EXPERIMENT_RANKLIB_FEATURES_FILE = updateRankIDHelper(this->paths->EXPERIMENT_TEMP_DIR, "ranklib_features", rankID);
		this->paths->EXPERIMENT_RANKLIB_SCORE_FILE = updateRankIDHelper(this->paths->EXPERIMENT_TEMP_DIR, "ranklib_score", rankID);

		this->paths->EXPERIMENT_HEURISTIC_FEATURES_FILE = updateRankIDHelper(this->paths->EXPERIMENT_TEMP_DIR, "heuristicFeatures", rankID);
		this->paths->EXPERIMENT_COST_H_FEATURES_FILE = updateRankIDHelper(this->paths->EXPERIMENT_TEMP_DIR, "costGivenHFeatures", rankID);
		this->paths->EXPERIMENT_COST_L_FEATURES_FILE = updateRankIDHelper(this->paths->EXPERIMENT_TEMP_DIR, "costGivenLFeatures", rankID);

		this->paths->EXPERIMENT_HEURISTIC_ONLINE_WEIGHTS_FILE = updateRankIDHelper(this->paths->EXPERIMENT_TEMP_DIR, "heuristicOnlineWeights", rankID);
		this->paths->EXPERIMENT_COST_H_ONLINE_WEIGHTS_FILE = updateRankIDHelper(this->paths->EXPERIMENT_TEMP_DIR, "costGivenHOnlineWeights", rankID);
		this->paths->EXPERIMENT_COST_L_ONLINE_WEIGHTS_FILE = updateRankIDHelper(this->paths->EXPERIMENT_TEMP_DIR, "costGivenLOnlineWeights", rankID);
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

		SYSTEM_COPY_CMD = "copy";
		SYSTEM_MKDIR_CMD = "mkdir";
		SYSTEM_RM_CMD = "del";

		LIBLINEAR_PREDICT_CMD = paths->LIBLINEAR_DIR + "windows" + paths->DIR_SEP + "predict";
		LIBLINEAR_TRAIN_CMD = paths->LIBLINEAR_DIR + "windows" + paths->DIR_SEP + "train";
	
		LIBSVM_PREDICT_CMD = paths->LIBSVM_DIR + "windows" + paths->DIR_SEP + "svm-predict";
		LIBSVM_TRAIN_CMD = paths->LIBSVM_DIR + "windows" + paths->DIR_SEP + "svm-train";

		SVMRANK_LEARN_CMD = paths->SVMRANK_DIR + "svm_rank_learn";

		RANKLIB_CMD = "java -jar " + paths->RANKLIB_DIR + "RankLib.jar";
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

		SEARCH_SPACE_TYPE = SearchSpaceMode::STOCHASTIC;
		SEARCH_PROCEDURE_TYPE = SearchProcedureMode::GREEDY;
		FEAT_FUNC_GLOBAL_POTENTIAL_MODE = GlobalPotentialMode::NONE;
		INIT_FUNC_MODE = InitialFunctionMode::LOG_REG;
		STOCHASTIC_CUT_MODE = StochasticCutMode::EDGES;
		USE_ONLINE_LEARNING = true;
		ONLINE_LEARNING_NUM_ITERATIONS = 1;
		SAVE_ANYTIME = true;
		OFFLINE_SAVE_FEATURES = false;
		USE_DAGGER = false;

		/**************** Experiment Settings ****************/

		EXPERIMENT_NAME = "";
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