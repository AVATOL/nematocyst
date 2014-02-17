#include <iostream>
#include <ctime>
#include "HCSearch.hpp"
#include "MyFileSystem.hpp"

using namespace std;

namespace HCSearch
{
	/**************** Initialize/Finalize ****************/

	void Setup::initialize(int argc, char* argv[])
	{
#ifdef USE_MPI
		// initialize MPI
		int rank, size;
		int rc = MPI_Init(&argc, &argv);
		if (rc != MPI_SUCCESS)
		{
			LOG(ERROR) << "error starting MPI program. Terminating.";
			exit(1);
		}

		// get size and rank
		MPI_Comm_size(MPI_COMM_WORLD, &size);
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);

		// initialize settings/logger
		initializeHelper();

		// set rank and number of processes
		Global::settings->RANK = rank;
		Global::settings->NUM_PROCESSES = size;
		Global::settings->MPI_STATUS = new MPI_Status();

		LOG() << "Process [" << Global::settings->RANK << "/" 
			<< Global::settings->NUM_PROCESSES 
			<< "]: MPI initialized!" << endl << endl;
#else
		initializeHelper();
#endif
	}

	void Setup::configure(string datasetPath, string outputPath)
	{
		// refresh Settings
		Global::settings->refresh(MyFileSystem::FileSystem::normalizeDirString(datasetPath), 
			MyFileSystem::FileSystem::normalizeDirString(outputPath));

		// create output folders
		MyFileSystem::FileSystem::createFolder(Global::settings->paths->OUTPUT_DIR);
		MyFileSystem::FileSystem::createFolder(Global::settings->paths->OUTPUT_TEMP_DIR);
		MyFileSystem::FileSystem::createFolder(Global::settings->paths->OUTPUT_RESULTS_DIR);
		MyFileSystem::FileSystem::createFolder(Global::settings->paths->OUTPUT_LOGS_DIR);
		MyFileSystem::FileSystem::createFolder(Global::settings->paths->OUTPUT_MODELS_DIR);

		// set up logging
		MyLogger::Logger::initialize(Global::settings->RANK, Global::settings->NUM_PROCESSES, Global::settings->paths->OUTPUT_LOG_FILE);

		// set classes
		setClasses();
	}

	void Setup::finalize()
	{
#ifdef USE_MPI
		MPI::Synchronize::masterWait("DONESTART");
		MPI::Synchronize::slavesWait("DONEEND");

		finalizeHelper();

		LOG() << "Process [" << Global::settings->RANK << "/" 
			<< Global::settings->NUM_PROCESSES 
			<< "] is DONE and exiting..." << endl;
		MPI_Finalize();
#else
		finalizeHelper();
#endif
	}

	void Setup::initializeHelper()
	{
		LOG() << "Initializing HCSearch... ";

		// initialize settings
		Global::settings = new Settings();

		LOG() << "done!" << endl << endl;
	}

	void Setup::finalizeHelper()
	{
		if (Global::settings != NULL)
			delete Global::settings;

		MyLogger::Logger::finalize();
	}

	void Setup::setClasses()
	{
		set<int> allClassesSet, backgroundClassesSet, foregroundClassesSet;
		int backgroundLabel;
		bool foundBackgroundLabel = false;
		string line;
		string filename = Global::settings->paths->INPUT_METADATA_FILE;
		ifstream fh(filename.c_str());
		if (fh.is_open())
		{
			while (fh.good())
			{
				getline(fh, line);
				stringstream ss(line);
				string exampletype;
				string list;
				getline(ss, exampletype, '=');
				getline(ss, list, '=');
				if (exampletype.compare("classes") == 0)
				{
					allClassesSet = parseList(list);
				}
				else if (exampletype.compare("backgroundclasses") == 0)
				{
					backgroundClassesSet = parseList(list);
				}
				else if (exampletype.compare("backgroundlabel") == 0)
				{
					stringstream ss(list);
					string num;
					getline(ss, num, ',');
					backgroundLabel = atoi(num.c_str());
					foundBackgroundLabel = true;
				}
			}
			fh.close();
		}
		else
		{
			LOG(ERROR) << "cannot open meta file for reading!";
		}

		// process class index and labels
		set_difference(allClassesSet.begin(), allClassesSet.end(), 
			backgroundClassesSet.begin(), backgroundClassesSet.end(), 
			inserter(foregroundClassesSet, foregroundClassesSet.end()));

		LOG() << "=== Class Statistics ===" << endl;
		LOG() << "Class Labels: ";
		int classIndex = 0;
		for (set<int>::iterator it = foregroundClassesSet.begin(); it != foregroundClassesSet.end(); ++it)
		{
			int label = *it;
			Global::settings->CLASSES.addClass(classIndex, label, false);
			LOG() << label << ", ";
			classIndex++;
		}
		LOG() << endl;
		LOG() << "Background Class Labels: ";
		for (set<int>::iterator it = backgroundClassesSet.begin(); it != backgroundClassesSet.end(); ++it)
		{
			int label = *it;
			Global::settings->CLASSES.addClass(classIndex, label, true);
			LOG() << label << ", ";
			classIndex++;
		}
		LOG() << endl;
		if (foundBackgroundLabel)
		{
			Global::settings->CLASSES.setBackgroundLabel(backgroundLabel);
			LOG() << "Main Background Label: " << backgroundLabel << endl;
		}
		else if (backgroundClassesSet.size() == 1)
		{
			for (set<int>::iterator it = backgroundClassesSet.begin(); it != backgroundClassesSet.end(); ++it)
			{
				int label = *it;
				Global::settings->CLASSES.setBackgroundLabel(label);
				LOG() << "Main Background Class: " << label << endl;
			}
		}
		LOG() << endl;
	}

	set<int> Setup::parseList(string str)
	{
		set<int> list = set<int>();

		if (!str.empty())
		{
			stringstream ss(str);
			string num;
			while (getline(ss, num, ','))
			{
				list.insert(atoi(num.c_str()));
			}
		}

		return list;
	}

	/**************** Dataset ****************/

	void Dataset::loadDataset(vector< ImgFeatures* >& XTrain, vector< ImgLabeling* >& YTrain, 
		vector< ImgFeatures* >& XValidation, vector< ImgLabeling* >& YValidation, 
		vector< ImgFeatures* >& XTest, vector< ImgLabeling* >& YTest)
	{
		LOG() << "=== Loading Dataset ===" << endl;

		// read in training data
		string trainSplitFile = Global::settings->paths->INPUT_SPLITS_TRAIN_FILE;
		LOG() << endl << "Reading from " << trainSplitFile << "..." << endl;
		vector<string> trainFiles = readSplitsFile(trainSplitFile);
		loadDatasetHelper(trainFiles, XTrain, YTrain);

		// read in validation data
		string validSplitFile = Global::settings->paths->INPUT_SPLITS_VALIDATION_FILE;
		LOG() << endl << "Reading from " << validSplitFile << "..." << endl;
		vector<string> validFiles = readSplitsFile(validSplitFile);
		loadDatasetHelper(validFiles, XValidation, YValidation);

		// read in test data
		string testSplitFile = Global::settings->paths->INPUT_SPLITS_TEST_FILE;
		LOG() << endl << "Reading from " << testSplitFile << "..." << endl;
		vector<string> testFiles = readSplitsFile(testSplitFile);
		loadDatasetHelper(testFiles, XTest, YTest);

		LOG() << endl;
	}

	void Dataset::unloadDataset(vector< ImgFeatures* >& XTrain, vector< ImgLabeling* >& YTrain, 
		vector< ImgFeatures* >& XValidation, vector< ImgLabeling* >& YValidation, 
		vector< ImgFeatures* >& XTest, vector< ImgLabeling* >& YTest)
	{
		for (vector< ImgFeatures* >::iterator it = XTrain.begin(); it != XTrain.end(); ++it)
		{
			ImgFeatures* object = *it;
			delete object;
		}
		XTrain.clear();
		for (vector< ImgLabeling* >::iterator it = YTrain.begin(); it != YTrain.end(); ++it)
		{
			ImgLabeling* object = *it;
			delete object;
		}
		YTrain.clear();
		for (vector< ImgFeatures* >::iterator it = XValidation.begin(); it != XValidation.end(); ++it)
		{
			ImgFeatures* object = *it;
			delete object;
		}
		XValidation.clear();
		for (vector< ImgLabeling* >::iterator it = YValidation.begin(); it != YValidation.end(); ++it)
		{
			ImgLabeling* object = *it;
			delete object;
		}
		YValidation.clear();
		for (vector< ImgFeatures* >::iterator it = XTest.begin(); it != XTest.end(); ++it)
		{
			ImgFeatures* object = *it;
			delete object;
		}
		XTest.clear();
		for (vector< ImgLabeling* >::iterator it = YTest.begin(); it != YTest.end(); ++it)
		{
			ImgLabeling* object = *it;
			delete object;
		}
		XTest.clear();
	}

	void Dataset::computeTaskRange(int rank, int numTasks, int numProcesses, int& start, int& end)
	{
		if (rank >= numTasks)
		{
			start = 0;
			end = 0;
		}
		else
		{
			if (rank < numTasks%numProcesses)
			{
				start = (int)( rank*ceil(1.0*numTasks/numProcesses) );
				end = (int)( (rank+1)*ceil(1.0*numTasks/numProcesses) );
			}
			else
			{
				start = (int)( (numTasks%numProcesses)*ceil(1.0*numTasks/numProcesses) + (rank - numTasks%numProcesses)*floor(1.0*numTasks/numProcesses) );
				end = (int)( (numTasks%numProcesses)*ceil(1.0*numTasks/numProcesses) + (rank+1 - numTasks%numProcesses)*floor(1.0*numTasks/numProcesses) );
			}
		}
	}

	void Dataset::loadDatasetHelper(vector<string>& files, vector< ImgFeatures* >& XSet, vector< ImgLabeling* >& YSet)
	{
		for (vector<string>::iterator it = files.begin(); it != files.end(); ++it)
		{
			string filename = *it;
			LOG() << "\tLoading " << filename << "..." << endl;

			// read meta file
			string metaFile = Global::settings->paths->INPUT_META_DIR + filename + ".txt";
			int numNodes, numFeatures, height, width;
			readMetaFile(metaFile, numNodes, numFeatures, height, width);

			// read nodes file
			string nodesFile = Global::settings->paths->INPUT_NODES_DIR + filename + ".txt";
			VectorXi labels = VectorXi::Zero(numNodes);
			MatrixXd features = MatrixXd::Zero(numNodes, numFeatures);
			readNodesFile(nodesFile, labels, features);

			// read node locations
			string nodeLocationsFile = Global::settings->paths->INPUT_NODE_LOCATIONS_DIR + filename + ".txt";
			MatrixXd nodeLocations = MatrixXd::Zero(numNodes, 2);
			readNodeLocationsFile(nodeLocationsFile, nodeLocations);

			// read edges file
			string edgesFile = Global::settings->paths->INPUT_EDGES_DIR + filename + ".txt";
			AdjList_t edges;
			readEdgesFile(edgesFile, edges);

			// read segments file
			string segmentsFile = Global::settings->paths->INPUT_SEGMENTS_DIR + filename + ".txt";
			MatrixXi segments = MatrixXi::Zero(height, width);
			readSegmentsFile(segmentsFile, segments);

			// construct ImgFeatures
			FeatureGraph featureGraph;
			featureGraph.adjList = edges;
			featureGraph.nodesData = features;
			ImgFeatures* X = new ImgFeatures();
			X->graph = featureGraph;
			X->filename = filename;
			X->segmentsAvailable = true;
			X->segments = segments;
			X->nodeLocationsAvailable = true;
			X->nodeLocations = nodeLocations;

			// construct ImgLabeling
			LabelGraph labelGraph;
			labelGraph.adjList = edges;
			labelGraph.nodesData = labels;
			ImgLabeling* Y = new ImgLabeling();
			Y->graph = labelGraph;
			
			// push into list
			XSet.push_back(X);
			YSet.push_back(Y);
		}
	}

	vector<string> Dataset::readSplitsFile(string filename)
	{
		vector<string> filenames;

		string line;
		ifstream fh(filename.c_str());
		if (fh.is_open())
		{
			while (fh.good())
			{
				getline(fh, line);
				if (!line.empty())
				{
					filenames.push_back(line);
				}
			}
			fh.close();
		}
		else
		{
			LOG(ERROR) << "cannot open splits file!";
			abort();
		}

		return filenames;
	}

	void Dataset::readMetaFile(string filename, int& numNodes, int& numFeatures, int& height, int& width)
	{
		string line;
		ifstream fh(filename.c_str());
		if (fh.is_open())
		{
			while (fh.good())
			{
				getline(fh, line);
				stringstream ss(line);
				string tag;
				string num;
				getline(ss, tag, '=');
				getline(ss, num, '=');
				if (tag.compare("nodes") == 0)
				{
					numNodes = atoi(num.c_str());
				}
				else if (tag.compare("features") == 0)
				{
					numFeatures = atoi(num.c_str());
				}
				else if (tag.compare("height") == 0)
				{
					height = atoi(num.c_str());
				}
				else if (tag.compare("width") == 0)
				{
					width = atoi(num.c_str());
				}
			}
			fh.close();
		}
		else
		{
			LOG(ERROR) << "cannot open meta file!";
			abort();
		}
	}

	void Dataset::readNodesFile(string filename, VectorXi& labels, MatrixXd& features)
	{
		string line;
		ifstream fh(filename.c_str());
		if (fh.is_open())
		{
			int lineIndex = 0;
			while (fh.good())
			{
				getline(fh, line);
				if (!line.empty())
				{
					// parse line
					istringstream iss(line);
					string token;

					// get label
					getline(iss, token, ' ');
					labels(lineIndex) = atoi(token.c_str());

					// get features
					while (getline(iss, token, ' '))
					{
						if (!token.empty())
						{
							istringstream iss2(token);
							string sIndex;
							getline(iss2, sIndex, ':');
							string sValue;
							getline(iss2, sValue, ':');

							int featureIndex = atoi(sIndex.c_str()) - 1;
							double value = atof(sValue.c_str());

							features(lineIndex, featureIndex) = value;
						}
					}
				}
				lineIndex++;
			}
			fh.close();
		}
		else
		{
			LOG(ERROR) << "cannot open file to nodes data!";
			abort();
		}
	}

	void Dataset::readNodeLocationsFile(string filename, MatrixXd& nodeLocations)
	{
		string line;
		ifstream fh(filename.c_str());
		if (fh.is_open())
		{
			int lineIndex = 0;
			while (fh.good())
			{
				getline(fh, line);
				if (!line.empty())
				{
					// parse line
					istringstream iss(line);

					// get x position
					string token1;
					getline(iss, token1, ' ');
					nodeLocations(lineIndex, 1) = atof(token1.c_str());

					// get y position
					string token2;
					getline(iss, token2, ' ');
					nodeLocations(lineIndex, 2) = atof(token2.c_str());
				}
				lineIndex++;
			}
			fh.close();
		}
		else
		{
			LOG(ERROR) << "cannot open file to node locations data!";
			abort();
		}
	}

	void Dataset::readEdgesFile(string filename, AdjList_t& edges)
	{
		// if 1, then node indices in edge file are 1-based
		// if 0, then node indices in edge file are 0-based
		const int ONE_OFFSET = 1;

		string line;
		ifstream fh(filename.c_str());
		if (fh.is_open())
		{
			// current line = current node
			while (fh.good())
			{
				getline(fh, line);
				if (!line.empty())
				{
					// parse line
					istringstream iss(line);

					// get node1
					string token1;
					getline(iss, token1, ' ');
					int node1 = atoi(token1.c_str()) - ONE_OFFSET;

					// get node2
					string token2;
					getline(iss, token2, ' ');
					int node2 = atoi(token2.c_str()) - ONE_OFFSET;

					// get 1 (or weight)
					string token3;
					getline(iss, token3, ' ');
					double edgeWeight = atof(token3.c_str());

					// add to map
					if (edges.count(node1) == 0)
					{
						edges[node1] = set<int>();
					}
					edges[node1].insert(node2);
				}
			}
			fh.close();
		}
		else
		{
			LOG(ERROR) << "cannot open file to edges data!";
			abort();
		}
	}

	void Dataset::readSegmentsFile(string filename, MatrixXi& segments)
	{
		string line;
		ifstream fh(filename.c_str());
		if (fh.is_open())
		{
			// current line
			int lineIndex = 0;
			while (fh.good())
			{
				getline(fh, line);
				if (!line.empty())
				{
					// parse line
					istringstream iss(line);
					string token;

					// current column
					int columnIndex = 0;
					while (getline(iss, token, ' '))
					{
						if (!token.empty())
						{
							int value = atoi(token.c_str());
							segments(lineIndex, columnIndex) = value;
						}
						columnIndex++;
					}
				}
				lineIndex++;
			}
			fh.close();
		}
		else
		{
			LOG(ERROR) << "cannot open file to segments data!";
			abort();
		}
	}

	/**************** Model ****************/

	IRankModel* Model::loadModel(string fileName, RankerType rankerType)
	{
		if (rankerType == ONLINE_RANK)
		{
			OnlineRankModel* model = new OnlineRankModel();
			model->load(fileName);
			return model;
		}
		else if (rankerType == SVM_RANK)
		{
			SVMRankModel* model = new SVMRankModel();
			model->load(fileName);
			return model;
		}
		else
		{
			LOG(ERROR) << "ranker type is invalid for loading model";
			return NULL;
		}
	}

	void Model::saveModel(IRankModel* model, string fileName, RankerType rankerType)
	{
		if (model == NULL)
		{
			LOG(ERROR) << "rank model is NULL, so cannot save it.";
			return;
		}

		if (rankerType == SVM_RANK)
		{
			SVMRankModel* modelCast = dynamic_cast<SVMRankModel*>(model);
			modelCast->save(fileName);
		}
		else if (rankerType == ONLINE_RANK)
		{
			OnlineRankModel* modelCast = dynamic_cast<OnlineRankModel*>(model);
			modelCast->save(fileName);
		}
		else
		{
			LOG(ERROR) << "ranker type is invalid for saving model";
		}
	}

	/**************** Learning ****************/

	IRankModel* Learning::learnH(vector< ImgFeatures* >& XTrain, vector< ImgLabeling* >& YTrain, 
		vector< ImgFeatures* >& XValidation, vector< ImgLabeling* >& YValidation, 
		int timeBound, SearchSpace* searchSpace, ISearchProcedure* searchProcedure, RankerType rankerType, int numIter)
	{
		clock_t tic = clock();

		LOG() << "Learning the heuristic function..." << endl;
		
		// Setup model for learning
		IRankModel* learningModel = initializeLearning(rankerType, LEARN_H);

		// Learn on each training example
		int start, end;
		HCSearch::Dataset::computeTaskRange(HCSearch::Global::settings->RANK, XTrain.size(), 
			HCSearch::Global::settings->NUM_PROCESSES, start, end);
		for (int i = start; i < end; i++)
		{
			for (int iter = 0; iter < numIter; iter++)
			{
				LOG() << "Heuristic learning: (iter " << iter << ") beginning search on " << XTrain[i]->getFileName() << " (example " << i << ")..." << endl;

				HCSearch::ISearchProcedure::SearchMetadata meta;
				meta.saveAnytimePredictions = false;
				meta.setType = HCSearch::TRAIN;
				meta.exampleName = XTrain[i]->getFileName();
				meta.iter = iter;

				// run search
				searchProcedure->learnH(*XTrain[i], YTrain[i], timeBound, searchSpace, learningModel, meta);

				// save online weights progress in case
				if (learningModel->rankerType() == ONLINE_RANK)
					learningModel->save(Global::settings->paths->OUTPUT_HEURISTIC_ONLINE_WEIGHTS_FILE);
			}
		}
		
		// Merge and learn step
		finishLearning(learningModel, LEARN_H);

		clock_t toc = clock();
		LOG() << "total learnH time: " << (double)(toc - tic)/CLOCKS_PER_SEC << endl << endl;

		return learningModel;
	}

	IRankModel* Learning::learnC(vector< ImgFeatures* >& XTrain, vector< ImgLabeling* >& YTrain, 
		vector< ImgFeatures* >& XValidation, vector< ImgLabeling* >& YValidation, 
		IRankModel* heuristicModel, int timeBound, SearchSpace* searchSpace, ISearchProcedure* searchProcedure, RankerType rankerType, int numIter)
	{
		clock_t tic = clock();

		LOG() << "Learning the cost function with learned heuristic..." << endl;
		
		// Setup model for learning
		IRankModel* learningModel = initializeLearning(rankerType, LEARN_C);

		// Learn on each training example
		int start, end;
		HCSearch::Dataset::computeTaskRange(HCSearch::Global::settings->RANK, XTrain.size(), 
			HCSearch::Global::settings->NUM_PROCESSES, start, end);
		for (int i = start; i < end; i++)
		{
			for (int iter = 0; iter < numIter; iter++)
			{
				LOG() << "Cost learning: (iter " << iter << ") beginning search on " << XTrain[i]->getFileName() << " (example " << i << ")..." << endl;

				HCSearch::ISearchProcedure::SearchMetadata meta;
				meta.saveAnytimePredictions = false;
				meta.setType = HCSearch::TRAIN;
				meta.exampleName = XTrain[i]->getFileName();
				meta.iter = iter;

				// run search
				searchProcedure->learnC(*XTrain[i], YTrain[i], timeBound, searchSpace, heuristicModel, learningModel, meta);

				// save online weights progress in case
				if (learningModel->rankerType() == ONLINE_RANK)
					learningModel->save(Global::settings->paths->OUTPUT_COST_H_ONLINE_WEIGHTS_FILE);
			}
		}
		
		// Merge and learn step
		finishLearning(learningModel, LEARN_C);

		clock_t toc = clock();
		LOG() << "total learnC time: " << (double)(toc - tic)/CLOCKS_PER_SEC << endl << endl;

		return learningModel;
	}

	IRankModel* Learning::learnCWithOracleH(vector< ImgFeatures* >& XTrain, vector< ImgLabeling* >& YTrain, 
		vector< ImgFeatures* >& XValidation, vector< ImgLabeling* >& YValidation, 
		int timeBound, SearchSpace* searchSpace, ISearchProcedure* searchProcedure, RankerType rankerType, int numIter)
	{
		clock_t tic = clock();

		LOG() << "Learning the cost function with oracle heuristic..." << endl;

		// Setup model for learning
		IRankModel* learningModel = initializeLearning(rankerType, LEARN_C_ORACLE_H);

		// Learn on each training example
		int start, end;
		HCSearch::Dataset::computeTaskRange(HCSearch::Global::settings->RANK, XTrain.size(), 
			HCSearch::Global::settings->NUM_PROCESSES, start, end);
		for (int i = start; i < end; i++)
		{
			for (int iter = 0; iter < numIter; iter++)
			{
				LOG() << "Cost with oracle H learning: (iter " << iter << ") beginning search on " << XTrain[i]->getFileName() << " (example " << i << ")..." << endl;

				HCSearch::ISearchProcedure::SearchMetadata meta;
				meta.saveAnytimePredictions = false;
				meta.setType = HCSearch::TRAIN;
				meta.exampleName = XTrain[i]->getFileName();
				meta.iter = iter;

				// run search
				searchProcedure->learnCWithOracleH(*XTrain[i], YTrain[i], timeBound, searchSpace, learningModel, meta);

				// save online weights progress in case
				if (learningModel->rankerType() == ONLINE_RANK)
					learningModel->save(Global::settings->paths->OUTPUT_COST_ORACLE_H_ONLINE_WEIGHTS_FILE);
			}
		}
		
		// Merge and learn step
		finishLearning(learningModel, LEARN_C_ORACLE_H);

		clock_t toc = clock();
		LOG() << "total learnCWithOracleH time: " << (double)(toc - tic)/CLOCKS_PER_SEC << endl << endl;

		return learningModel;
	}

	IRankModel* Learning::learnCWithRandomH(vector< ImgFeatures* >& XTrain, vector< ImgLabeling* >& YTrain, 
		vector< ImgFeatures* >& XValidation, vector< ImgLabeling* >& YValidation, 
		int timeBound, SearchSpace* searchSpace, ISearchProcedure* searchProcedure, RankerType rankerType, int numIter)
	{
		clock_t tic = clock();

		LOG() << "Learning the cost function with random heuristic..." << endl;

		// Setup model for learning
		IRankModel* learningModel = initializeLearning(rankerType, LEARN_C_RANDOM_H);

		// Learn on each training example
		int start, end;
		HCSearch::Dataset::computeTaskRange(HCSearch::Global::settings->RANK, XTrain.size(), 
			HCSearch::Global::settings->NUM_PROCESSES, start, end);
		for (int i = start; i < end; i++)
		{
			for (int iter = 0; iter < numIter; iter++)
			{
				LOG() << "Cost with random H learning: (iter " << iter << ") beginning search on " << XTrain[i]->getFileName() << " (example " << i << ")..." << endl;

				HCSearch::ISearchProcedure::SearchMetadata meta;
				meta.saveAnytimePredictions = false;
				meta.setType = HCSearch::TRAIN;
				meta.exampleName = XTrain[i]->getFileName();
				meta.iter = iter;

				// run search
				searchProcedure->learnCWithRandomH(*XTrain[i], YTrain[i], timeBound, searchSpace, learningModel, meta);

				// save online weights progress in case
				if (learningModel->rankerType() == ONLINE_RANK)
					learningModel->save(Global::settings->paths->OUTPUT_COST_RANDOM_H_ONLINE_WEIGHTS_FILE);
			}
		}
		
		// Merge and learn step
		finishLearning(learningModel, LEARN_C_RANDOM_H);

		clock_t toc = clock();
		LOG() << "total learnCWithRandomH time: " << (double)(toc - tic)/CLOCKS_PER_SEC << endl << endl;

		return learningModel;
	}

	IRankModel* Learning::initializeLearning(RankerType rankerType, SearchType searchType)
	{
		// Setup model for learning
		IRankModel* learningModel = NULL;
		
		if (rankerType == SVM_RANK)
		{
			learningModel = new SVMRankModel();
			SVMRankModel* svmRankModel = dynamic_cast<SVMRankModel*>(learningModel);
			if (searchType == LEARN_H)
				svmRankModel->startTraining(Global::settings->paths->OUTPUT_HEURISTIC_FEATURES_FILE);
			else if (searchType == LEARN_C)
				svmRankModel->startTraining(Global::settings->paths->OUTPUT_COST_H_FEATURES_FILE);
			else if (searchType == LEARN_C_ORACLE_H)
				svmRankModel->startTraining(Global::settings->paths->OUTPUT_COST_ORACLE_H_FEATURES_FILE);
			else if (searchType == LEARN_C_RANDOM_H)
				svmRankModel->startTraining(Global::settings->paths->OUTPUT_COST_RANDOM_H_FEATURES_FILE);
			else
			{
				LOG(ERROR) << "unknown search type!";
				abort();
			}
		}
		else if (rankerType == ONLINE_RANK)
		{
			learningModel = new OnlineRankModel();
			// at this point, it is still not initialized!
		}
		else
		{
			LOG(ERROR) << "unsupported rank learner.";
			abort();
		}

		return learningModel;
	}

	void Learning::finishLearning(IRankModel* learningModel, SearchType searchType)
	{
		if (learningModel->rankerType() == SVM_RANK)
		{
			SVMRankModel* svmRankModel = dynamic_cast<SVMRankModel*>(learningModel);
			if (searchType == LEARN_H)
				svmRankModel->finishTraining(Global::settings->paths->OUTPUT_HEURISTIC_MODEL_FILE, searchType);
			else if (searchType == LEARN_C)
				svmRankModel->finishTraining(Global::settings->paths->OUTPUT_COST_H_MODEL_FILE, searchType);
			else if (searchType == LEARN_C_ORACLE_H)
				svmRankModel->finishTraining(Global::settings->paths->OUTPUT_COST_ORACLE_H_MODEL_FILE, searchType);
			else if (searchType == LEARN_C_RANDOM_H)
				svmRankModel->finishTraining(Global::settings->paths->OUTPUT_COST_RANDOM_H_MODEL_FILE, searchType);
			else
			{
				LOG(ERROR) << "unknown search type!";
				abort();
			}
		}
		else if (learningModel->rankerType() == ONLINE_RANK)
		{
			// do nothing - online weights just stay persistent
			// ...unless using MPI, then merge...
#ifdef USE_MPI
		string STARTMSG;
		string ENDMSG;
		string onlineModelFileBase;
		if (searchType == LEARN_H)
		{
			STARTMSG = "MERGEHSTART";
			ENDMSG = "MERGEHEND";
			onlineModelFileBase = Global::settings->paths->OUTPUT_HEURISTIC_ONLINE_WEIGHTS_FILE_BASE;
		}
		else if (searchType == LEARN_C)
		{
			STARTMSG = "MERGECSTART";
			ENDMSG = "MERGECEND";
			onlineModelFileBase = Global::settings->paths->OUTPUT_COST_H_ONLINE_WEIGHTS_FILE_BASE;
		}
		else if (searchType == LEARN_C_ORACLE_H)
		{
			STARTMSG = "MERGECOHSTART";
			ENDMSG = "MERGECOHEND";
			onlineModelFileBase = Global::settings->paths->OUTPUT_COST_ORACLE_H_ONLINE_WEIGHTS_FILE_BASE;
		}
		else if (searchType == LEARN_C_RANDOM_H)
		{
			STARTMSG = "MERGECRHSTART";
			ENDMSG = "MERGECRHEND";
			onlineModelFileBase = Global::settings->paths->OUTPUT_COST_RANDOM_H_ONLINE_WEIGHTS_FILE_BASE;
		}
		else
		{
			LOG(ERROR) << "unknown search type!";
			abort();
		}

		MPI::Synchronize::masterWait(STARTMSG);

		if (Global::settings->RANK == 0)
		{
			OnlineRankModel* onlineRankModel = dynamic_cast<OnlineRankModel*>(learningModel);
			onlineRankModel->performMerge(onlineModelFileBase, searchType);
		}

		MPI::Synchronize::slavesWait(ENDMSG);
#endif
		}
		else
		{
			LOG(ERROR) << "unsupported rank learner.";
			abort();
		}
	}

	/**************** Inference ****************/

	ImgLabeling Inference::runLLSearch(ImgFeatures* X, ImgLabeling* YTruth, 
		int timeBound, SearchSpace* searchSpace, ISearchProcedure* searchProcedure, 
		ISearchProcedure::SearchMetadata searchMetadata)
	{
		return searchProcedure->llSearch(*X, YTruth, timeBound, searchSpace, searchMetadata);
	}

	ImgLabeling Inference::runHLSearch(ImgFeatures* X, ImgLabeling* YTruth, 
		int timeBound, SearchSpace* searchSpace, ISearchProcedure* searchProcedure,
		IRankModel* heuristicModel, ISearchProcedure::SearchMetadata searchMetadata)
	{
		return searchProcedure->hlSearch(*X, YTruth, timeBound, searchSpace, heuristicModel, searchMetadata);
	}

	ImgLabeling Inference::runLCSearch(ImgFeatures* X, ImgLabeling* YTruth, 
		int timeBound, SearchSpace* searchSpace, ISearchProcedure* searchProcedure,
		IRankModel* costOracleHModel, ISearchProcedure::SearchMetadata searchMetadata)
	{
		return searchProcedure->lcSearch(*X, YTruth, timeBound, searchSpace, costOracleHModel, searchMetadata);
	}

	ImgLabeling Inference::runHCSearch(ImgFeatures* X, int timeBound, 
		SearchSpace* searchSpace, ISearchProcedure* searchProcedure,
		IRankModel* heuristicModel, IRankModel* costModel, 
		ISearchProcedure::SearchMetadata searchMetadata)
	{
		return searchProcedure->hcSearch(*X, timeBound, searchSpace, heuristicModel, costModel, searchMetadata);
	}

	ImgLabeling Inference::runRLSearch(ImgFeatures* X, ImgLabeling* YTruth, 
		int timeBound, SearchSpace* searchSpace, ISearchProcedure* searchProcedure, 
		ISearchProcedure::SearchMetadata searchMetadata)
	{
		return searchProcedure->rlSearch(*X, YTruth, timeBound, searchSpace, searchMetadata);
	}

	ImgLabeling Inference::runRCSearch(ImgFeatures* X, 
		int timeBound, SearchSpace* searchSpace, ISearchProcedure* searchProcedure,
		IRankModel* costOracleHModel, ISearchProcedure::SearchMetadata searchMetadata)
	{
		return searchProcedure->rcSearch(*X, timeBound, searchSpace, costOracleHModel, searchMetadata);
	}
}