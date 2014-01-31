#include <iostream>
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
			cerr << "error starting MPI program. Terminating." << endl;
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

		cout << "Process [" << Global::settings->RANK << "/" 
			<< Global::settings->NUM_PROCESSES 
			<< "]: MPI initialized!" << endl;
#else
		initializeHelper();
#endif
	}

	void Setup::configure(string datasetPath, string outputPath)
	{
		Global::settings->refresh(MyFileSystem::FileSystem::normalizeDirString(datasetPath), 
			MyFileSystem::FileSystem::normalizeDirString(outputPath));
	}

	void Setup::finalize()
	{
#ifdef USE_MPI
		finalizeHelper();

		synchronize_MPI();

		cout << "Process [" << Global::settings->RANK << "/" 
			<< Global::settings->NUM_PROCESSES 
			<< "] is DONE and exiting..." << endl;
		MPI_Finalize();
#else
		finalizeHelper();
#endif
	}

	void Setup::initializeHelper()
	{
		cout << "Initializing HCSearch... ";

		// initialize settings
		Global::settings = new Settings();

		// initialize logger
		Global::log = new MyLogger::Logger();

		cout << "done!" << endl;
	}

	void Setup::finalizeHelper()
	{
		if (Global::settings != NULL)
			delete Global::settings;
		
		if (Global::log != NULL)
			delete Global::log;

		//Global::settings = NULL;
		//Global::log = NULL;
	}

#ifdef USE_MPI
	void Setup::synchronize_MPI()
	{
		char recvbuff[256];

		char SLAVE_END_CMD[] = "ENDM";
		char FINISH_CMD[] = "FINISHM";

		int rank = Global::settings->RANK;
		int numProcesses = Global::settings->NUM_PROCESSES;

		// wait for everyone to finish before exiting program
		if (rank == 0)
		{
			cout << "Master process [" << rank << "] is done and waiting for slaves to finish..." << endl;

			// wait until all slaves are done
			bool finish[512];
			finish[0] = true;
			for (int i = 1; i < numProcesses; i++)
			{
				finish[i] = false;
			}

			// master process is now waiting for all slave processes to finish
			while (true)
			{
				if (numProcesses <= 1)
					break;

				int msgFlag = 0;
				MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &msgFlag, Global::settings->MPI_STATUS);

				// if a messsage is available
				if (msgFlag)
				{
					// get the message tag and especially source
					int messageID = (*Global::settings->MPI_STATUS).MPI_TAG;
					int messageSource = (*Global::settings->MPI_STATUS).MPI_SOURCE;
					cout << "A message from process [" << messageSource << "]." << endl;

					// receive the message into the buffer and check if it is the ENDHL command
					int ierr = MPI_Recv(recvbuff, 4, MPI_CHAR, messageSource, 0, MPI_COMM_WORLD, Global::settings->MPI_STATUS);
					if (recvbuff[0] == 'E' &&
						recvbuff[1] == 'N' &&
						recvbuff[2] == 'D' &&
						recvbuff[3] == 'M')
					{
						cout << "Received ENDM message from process [" << messageSource << "]." << endl;
						finish[messageSource] = true;

						// test if every process is finished
						bool allFinish = true;
						for (int i = 1; i < numProcesses; i++)
						{
							if (!finish[i])
							{
								cout << "Process [" << i << "] is still finishing up." << endl;
								allFinish = false;
							}
						}
						if (allFinish)
						{
							break;
						}
						cout << "Still waiting for all slaves to finish up." << endl;
					}
				}
			}

			// now tell each slave to continue
			for (int j = 1; j < numProcesses; j++)
			{
				int ierr = MPI_Send(FINISH_CMD, 7, MPI_CHAR, j, 0, MPI_COMM_WORLD);
			}
		}
		else
		{
			cout << "Slave process [" << rank << "] is done and waiting for master..." << endl;

			// send finish heuristic learning message to master
			int ierr = MPI_Send(SLAVE_END_CMD, 4, MPI_CHAR, 0, 0, MPI_COMM_WORLD);

			// now wait until master gives the continue signal
			while (true)
			{
				int ierr = MPI_Recv(recvbuff, 7, MPI_CHAR, 0, 0, MPI_COMM_WORLD, Global::settings->MPI_STATUS);

				if (recvbuff[0] == 'F' &&
					recvbuff[1] == 'I' &&
					recvbuff[2] == 'N' &&
					recvbuff[3] == 'I' &&
					recvbuff[4] == 'S' &&
					recvbuff[5] == 'H' &&
					recvbuff[6] == 'M')
				{
					cout << "Slave process [" << rank << "] got the FINISHM message." << endl;
					break;
				}
			}
		}
	}
#endif

	/**************** Dataset ****************/

	void Dataset::loadDataset(vector< ImgFeatures* >& XTrain, vector< ImgLabeling* >& YTrain, 
		vector< ImgFeatures* >& XValidation, vector< ImgLabeling* >& YValidation, 
		vector< ImgFeatures* >& XTest, vector< ImgLabeling* >& YTest)
	{
		// read in training data
		string trainSplitFile = Global::settings->paths->INPUT_SPLITS_TRAIN_FILE;
		cout << endl << "Reading from " << trainSplitFile << "..." << endl;
		vector<string> trainFiles = readSplitsFile(trainSplitFile);
		for (vector<string>::iterator it = trainFiles.begin(); it != trainFiles.end(); ++it)
		{
			string filename = *it;
			cout << "Loading " << filename << "..." << endl;

			// read meta file
			string metaFile = Global::settings->paths->INPUT_META_DIR + filename + ".txt";
			int numNodes, numFeatures;
			readMetaFile(metaFile, numNodes, numFeatures);

			// read nodes file
			string nodesFile = Global::settings->paths->INPUT_NODES_DIR + filename + ".txt";
			VectorXi labels = VectorXi::Zero(numNodes);
			MatrixXd features = MatrixXd::Zero(numNodes, numFeatures);
			readNodesFile(nodesFile, labels, features);

			// read edges file
			string edgesFile = Global::settings->paths->INPUT_EDGES_DIR + filename + ".txt";
			AdjList_t edges;
			readEdgesFile(edgesFile, edges);

			// construct ImgFeatures
			FeatureGraph featureGraph;
			featureGraph.adjList = edges;
			featureGraph.nodesData = features;
			ImgFeatures* X = new ImgFeatures();
			X->graph = featureGraph;

			// construct ImgLabeling
			LabelGraph labelGraph;
			labelGraph.adjList = edges;
			labelGraph.nodesData = labels;
			ImgLabeling* Y = new ImgLabeling();
			Y->graph = labelGraph;
			
			// push into list
			XTrain.push_back(X);
			YTrain.push_back(Y);
		}

		// read in validation data
		string validSplitFile = Global::settings->paths->INPUT_SPLITS_VALIDATION_FILE;
		cout << endl << "Reading from " << validSplitFile << "..." << endl;
		vector<string> validFiles = readSplitsFile(validSplitFile);
		for (vector<string>::iterator it = validFiles.begin(); it != validFiles.end(); ++it)
		{
			string filename = *it;
			cout << "Loading " << filename << "..." << endl;

			// read meta file
			string metaFile = Global::settings->paths->INPUT_META_DIR + filename + ".txt";
			int numNodes, numFeatures;
			readMetaFile(metaFile, numNodes, numFeatures);

			// read nodes file
			string nodesFile = Global::settings->paths->INPUT_NODES_DIR + filename + ".txt";
			VectorXi labels = VectorXi::Zero(numNodes);
			MatrixXd features = MatrixXd::Zero(numNodes, numFeatures);
			readNodesFile(nodesFile, labels, features);

			// read edges file
			string edgesFile = Global::settings->paths->INPUT_EDGES_DIR + filename + ".txt";
			AdjList_t edges;
			readEdgesFile(edgesFile, edges);

			// construct ImgFeatures
			FeatureGraph featureGraph;
			featureGraph.adjList = edges;
			featureGraph.nodesData = features;
			ImgFeatures* X = new ImgFeatures();
			X->graph = featureGraph;

			// construct ImgLabeling
			LabelGraph labelGraph;
			labelGraph.adjList = edges;
			labelGraph.nodesData = labels;
			ImgLabeling* Y = new ImgLabeling();
			Y->graph = labelGraph;
			
			// push into list
			XValidation.push_back(X);
			YValidation.push_back(Y);
		}

		// read in test data
		string testSplitFile = Global::settings->paths->INPUT_SPLITS_TEST_FILE;
		cout << endl << "Reading from " << testSplitFile << "..." << endl;
		vector<string> testFiles = readSplitsFile(testSplitFile);
		for (vector<string>::iterator it = testFiles.begin(); it != testFiles.end(); ++it)
		{
			string filename = *it;
			cout << "Loading " << filename << "..." << endl;

			// read meta file
			string metaFile = Global::settings->paths->INPUT_META_DIR + filename + ".txt";
			int numNodes, numFeatures;
			readMetaFile(metaFile, numNodes, numFeatures);

			// read nodes file
			string nodesFile = Global::settings->paths->INPUT_NODES_DIR + filename + ".txt";
			VectorXi labels = VectorXi::Zero(numNodes);
			MatrixXd features = MatrixXd::Zero(numNodes, numFeatures);
			readNodesFile(nodesFile, labels, features);

			// read edges file
			string edgesFile = Global::settings->paths->INPUT_EDGES_DIR + filename + ".txt";
			AdjList_t edges;
			readEdgesFile(edgesFile, edges);

			// construct ImgFeatures
			FeatureGraph featureGraph;
			featureGraph.adjList = edges;
			featureGraph.nodesData = features;
			ImgFeatures* X = new ImgFeatures();
			X->graph = featureGraph;

			// construct ImgLabeling
			LabelGraph labelGraph;
			labelGraph.adjList = edges;
			labelGraph.nodesData = labels;
			ImgLabeling* Y = new ImgLabeling();
			Y->graph = labelGraph;
			
			// push into list
			XTest.push_back(X);
			YTest.push_back(Y);
		}
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
			cerr << "[Error] cannot open splits file!" << endl;
			abort();
		}

		return filenames;
	}

	void Dataset::readMetaFile(string filename, int& numNodes, int& numFeatures)
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
			}
			fh.close();
		}
		else
		{
			cerr << "[Error] cannot open meta file!" << endl;
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
			cerr << "[Error] cannot open file to nodes data!" << endl;
			abort();
		}
	}

	void Dataset::readEdgesFile(string filename, AdjList_t& edges)
	{
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
					int node1 = atoi(token1.c_str());

					// get node2
					string token2;
					getline(iss, token2, ' ');
					int node2 = atoi(token2.c_str());

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
			cerr << "[Error] cannot open file to edges data!" << endl;
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
			cerr << "[Error] ranker type is invalid for loading model" << endl;
			return NULL;
		}
	}

	void Model::saveModel(IRankModel* model, string fileName, RankerType rankerType)
	{
		if (rankerType == ONLINE_RANK)
		{
			OnlineRankModel* modelCast = dynamic_cast<OnlineRankModel*>(model);
			modelCast->save(fileName);
		}
		else if (rankerType == SVM_RANK)
		{
			//TODO
		}
		else
		{
			cerr << "[Error] ranker type is invalid for saving model" << endl;
		}
	}

	/**************** Learning ****************/

	IRankModel* Learning::learnH(vector< ImgFeatures* >& XTrain, vector< ImgLabeling* >& YTrain, 
		vector< ImgFeatures* >& XValidation, vector< ImgLabeling* >& YValidation, 
		int timeBound, SearchSpace* searchSpace, ISearchProcedure* searchProcedure)
	{
		//TODO
		return NULL;
	}

	IRankModel* Learning::learnC(vector< ImgFeatures* >& XTrain, vector< ImgLabeling* >& YTrain, 
		vector< ImgFeatures* >& XValidation, vector< ImgLabeling* >& YValidation, 
		IRankModel* heuristicModel, int timeBound, SearchSpace* searchSpace, ISearchProcedure* searchProcedure)
	{
		//TODO
		return NULL;
	}

	IRankModel* Learning::learnCWithOracleH(vector< ImgFeatures* >& XTrain, vector< ImgLabeling* >& YTrain, 
		vector< ImgFeatures* >& XValidation, vector< ImgLabeling* >& YValidation, 
		int timeBound, SearchSpace* searchSpace, ISearchProcedure* searchProcedure)
	{
		//TODO
		return NULL;
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
}