#include "MyFileSystem.hpp"
#include "InitialStateFunction.hpp"
#include "Globals.hpp"

namespace HCSearch
{
	/**************** Initial Prediction Functions ****************/

	const double LogRegInit::DEFAULT_C = 10;
	const double LogRegInit::BINARY_CONFIDENCE_THRESHOLD = 0.75;

	LogRegInit::LogRegInit()
	{
#ifdef USE_MPI
		MPI::Synchronize::masterWait("INITPREDSTART");
#endif

		if (Global::settings->RANK == 0)
			trainClassifier();

#ifdef USE_MPI
		MPI::Synchronize::slavesWait("INITPREDEND");
#endif
	}

	LogRegInit::~LogRegInit()
	{
	}

	ImgLabeling LogRegInit::getInitialPrediction(ImgFeatures& X)
	{
		// output features
		imgfeatures2liblinear(X, Global::settings->paths->OUTPUT_INITFUNC_FEATURES_FILE);
		
		// perform IID SVM prediction on patches
		stringstream ssPredictInitFuncCmd;
		ssPredictInitFuncCmd << Global::settings->cmds->LIBLINEAR_PREDICT_CMD << " -b 1 " 
			<< Global::settings->paths->OUTPUT_INITFUNC_FEATURES_FILE << " " + Global::settings->paths->OUTPUT_INITFUNC_MODEL_FILE 
			<< " " << Global::settings->paths->OUTPUT_INITFUNC_PREDICT_FILE;

		int retcode = MyFileSystem::Executable::executeRetries(ssPredictInitFuncCmd.str());
		if (retcode != 0)
		{
			LOG(ERROR) << "Initial prediction failed!";
			abort();
		}

		ImgLabeling Y = ImgLabeling();
		Y.graph = LabelGraph();
		Y.graph.adjList = X.graph.adjList;
		Y.graph.nodesData = VectorXi::Ones(X.getNumNodes());

		// now need to get labels data and confidences...
		// read in initial prediction
		liblinear2imglabeling(Y, Global::settings->paths->OUTPUT_INITFUNC_PREDICT_FILE);

		// eliminate 1-islands
		eliminateIslands(Y);

		return Y;
	}

	void LogRegInit::eliminateIslands(ImgLabeling& Y)
	{
		if (!Global::settings->CLASSES.backgroundClassExists())
			return;

		const int numNodes = Y.getNumNodes();
		for (int node = 0; node < numNodes; node++)
		{
			if (!Global::settings->CLASSES.classLabelIsBackground(Y.getLabel(node)) && !hasForegroundNeighbors(Y, node))
			{
				int label = Y.getLabel(node);
				double probEstimate = Y.confidences(node, Global::settings->CLASSES.getClassIndex(label));
				if (probEstimate < BINARY_CONFIDENCE_THRESHOLD)
				{
					Y.graph.nodesData(node) = Global::settings->CLASSES.getBackgroundLabel();
				}
			}
		}
	}

	// Train logistic regression model
	void LogRegInit::trainClassifier()
	{
		// train on IID classifier first for initial function
		string fileName = Global::settings->paths->OUTPUT_INITFUNC_MODEL_FILE;
		ifstream fh(fileName.c_str());
		if (!fh.is_open())
		{
			LOG() << "Training log reg initial function model..." << endl;
			stringstream ssTrainInitFuncCmd;

			// LOGISTIC REGRESSION
			ssTrainInitFuncCmd << Global::settings->cmds->LIBLINEAR_TRAIN_CMD << " -s 7 -c " << DEFAULT_C << " ";

			// the rest of the training cmd
			ssTrainInitFuncCmd << Global::settings->paths->INPUT_INITFUNC_TRAINING_FILE << " " + Global::settings->paths->OUTPUT_INITFUNC_MODEL_FILE;

			// run command
			MyFileSystem::Executable::executeRetries(ssTrainInitFuncCmd.str());

			LOG() << "...Finished training initial function model." << endl;
		}
		else
		{
			LOG() << "Initial function model found. Using it..." << endl;
			fh.close();
		}
	}

	void LogRegInit::imgfeatures2liblinear(ImgFeatures& X, string filename)
	{
		const int numNodes = X.getNumNodes();
		const int featureDim = X.getFeatureDim();

		ofstream fh(filename.c_str());
		if (fh.is_open())
		{
			for (int node = 0; node < numNodes; node++)
			{
				fh << "1"; // dummy class for prediction
				for (int feat = 0; feat < featureDim; feat++)
				{
					if (X.getFeature(node, feat) != 0)
					{
						fh << " " << feat+1 << ":" << X.getFeature(node, feat);
					}
				}
				fh << endl;
			}
			fh.close();
		}
		else
		{
			LOG(ERROR) << "cannot open file for writing LIBLINEAR/LIBSVM features!";
			abort();
		}
	}

	void LogRegInit::liblinear2imglabeling(ImgLabeling& Y, string filename)
	{
		const int numClasses = Global::settings->CLASSES.numClasses();
		const int numNodes = Y.getNumNodes();

		vector<int> labelOrderFound;

		int lineIndex = 0;
		string line;
		ifstream fh(filename.c_str());
		if (fh.is_open())
		{
			int numClassesFound = 0;
			while (fh.good() && lineIndex < numNodes+1)
			{
				getline(fh, line);
				if (lineIndex == 0)
				{
					// parse first line to get label order
					stringstream ss(line);
					string token;
					int columnIndex = 0;
					while (getline(ss, token, ' '))
					{
						// first token on first line should be "labels"
						if (columnIndex == 0)
						{
							if (token.compare("labels") != 0)
							{
								LOG(ERROR) << "parsing invalid prediction file while trying to get liblinear confidences!";
								fh.close();
								abort();
							}
							columnIndex++;
							continue;
						}

						int label = atoi(token.c_str());
						labelOrderFound.push_back(label);

						columnIndex++;
					}

					numClassesFound = labelOrderFound.size();
					if (numClassesFound == 0)
					{
						Y.confidencesAvailable = false;
					}
					else if (numClassesFound != numClasses)
					{
						LOG(ERROR) << "number of classes found in prediction file while trying to get liblinear confidences is not correct!" << endl
							<< "\texpected: " << numClasses << endl
							<< "\tfound: " << numClassesFound << endl
							<< "\tglobal: " << Global::settings->CLASSES.numClasses();

						LOG(ERROR) << "parsing invalid prediction file while trying to get liblinear confidences!";
						fh.close();
						abort();
					}
					else
					{
						Y.confidencesAvailable = true;
						Y.confidences = MatrixXd::Zero(numNodes, numClasses);
					}
				}
				else if (!line.empty())
				{
					// parse line to get label and confidences
					stringstream ss(line);
					string token;
					int columnIndex = 0;
					while (getline(ss, token, ' '))
					{
						if (columnIndex == 0)
						{
							int nodeIndex = lineIndex-1;
							Y.graph.nodesData(nodeIndex) = atoi(token.c_str());
						}
						else if (Y.confidencesAvailable)
						{
							int nodeIndex = lineIndex-1;
							int classIndex = Global::settings->CLASSES.getClassIndex(labelOrderFound[columnIndex-1]);
							Y.confidences(nodeIndex, classIndex) = atof(token.c_str());
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
			LOG(ERROR) << "cannot open file for reading LIBLINEAR/LIBSVM confidences!";
			abort();
		}
	}

	bool LogRegInit::hasForegroundNeighbors(ImgLabeling& Y, int node)
	{
		int nodeLabel = Y.getLabel(node);
		NeighborSet_t neighbors = Y.graph.adjList[node];

		bool hasNeighbors = false;

		for (NeighborSet_t::iterator it = neighbors.begin(); it != neighbors.end(); ++it)
		{
			int neighborNode = *it;
			int neighborLabel = Y.getLabel(neighborNode);
			
			if (!Global::settings->CLASSES.classLabelIsBackground(neighborLabel))
			{
				hasNeighbors = true;
				break;
			}
		}

		return hasNeighbors;
	}
}