#include <iostream>
#include <vector>
#include "MyFileSystem.hpp"
#include "SearchSpace.hpp"
#include "Settings.hpp"
#include "Globals.hpp"
#include "MyLogger.hpp"

using namespace std;

namespace HCSearch
{
	/**************** Search Space Functions Abstract Definitions ****************/

	int IFeatureFunction::featureSize(ImgFeatures& X, ImgLabeling& Y)
	{
		// This is inefficient but does the job
		// Better to define more efficient functions for extended classes
		return computeFeatures(X, Y).data.size();
	}

	/**************** Feature Functions ****************/

	StandardFeatures::StandardFeatures()
	{
	}

	StandardFeatures::~StandardFeatures()
	{
	}

	RankFeatures StandardFeatures::computeFeatures(ImgFeatures& X, ImgLabeling& Y)
	{
		int numNodes = X.getNumNodes();
		int featureDim = X.getFeatureDim();
		int numClasses = Global::settings->CLASSES.numClasses();

		int unaryFeatDim = 1+featureDim;
		int pairwiseFeatDim = featureDim;

		VectorXd phi = VectorXd::Zero(featureSize(X, Y));
		
		VectorXd unaryTerm = computeUnaryTerm(X, Y);
		VectorXd pairwiseTerm = computePairwiseTerm(X, Y);

		phi.segment(0, numClasses*unaryFeatDim) = unaryTerm;
		phi.segment(numClasses*unaryFeatDim, (numClasses+1)*pairwiseFeatDim) = pairwiseTerm;

		return RankFeatures(phi);
	}

	int StandardFeatures::featureSize(ImgFeatures& X, ImgLabeling& Y)
	{
		int numNodes = X.getNumNodes();
		int featureDim = X.getFeatureDim();
		int unaryFeatDim = 1+featureDim;
		int pairwiseFeatDim = featureDim;
		int numClasses = Global::settings->CLASSES.numClasses();

		return numClasses*unaryFeatDim + (numClasses+1)*pairwiseFeatDim;
	}

	VectorXd StandardFeatures::computeUnaryTerm(ImgFeatures& X, ImgLabeling& Y)
	{
		const int numNodes = X.getNumNodes();
		const int numClasses = Global::settings->CLASSES.numClasses();
		const int featureDim = X.getFeatureDim();
		const int unaryFeatDim = 1+featureDim;
		
		VectorXd phi = VectorXd::Zero(numClasses*unaryFeatDim);

		// unary potential
		for (int node = 0; node < numNodes; node++)
		{
			// get node features and label
			VectorXd nodeFeatures = X.graph.nodesData.row(node);
			int nodeLabel = Y.getLabel(node);

			// map node label to indexing value in phi vector
			int classIndex = Global::settings->CLASSES.getClassIndex(nodeLabel);

			// assignment: bias and unary feature
			phi(classIndex*unaryFeatDim) += 1;
			phi.segment(classIndex*unaryFeatDim+1, featureDim) += nodeFeatures;
		}

		return phi;
	}
	
	VectorXd StandardFeatures::computePairwiseTerm(ImgFeatures& X, ImgLabeling& Y)
	{
		const int numNodes = X.getNumNodes();
		const int numClasses = Global::settings->CLASSES.numClasses();
		const int featureDim = X.getFeatureDim();
		const int pairwiseFeatDim = featureDim;
		
		VectorXd phi = VectorXd::Zero((numClasses+1)*pairwiseFeatDim);

		for (int node1 = 0; node1 < numNodes; node1++)
		{
			if (X.graph.adjList.count(node1) == 0)
				continue;

			// get neighbors (ending nodes) of starting node
			NeighborSet_t neighbors = X.graph.adjList[node1];
			const int numNeighbors = neighbors.size();
			for (NeighborSet_t::iterator it = neighbors.begin(); it != neighbors.end(); ++it)
			{
				int node2 = *it;

				// get node features and label
				VectorXd nodeFeatures1 = X.graph.nodesData.row(node1);
				int nodeLabel1 = Y.getLabel(node1);

				VectorXd nodeFeatures2 = X.graph.nodesData.row(node2);
				int nodeLabel2 = Y.getLabel(node2);

				int classIndex = -1;
				VectorXd edgeFeatureVector = computePairwiseFeatures(nodeFeatures1, nodeFeatures2, nodeLabel1, nodeLabel2, classIndex);
				phi.segment(classIndex*pairwiseFeatDim, pairwiseFeatDim) += edgeFeatureVector; // contrast sensitive pairwise potential
			}
		}

		return 0.5*phi;
	}

	VectorXd StandardFeatures::computePairwiseFeatures(VectorXd& nodeFeatures1, VectorXd& nodeFeatures2, 
		int nodeLabel1, int nodeLabel2, int& classIndex)
	{
		// phi features depend on labels
		if (nodeLabel1 != nodeLabel2)
		{
			VectorXd diff = nodeFeatures1 - nodeFeatures2;
			VectorXd negdiffabs2 = -diff.cwiseAbs2();
			VectorXd expnegdiffabs2 = negdiffabs2.array().exp();
			classIndex = Global::settings->CLASSES.numClasses(); // numClasses

			// assignment
			return expnegdiffabs2;
		}
		else
		{
			VectorXd diff = nodeFeatures1 - nodeFeatures2;
			VectorXd negdiffabs2 = -diff.cwiseAbs2();
			VectorXd expnegdiffabs2 = 1 - negdiffabs2.array().exp();

			// map node label to indexing value in phi vector
			classIndex = Global::settings->CLASSES.getClassLabel(nodeLabel1);

			// assignment
			return expnegdiffabs2;
		}
	}

	/**************** Initial Prediction Functions ****************/

	const double LogRegInit::DEFAULT_C = 10;
	const double LogRegInit::BINARY_CONFIDENCE_THRESHOLD = 0.75;

	LogRegInit::LogRegInit()
	{
	}

	LogRegInit::LogRegInit(string fileName)
	{
		trainClassifier(fileName);
	}

	LogRegInit::~LogRegInit()
	{
	}

	//TODO: test
	ImgLabeling LogRegInit::getInitialPrediction(ImgFeatures& X)
	{
		// This implements logistic regression

		const int numNodes = X.getNumNodes();

		// output features
		imgfeatures2liblinear(X, Global::settings->paths->EXPERIMENT_INITFUNC_FEATURES_FILE);
		
		// perform IID SVM prediction on patches
		stringstream ssPredictInitFuncCmd;
		ssPredictInitFuncCmd << Global::settings->cmds->LIBLINEAR_PREDICT_CMD << " -b 1 " 
			<< Global::settings->paths->EXPERIMENT_INITFUNC_FEATURES_FILE << " " + Global::settings->paths->EXPERIMENT_INITFUNC_MODEL_FILE 
			<< " " << Global::settings->paths->EXPERIMENT_INITFUNC_PREDICT_FILE;

		int retcode = MyFileSystem::Executable::execute(ssPredictInitFuncCmd.str().c_str());
		if (retcode != 0)
		{
			cerr << "[Error] Initial prediction failed!" << endl;
			exit(1);
		}

		ImgLabeling Y = ImgLabeling();
		Y.graph = LabelGraph();
		Y.graph.adjList = X.graph.adjList;
		Y.graph.nodesData = VectorXi::Ones(X.getNumNodes());

		// now need to get labels data and confidences...

		// read in initial prediction
		liblinear2imglabeling(Y, Global::settings->paths->EXPERIMENT_INITFUNC_PREDICT_FILE);

		// eliminate 1-islands
		for (int node = 0; node < numNodes; node++)
		{
			if (!Global::settings->CLASSES.classLabelIsBackground(Y.getLabel(node)) && !hasForegroundNeighbors(Y, node))
			{
				int label = Y.getLabel(node);
				double probEstimate = Y.confidences(node, Global::settings->CLASSES.getClassIndex(label));
				if (probEstimate < BINARY_CONFIDENCE_THRESHOLD)
				{
					Y.graph.nodesData(node, 1) = Global::settings->CLASSES.getBackgroundLabel();
				}
			}
		}

		return Y;
	}

	// Train logistic regression model
	void LogRegInit::trainClassifier(string fileName)
	{
		// train on IID classifier first for initial function
		ifstream fh(fileName.c_str());
		if (!fh.is_open())
		{
			cout << "Training log reg initial function model..." << endl;
			stringstream ssTrainInitFuncCmd;

			// LOGISTIC REGRESSION
			ssTrainInitFuncCmd << Global::settings->cmds->LIBLINEAR_TRAIN_CMD << " -s 7 -c " << DEFAULT_C << " ";

			// the rest of the training cmd
			ssTrainInitFuncCmd << Global::settings->paths->DATA_INITFUNC_TRAINING_FILE << " " + Global::settings->paths->EXPERIMENT_INITFUNC_MODEL_FILE;

			// run command
			MyFileSystem::Executable::executeRetries(ssTrainInitFuncCmd.str());

			cout << "Finished training initial function model..." << endl;
		}
		else
		{
			cout << "Initial function model found. Using it..." << endl;
			fh.close();
		}
	}

	int LogRegInit::imgfeatures2liblinear(ImgFeatures& X, string filename)
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
			return 0;
		}
		else
		{
			cerr << "[Error] cannot open file for writing LIBLINEAR/LIBSVM features!" << endl;
			return -1;
		}
	}

	int LogRegInit::liblinear2imglabeling(ImgLabeling& Y, string filename)
	{
		const int numClasses = Global::settings->CLASSES.numClasses();
		const int numNodes = Y.getNumNodes();

		vector<int> labelOrderFound = vector<int>();

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
								cerr << "[Error] parsing invalid prediction file while trying to get liblinear confidences!" << endl;
								fh.close();
								return 1;
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
						cerr << "[Error] number of classes found in prediction file while trying to get liblinear confidences is not correct!" << endl;
						cerr << "\texpected: " << numClasses << endl;
						cerr << "\tfound: " << numClassesFound << endl;
						cerr << "\tglobal: " << Global::settings->CLASSES.numClasses() << endl;

						cerr << "[Error] parsing invalid prediction file while trying to get liblinear confidences!" << endl;
						fh.close();
						return 2;
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
							Y.graph.nodesData(nodeIndex-1, 1) = atoi(token.c_str());
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
			return 0;
		}
		else
		{
			cerr << "[Error] cannot open file for reading LIBLINEAR/LIBSVM confidences!" << endl;
			return 3;
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

	/**************** Successor Functions ****************/

	const int FlipbitSuccessor::NUM_TOP_LABELS_KEEP = 2;
	const double FlipbitSuccessor::BINARY_CONFIDENCE_THRESHOLD = 0.75;

	FlipbitSuccessor::FlipbitSuccessor()
	{
	}

	FlipbitSuccessor::~FlipbitSuccessor()
	{
	}
	
	vector< ImgLabeling > FlipbitSuccessor::generateSuccessors(ImgFeatures& X, ImgLabeling& YPred)
	{
		// This implements "smart" flip bit successors

		vector< ImgLabeling > successors = vector< ImgLabeling >();

		// for all nodes
		const int numNodes = YPred.getNumNodes();
		for (int node = 0; node < numNodes; node++)
		{
			// set up label sets for convenience
			set<int> allLabelsSet = Global::settings->CLASSES.getLabels();
			set<int> foregroundLabelsSet = Global::settings->CLASSES.getForegroundLabels();
			set<int> backgroundLabelsSet = Global::settings->CLASSES.getBackgroundLabels();

			// variables for current node
			set<int> candidateLabelsSet = set<int>();
			int nodeLabel = YPred.getLabel(node);

			// if confidences are available, use them
			// otherwise generate all labels except for the current node label
			if (YPred.confidencesAvailable)
			{
				VectorXd confidences = YPred.confidences.row(node);

				const int numClasses = allLabelsSet.size();
				if (numClasses == 2)
				{
					// get the other label and check if greater than some confidence threshold
					int otherClassIndex = 1 - Global::settings->CLASSES.getClassIndex(nodeLabel); // this ilookup should return 0 or 1
					if (confidences(otherClassIndex) > BINARY_CONFIDENCE_THRESHOLD)
					{
						candidateLabelsSet.insert(Global::settings->CLASSES.getClassLabel(otherClassIndex));
					}
				}
				else
				{
					// sort by confidences
					mypq_confidences confidencesSorted = mypq_confidences();
					for (int i = 0; i < confidences.size(); i++)
					{
						if (Global::settings->CLASSES.getClassLabel(i) != nodeLabel)
						{
							ConfidenceIndexPair_t pair = ConfidenceIndexPair_t(confidences(i), i);
							confidencesSorted.push(pair);
						}
					}

					// get top K labels
					bool sameLabelFound = false;
					for (int i = 0; i < NUM_TOP_LABELS_KEEP; i++)
					{
						if (confidencesSorted.empty())
							break;

						ConfidenceIndexPair_t topConfidence = confidencesSorted.top();
						confidencesSorted.pop();
						candidateLabelsSet.insert(Global::settings->CLASSES.getClassLabel(topConfidence.second));
					}
				}
			}
			else
			{
				//cout << "successors checkpoint 2.1" << endl;

				// if current node is background, only consider foreground successors
				// otherwise consider everything
				if (backgroundLabelsSet.count(nodeLabel) == 1)
				{
					cout << "successors checkpoint 3.1.1" << endl;

					// if has foreground neighbors then add foreground labels to successors
					// otherwise current node is background and neighbor is background, so no successor
					if (hasForegroundNeighbors(YPred, node))
					{
						candidateLabelsSet = foregroundLabelsSet;
					}
				}
				else
				{
					candidateLabelsSet = allLabelsSet;
					candidateLabelsSet.erase(nodeLabel);
				}
			}

			// don't flip the current node to the same label
			if (candidateLabelsSet.count(nodeLabel) == 1)
			{
				candidateLabelsSet.erase(nodeLabel);
			}

			// for each candidate, add to successors list for returning
			for (set<int>::iterator it = candidateLabelsSet.begin(); it != candidateLabelsSet.end(); ++it)
			{
				int candidateLabel = *it;

				// form successor object
				LabelGraph graphNew = LabelGraph();
				graphNew.nodesData = YPred.graph.nodesData;
				graphNew.nodesData(node) = candidateLabel;
				graphNew.adjList = YPred.graph.adjList;

				ImgLabeling YNew = ImgLabeling();
				YNew.confidences = YPred.confidences;
				YNew.confidencesAvailable = YPred.confidencesAvailable;
				YNew.graph = graphNew;

				// add candidate to successors
				successors.push_back(YNew);
			}
		}

		return successors;
	}

	bool FlipbitSuccessor::hasForegroundNeighbors(ImgLabeling& Y, int node)
	{
		int nodeLabel = Y.getLabel(node);
		NeighborSet_t neighbors = Y.graph.adjList[node];
		const int numNeighbors = neighbors.size();

		bool hasNeighbors = false;

		for (NeighborSet_t::iterator it = neighbors.begin(); it != neighbors.end(); ++it)
		{
			int neighborLabel = Y.getLabel(*it);

			if (!Global::settings->CLASSES.classLabelIsBackground(neighborLabel))
			{
				hasNeighbors = true;
				break;
			}
		}

		return hasNeighbors;
	}

	const double StochasticSuccessor::DEFAULT_T_PARM = 0.5;

	StochasticSuccessor::StochasticSuccessor()
	{
	}

	StochasticSuccessor::~StochasticSuccessor()
	{
	}
	
	vector< ImgLabeling > StochasticSuccessor::generateSuccessors(ImgFeatures& X, ImgLabeling& YPred)
	{
		//TODO
		return vector< ImgLabeling >();
	}

	/**************** Loss Functions ****************/

	HammingLoss::HammingLoss()
	{
	}

	HammingLoss::~HammingLoss()
	{
	}

	double HammingLoss::computeLoss(ImgLabeling& YPred, const ImgLabeling& YTruth)
	{
		Matrix<bool, Dynamic, 1> diff = YPred.graph.nodesData.leftCols(1).array() != YTruth.graph.nodesData.leftCols(1).array();
		double loss = 0.0;
		for (int i = 0; i < diff.size(); i++)
		{
			if (diff(i))
				loss++;
		}
		return loss/diff.size();
	}

	/**************** Search Space ****************/

	SearchSpace::SearchSpace()
	{
		this->heuristicFeatureFunction = NULL;
		this->costFeatureFunction = NULL;
		this->initialPredictionFunction = NULL;
		this->successorFunction = NULL;
		this->lossFunction = NULL;
	}

	SearchSpace::SearchSpace(IFeatureFunction* heuristicFeatureFunction, IFeatureFunction* costFeatureFunction,
		IInitialPredictionFunction* initialPredictionFunction, ISuccessorFunction* successorFunction,
		ILossFunction* lossFunction)
	{
		this->heuristicFeatureFunction = heuristicFeatureFunction;
		this->costFeatureFunction = costFeatureFunction;
		this->initialPredictionFunction = initialPredictionFunction;
		this->successorFunction = successorFunction;
		this->lossFunction = lossFunction;
	}

	SearchSpace::~SearchSpace()
	{
	}

	RankFeatures SearchSpace::computeHeuristicFeatures(ImgFeatures& X, ImgLabeling& Y)
	{
		if (this->heuristicFeatureFunction == NULL)
		{
			cerr << "[ERROR] heuristic feature function is null" << endl;
			exit(1);
		}

		return this->heuristicFeatureFunction->computeFeatures(X, Y);
	}

	RankFeatures SearchSpace::computeCostFeatures(ImgFeatures& X, ImgLabeling& Y)
	{
		if (this->costFeatureFunction == NULL)
		{
			cerr << "[ERROR] cost feature function is null" << endl;
			exit(1);
		}

		return this->costFeatureFunction->computeFeatures(X, Y);
	}

	ImgLabeling SearchSpace::getInitialPrediction(ImgFeatures& X)
	{
		if (this->initialPredictionFunction == NULL)
		{
			cerr << "[ERROR] initial pred feature function is null" << endl;
			exit(1);
		}

		return this->initialPredictionFunction->getInitialPrediction(X);
	}

	vector< ImgLabeling > SearchSpace::generateSuccessors(ImgFeatures& X, ImgLabeling& YPred)
	{
		if (this->successorFunction == NULL)
		{
			cerr << "[ERROR] successor function is null" << endl;
			exit(1);
		}

		return this->successorFunction->generateSuccessors(X, YPred);
	}

	double SearchSpace::computeLoss(ImgLabeling& YPred, const ImgLabeling& YTruth)
	{
		if (this->lossFunction == NULL)
		{
			cerr << "[ERROR] loss function is null" << endl;
			exit(1);
		}

		return this->lossFunction->computeLoss(YPred, YTruth);
	}
}