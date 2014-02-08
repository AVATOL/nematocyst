#include <iostream>
#include <vector>
#include <ctime>
#include "MyFileSystem.hpp"
#include "SearchSpace.hpp"
#include "Settings.hpp"
#include "Globals.hpp"
#include "MyLogger.hpp"
#include "MyGraphAlgorithms.hpp"

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

	const int ISuccessorFunction::MAX_NUM_SUCCESSOR_CANDIDATES = 1000;

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
			classIndex = Global::settings->CLASSES.getClassIndex(nodeLabel1);

			// assignment
			return expnegdiffabs2;
		}
	}

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

	/**************** Successor Functions ****************/

	/**************** Flipbit Successor Function ****************/

	const int FlipbitSuccessor::NUM_TOP_LABELS_KEEP = 2;
	const double FlipbitSuccessor::BINARY_CONFIDENCE_THRESHOLD = 0.75;

	FlipbitSuccessor::FlipbitSuccessor()
	{
		this->maxNumSuccessorCandidates = MAX_NUM_SUCCESSOR_CANDIDATES;
	}

	FlipbitSuccessor::FlipbitSuccessor(int maxNumSuccessorCandidates)
	{
		this->maxNumSuccessorCandidates = maxNumSuccessorCandidates;
	}

	FlipbitSuccessor::~FlipbitSuccessor()
	{
	}
	
	vector< ImgLabeling > FlipbitSuccessor::generateSuccessors(ImgFeatures& X, ImgLabeling& YPred)
	{
		clock_t tic = clock();

		vector<ImgLabeling> successors;

		// for all nodes
		const int numNodes = YPred.getNumNodes();
		for (int node = 0; node < numNodes; node++)
		{
			// set up candidate label set
			set<int> candidateLabelsSet;
			int nodeLabel = YPred.getLabel(node);
			candidateLabelsSet.insert(nodeLabel);

			// flip to any possible class
			candidateLabelsSet = Global::settings->CLASSES.getLabels();

			candidateLabelsSet.erase(nodeLabel); // do not flip to same label

			// for each candidate label, add to successors list for returning
			for (set<int>::iterator it2 = candidateLabelsSet.begin(); it2 != candidateLabelsSet.end(); ++it2)
			{
				int candidateLabel = *it2;

				// form successor object
				LabelGraph graphNew;
				graphNew.nodesData = YPred.graph.nodesData;
				graphNew.nodesData(node) = candidateLabel; // flip bit node
				graphNew.adjList = YPred.graph.adjList;

				ImgLabeling YNew;
				YNew.confidences = YPred.confidences;
				YNew.confidencesAvailable = YPred.confidencesAvailable;
				YNew.graph = graphNew;

				// add candidate to successors
				successors.push_back(YNew);
			}
		}

		// prune to the bound
		const int originalSize = successors.size();
		if (originalSize > maxNumSuccessorCandidates)
		{
			random_shuffle(successors.begin(), successors.end());
			for (int i = 0; i < originalSize - maxNumSuccessorCandidates; i++)
				successors.pop_back();
		}

		LOG() << "num successors=" << successors.size() << endl;

		clock_t toc = clock();
		LOG() << "successor total time: " << (double)(toc - tic)/CLOCKS_PER_SEC << endl;

		return successors;
	}

	/**************** Flipbit Neighbor Successor Function ****************/

	FlipbitNeighborSuccessor::FlipbitNeighborSuccessor()
	{
		this->maxNumSuccessorCandidates = MAX_NUM_SUCCESSOR_CANDIDATES;
	}

	FlipbitNeighborSuccessor::FlipbitNeighborSuccessor(int maxNumSuccessorCandidates)
	{
		this->maxNumSuccessorCandidates = maxNumSuccessorCandidates;
	}

	FlipbitNeighborSuccessor::~FlipbitNeighborSuccessor()
	{
	}
	
	vector< ImgLabeling > FlipbitNeighborSuccessor::generateSuccessors(ImgFeatures& X, ImgLabeling& YPred)
	{
		clock_t tic = clock();

		vector<ImgLabeling> successors;

		// for all nodes
		const int numNodes = YPred.getNumNodes();
		for (int node = 0; node < numNodes; node++)
		{
			// set up candidate label set
			set<int> candidateLabelsSet;
			int nodeLabel = YPred.getLabel(node);
			candidateLabelsSet.insert(nodeLabel);

			if (YPred.hasNeighbors(node))
			{
				// add only neighboring labels to candidate label set
				set<int> neighborLabels = YPred.getNeighborLabels(node);
				for (set<int>::iterator it2 = neighborLabels.begin(); 
					it2 != neighborLabels.end(); ++it2)
				{
					candidateLabelsSet.insert(*it2);
				}
			}
			else
			{
				// if node is isolated without neighbors, then flip to any possible class
				candidateLabelsSet = Global::settings->CLASSES.getLabels();
			}

			candidateLabelsSet.erase(nodeLabel); // do not flip to same label

			// for each candidate label, add to successors list for returning
			for (set<int>::iterator it2 = candidateLabelsSet.begin(); it2 != candidateLabelsSet.end(); ++it2)
			{
				int candidateLabel = *it2;

				// form successor object
				LabelGraph graphNew;
				graphNew.nodesData = YPred.graph.nodesData;
				graphNew.nodesData(node) = candidateLabel; // flip bit node
				graphNew.adjList = YPred.graph.adjList;

				ImgLabeling YNew;
				YNew.confidences = YPred.confidences;
				YNew.confidencesAvailable = YPred.confidencesAvailable;
				YNew.graph = graphNew;

				// add candidate to successors
				successors.push_back(YNew);
			}
		}

		// prune to the bound
		const int originalSize = successors.size();
		if (originalSize > maxNumSuccessorCandidates)
		{
			random_shuffle(successors.begin(), successors.end());
			for (int i = 0; i < originalSize - maxNumSuccessorCandidates; i++)
				successors.pop_back();
		}

		LOG() << "num successors=" << successors.size() << endl;

		clock_t toc = clock();
		LOG() << "successor total time: " << (double)(toc - tic)/CLOCKS_PER_SEC << endl;

		return successors;
	}

	/**************** Stochastic Successor Function ****************/

	const double StochasticSuccessor::DEFAULT_T_PARM = 0.5;

	StochasticSuccessor::StochasticSuccessor()
	{
		this->maxNumSuccessorCandidates = MAX_NUM_SUCCESSOR_CANDIDATES;
		this->cutParam = DEFAULT_T_PARM;
		this->cutEdgesIndependently = true;
	}

	StochasticSuccessor::StochasticSuccessor(bool cutEdgesIndependently, double cutParam, int maxNumSuccessorCandidates)
	{
		this->maxNumSuccessorCandidates = maxNumSuccessorCandidates;
		this->cutParam = cutParam;
		this->cutEdgesIndependently = cutEdgesIndependently;
	}

	StochasticSuccessor::~StochasticSuccessor()
	{
	}
	
	vector< ImgLabeling > StochasticSuccessor::generateSuccessors(ImgFeatures& X, ImgLabeling& YPred)
	{
		clock_t tic = clock();

		// generate random threshold
		double threshold = Rand::unifDist(); // ~ Uniform(0, 1)
		LOG() << "Using threshold=" << threshold << endl;

		// perform cut
		MyGraphAlgorithms::SubgraphSet* subgraphs = cutEdges(X, YPred, threshold, this->cutParam);

		LOG() << "generating successors..." << endl;

		// generate candidates
		vector< ImgLabeling > successors = createCandidates(YPred, subgraphs);

		// prune to the bound
		const int originalSize = successors.size();
		if (originalSize > maxNumSuccessorCandidates)
		{
			random_shuffle(successors.begin(), successors.end());
			for (int i = 0; i < originalSize - maxNumSuccessorCandidates; i++)
				successors.pop_back();
		}

		LOG() << "num successors=" << successors.size() << endl;
		delete subgraphs;

		clock_t toc = clock();
		LOG() << "successor total time: " << (double)(toc - tic)/CLOCKS_PER_SEC << endl;

		return successors;
	}

	MyGraphAlgorithms::SubgraphSet* StochasticSuccessor::cutEdges(ImgFeatures& X, ImgLabeling& YPred, double threshold, double T)
	{
		const int numNodes = X.getNumNodes();
		map< int, set<int> > edges = YPred.graph.adjList;

		// store new cut edges
		map< int, set<int> > cutEdges;

		// convert to format storing (node1, node2) pairs
		vector< MyPrimitives::Pair< int, int > > edgeNodes;

		// edge weights using KL divergence measure
		vector<double> edgeWeights;

		// iterate over all edges to store
		for (map< int, set<int> >::iterator it = edges.begin();
			it != edges.end(); ++it)
		{
			int node1 = it->first;
			set<int> neighbors = it->second;
		
			// loop over neighbors
			for (set<int>::iterator it2 = neighbors.begin(); it2 != neighbors.end(); ++it2)
			{
				int node2 = *it2;

				// get features and labels
				VectorXd nodeFeatures1 = X.graph.nodesData.row(node1);
				VectorXd nodeFeatures2 = X.graph.nodesData.row(node2);

				// compute weights already
				double weight = exp( -(computeKL(nodeFeatures1, nodeFeatures2) + computeKL(nodeFeatures2, nodeFeatures1))*T/2 );
				edgeWeights.push_back(weight);

				// add
				MyPrimitives::Pair< int, int> nodePair = MyPrimitives::Pair< int, int >(node1, node2);
				edgeNodes.push_back(nodePair);
			}
		}

		// given the edge weights, do the actual cutting!
		const int numEdges = edgeNodes.size();
		for (int i = 0; i < numEdges; i++)
		{
			MyPrimitives::Pair< int, int > nodePair = edgeNodes[i];
			int node1 = nodePair.first;
			int node2 = nodePair.second;

			bool decideToCut;
			if (!cutEdgesIndependently)
			{
				// uniform state
				decideToCut = edgeWeights[i] <= threshold;
			}
			else
			{
				// bernoulli independent
				double biasedCoin = Rand::unifDist(); // ~ Uniform(0, 1)
				decideToCut = biasedCoin <= 1-edgeWeights[i];
			}

			if (!decideToCut)
			{
				// keep these uncut edges
				if (cutEdges.count(node1) == 0)
				{
					cutEdges[node1] = set<int>();
				}
				if (cutEdges.count(node2) == 0)
				{
					cutEdges[node2] = set<int>();
				}
				cutEdges[node1].insert(node2);
				cutEdges[node2].insert(node1);
			}
		}

		// create subgraphs
		ImgLabeling Ycopy;
		Ycopy.confidences = YPred.confidences;
		Ycopy.confidencesAvailable = YPred.confidencesAvailable;
		Ycopy.graph = YPred.graph;

		LOG() << "Getting subgraphs..." << endl;

		MyGraphAlgorithms::SubgraphSet* subgraphs = new MyGraphAlgorithms::SubgraphSet(Ycopy, cutEdges);

		return subgraphs;
	}

	vector< ImgLabeling > StochasticSuccessor::createCandidates(ImgLabeling& YPred, MyGraphAlgorithms::SubgraphSet* subgraphs)
	{
		using namespace MyGraphAlgorithms;

		vector< Subgraph* > subgraphset = subgraphs->getSubgraphs();

		// successors set
		vector< ImgLabeling > successors;

		// loop over each sub graph
		for (vector< Subgraph* >::iterator it = subgraphset.begin(); it != subgraphset.end(); ++it)
		{
			Subgraph* sub = *it;
			vector< ConnectedComponent* > ccset = sub->getConnectedComponents();

			// loop over each connected component
			for (vector< ConnectedComponent* >::iterator it2 = ccset.begin(); it2 != ccset.end(); ++it2)
			{
				ConnectedComponent* cc = *it2;

				set<int> candidateLabelsSet;
				int nodeLabel = cc->getLabel();
				candidateLabelsSet.insert(nodeLabel);

				// flip to any possible class
				candidateLabelsSet = Global::settings->CLASSES.getLabels();

				candidateLabelsSet.erase(nodeLabel);

				// loop over each candidate label
				for (set<int>::iterator it3 = candidateLabelsSet.begin(); it3 != candidateLabelsSet.end(); ++it3)
				{
					int label = *it3;

					// form successor object
					ImgLabeling YNew;
					YNew.confidences = YPred.confidences;
					YNew.confidencesAvailable = YPred.confidencesAvailable;
					YNew.stochasticCuts = subgraphs->getCuts();
					YNew.stochasticCutsAvailable = true;
					YNew.graph = YPred.graph;

					// make changes
					set<int> component = cc->getNodes();
					for (set<int>::iterator it4 = component.begin(); it4 != component.end(); ++it4)
					{
						int node = *it4;
						YNew.graph.nodesData(node) = label;
					}

					successors.push_back(YNew);
				}
			}
		}

		return successors;
	}

	double StochasticSuccessor::computeKL(const VectorXd& p, const VectorXd& q)
	{
		if (p.size() != q.size())
		{
			LOG(WARNING) << "dimensions of p and q are not the same!" << endl;
		}

		double KL = 0;

		VectorXd pn = p.normalized();
		VectorXd qn = q.normalized();

		for (int i = 0; i < p.size(); i++)
		{
			if (p(i) != 0)
			{
				if (q(i) == 0) // && p(i) != 0
				{
					// warning: q(i) = 0 does not imply p(i) = 0
				}
				else
				{
					KL += p(i) * log(p(i) / q(i));
				}
			}
		}

		return KL;
	}

	/**************** Stochastic Neighbor Successor Function ****************/

	StochasticNeighborSuccessor::StochasticNeighborSuccessor()
	{
		this->maxNumSuccessorCandidates = MAX_NUM_SUCCESSOR_CANDIDATES;
		this->cutParam = DEFAULT_T_PARM;
		this->cutEdgesIndependently = true;
	}

	StochasticNeighborSuccessor::StochasticNeighborSuccessor(bool cutEdgesIndependently, double cutParam, int maxNumSuccessorCandidates)
	{
		this->maxNumSuccessorCandidates = maxNumSuccessorCandidates;
		this->cutParam = cutParam;
		this->cutEdgesIndependently = cutEdgesIndependently;
	}

	StochasticNeighborSuccessor::~StochasticNeighborSuccessor()
	{
	}

	vector< ImgLabeling > StochasticNeighborSuccessor::createCandidates(ImgLabeling& YPred, MyGraphAlgorithms::SubgraphSet* subgraphs)
	{
		using namespace MyGraphAlgorithms;

		vector< Subgraph* > subgraphset = subgraphs->getSubgraphs();

		// successors set
		vector< ImgLabeling > successors;

		// loop over each sub graph
		for (vector< Subgraph* >::iterator it = subgraphset.begin(); it != subgraphset.end(); ++it)
		{
			Subgraph* sub = *it;
			vector< ConnectedComponent* > ccset = sub->getConnectedComponents();

			// loop over each connected component
			for (vector< ConnectedComponent* >::iterator it2 = ccset.begin(); it2 != ccset.end(); ++it2)
			{
				ConnectedComponent* cc = *it2;

				set<int> candidateLabelsSet;
				int nodeLabel = cc->getLabel();
				candidateLabelsSet.insert(nodeLabel);
				if (cc->hasNeighbors())
				{
					// add only neighboring labels to candidate label set
					candidateLabelsSet = cc->getNeighborLabels();
				}
				else
				{
					// if connected component is isolated without neighboring connected components, then flip to any possible class
					candidateLabelsSet = Global::settings->CLASSES.getLabels();
				}
				candidateLabelsSet.erase(nodeLabel);

				// loop over each candidate label
				for (set<int>::iterator it3 = candidateLabelsSet.begin(); it3 != candidateLabelsSet.end(); ++it3)
				{
					int label = *it3;

					// form successor object
					ImgLabeling YNew;
					YNew.confidences = YPred.confidences;
					YNew.confidencesAvailable = YPred.confidencesAvailable;
					YNew.stochasticCuts = subgraphs->getCuts();
					YNew.stochasticCutsAvailable = true;
					YNew.graph = YPred.graph;

					// make changes
					set<int> component = cc->getNodes();
					for (set<int>::iterator it4 = component.begin(); it4 != component.end(); ++it4)
					{
						int node = *it4;
						YNew.graph.nodesData(node) = label;
					}

					successors.push_back(YNew);
				}
			}
		}

		return successors;
	}

	/**************** Cut Schedule Neighbor Successor Function ****************/

	const int CutScheduleNeighborSuccessor::NUM_GOOD_SUBGRAPHS_THRESHOLD = 8;
	const double CutScheduleNeighborSuccessor::FINAL_THRESHOLD = 0.975;
	const double CutScheduleNeighborSuccessor::THRESHOLD_INCREMENT = 0.025;

	CutScheduleNeighborSuccessor::CutScheduleNeighborSuccessor()
	{
		this->maxNumSuccessorCandidates = MAX_NUM_SUCCESSOR_CANDIDATES;
		this->cutParam = DEFAULT_T_PARM;
		this->cutEdgesIndependently = false;
	}

	CutScheduleNeighborSuccessor::CutScheduleNeighborSuccessor(double cutParam, int maxNumSuccessorCandidates)
	{
		this->maxNumSuccessorCandidates = maxNumSuccessorCandidates;
		this->cutParam = cutParam;
		this->cutEdgesIndependently = false;
	}

	CutScheduleNeighborSuccessor::~CutScheduleNeighborSuccessor()
	{
	}
	
	MyGraphAlgorithms::SubgraphSet* CutScheduleNeighborSuccessor::cutEdges(ImgFeatures& X, ImgLabeling& YPred, double threshold, double T)
	{
		MyGraphAlgorithms::SubgraphSet* subgraphs = NULL;

		const int numNodes = X.getNumNodes();
		map< int, set<int> > edges = YPred.graph.adjList;

		// convert to format storing (node1, node2) pairs
		vector< MyPrimitives::Pair< int, int > > edgeNodes;

		// edge weights using KL divergence measure
		vector<double> edgeWeights;

		// iterate over all edges to store
		for (map< int, set<int> >::iterator it = edges.begin();
			it != edges.end(); ++it)
		{
			int node1 = it->first;
			set<int> neighbors = it->second;
		
			// loop over neighbors
			for (set<int>::iterator it2 = neighbors.begin(); it2 != neighbors.end(); ++it2)
			{
				int node2 = *it2;

				// get features and labels
				VectorXd nodeFeatures1 = X.graph.nodesData.row(node1);
				VectorXd nodeFeatures2 = X.graph.nodesData.row(node2);

				// compute weights already
				double weight = exp( -(computeKL(nodeFeatures1, nodeFeatures2) + computeKL(nodeFeatures2, nodeFeatures1))*T/2 );
				edgeWeights.push_back(weight);

				// add
				MyPrimitives::Pair< int, int> nodePair = MyPrimitives::Pair< int, int >(node1, node2);
				edgeNodes.push_back(nodePair);
			}
		}
		
		// increase threshold until good cuts; also cut by state
		for (double thresholdAttempt = threshold; thresholdAttempt <= 1.0; thresholdAttempt += THRESHOLD_INCREMENT)
		{
			// store new cut edges
			map< int, set<int> > cutEdges;

			LOG() << "Attempting threshold=" << thresholdAttempt << endl;
			// given the edge weights, do the actual cutting!
			const int numEdges = edgeNodes.size();
			for (int i = 0; i < numEdges; i++)
			{
				MyPrimitives::Pair< int, int > nodePair = edgeNodes[i];
				int node1 = nodePair.first;
				int node2 = nodePair.second;

				bool decideToCut;
				if (!cutEdgesIndependently)
				{
					// uniform state
					decideToCut = edgeWeights[i] <= threshold;
				}
				else
				{
					// bernoulli independent
					double biasedCoin = Rand::unifDist(); // ~ Uniform(0, 1)
					decideToCut = biasedCoin <= 1-edgeWeights[i];
				}

				if (!decideToCut)
				{
					// keep these uncut edges
					if (cutEdges.count(node1) == 0)
					{
						cutEdges[node1] = set<int>();
					}
					if (cutEdges.count(node2) == 0)
					{
						cutEdges[node2] = set<int>();
					}
					cutEdges[node1].insert(node2);
					cutEdges[node2].insert(node1);
				}
			}

			// create subgraphs
			ImgLabeling Ycopy;
			Ycopy.confidences = YPred.confidences;
			Ycopy.confidencesAvailable = YPred.confidencesAvailable;
			Ycopy.graph = YPred.graph;

			MyGraphAlgorithms::SubgraphSet* subgraphsTemp = new MyGraphAlgorithms::SubgraphSet(Ycopy, cutEdges);

			LOG() << "\tnum exactly one positive cc subgraphs=" << subgraphsTemp->getExactlyOnePositiveCCSubgraphs().size() << endl;
			if (subgraphsTemp->getExactlyOnePositiveCCSubgraphs().size() > NUM_GOOD_SUBGRAPHS_THRESHOLD)
			{
				subgraphs = subgraphsTemp;
				break;
			}

			if (thresholdAttempt >= FINAL_THRESHOLD)
			{
				cout << "reached final threshold" << endl;
				subgraphs = subgraphsTemp;
				break;
			}

			delete subgraphsTemp;
		}

		return subgraphs;
	}

	vector< ImgLabeling > CutScheduleNeighborSuccessor::createCandidates(ImgLabeling& YPred, MyGraphAlgorithms::SubgraphSet* subgraphs)
	{
		using namespace MyGraphAlgorithms;

		vector< Subgraph* > subgraphset = subgraphs->getExactlyOnePositiveCCSubgraphs();

		// successors set
		vector< ImgLabeling > successors;

		// loop over each sub graph
		for (vector< Subgraph* >::iterator it = subgraphset.begin(); it != subgraphset.end(); ++it)
		{
			Subgraph* sub = *it;
			vector< ConnectedComponent* > ccset = sub->getConnectedComponents();

			// loop over each connected component
			for (vector< ConnectedComponent* >::iterator it2 = ccset.begin(); it2 != ccset.end(); ++it2)
			{
				ConnectedComponent* cc = *it2;

				set<int> candidateLabelsSet;
				int nodeLabel = cc->getLabel();
				candidateLabelsSet.insert(nodeLabel);
				if (cc->hasNeighbors())
				{
					// add only neighboring labels to candidate label set
					candidateLabelsSet = cc->getNeighborLabels();
				}
				else
				{
					// if connected component is isolated without neighboring connected components, then flip to any possible class
					candidateLabelsSet = Global::settings->CLASSES.getLabels();
				}
				candidateLabelsSet.erase(nodeLabel);

				// loop over each candidate label
				for (set<int>::iterator it3 = candidateLabelsSet.begin(); it3 != candidateLabelsSet.end(); ++it3)
				{
					int label = *it3;

					// form successor object
					ImgLabeling YNew;
					YNew.confidences = YPred.confidences;
					YNew.confidencesAvailable = YPred.confidencesAvailable;
					YNew.stochasticCuts = subgraphs->getCuts();
					YNew.stochasticCutsAvailable = true;
					YNew.graph = YPred.graph;

					// make changes
					set<int> component = cc->getNodes();
					for (set<int>::iterator it4 = component.begin(); it4 != component.end(); ++it4)
					{
						int node = *it4;
						YNew.graph.nodesData(node) = label;
					}

					successors.push_back(YNew);
				}
			}
		}

		return successors;
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
		Matrix<bool, Dynamic, 1> diff = YPred.graph.nodesData.array() != YTruth.graph.nodesData.array();
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
		delete this->heuristicFeatureFunction;
		delete this->costFeatureFunction;
		delete this->initialPredictionFunction;
		delete this->successorFunction;
		delete this->lossFunction;
	}

	RankFeatures SearchSpace::computeHeuristicFeatures(ImgFeatures& X, ImgLabeling& Y)
	{
		if (this->heuristicFeatureFunction == NULL)
		{
			LOG(ERROR) << "heuristic feature function is null";
			abort();
		}

		return this->heuristicFeatureFunction->computeFeatures(X, Y);
	}

	RankFeatures SearchSpace::computeCostFeatures(ImgFeatures& X, ImgLabeling& Y)
	{
		if (this->costFeatureFunction == NULL)
		{
			LOG(ERROR) << "cost feature function is null";
			abort();
		}

		return this->costFeatureFunction->computeFeatures(X, Y);
	}

	ImgLabeling SearchSpace::getInitialPrediction(ImgFeatures& X)
	{
		if (this->initialPredictionFunction == NULL)
		{
			LOG(ERROR) << "initial pred feature function is null";
			abort();
		}

		return this->initialPredictionFunction->getInitialPrediction(X);
	}

	vector< ImgLabeling > SearchSpace::generateSuccessors(ImgFeatures& X, ImgLabeling& YPred)
	{
		if (this->successorFunction == NULL)
		{
			LOG(ERROR) << "successor function is null";
			abort();
		}

		return this->successorFunction->generateSuccessors(X, YPred);
	}

	double SearchSpace::computeLoss(ImgLabeling& YPred, const ImgLabeling& YTruth)
	{
		if (this->lossFunction == NULL)
		{
			LOG(ERROR) << "loss function is null";
			abort();
		}

		return this->lossFunction->computeLoss(YPred, YTruth);
	}
}