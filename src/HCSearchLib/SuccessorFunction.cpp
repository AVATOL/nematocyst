#include "SuccessorFunction.hpp"
#include "Globals.hpp"

namespace HCSearch
{
	/**************** Successor Functions ****************/

	/**************** Flipbit Successor Function ****************/

	const double FlipbitSuccessor::TOP_CONFIDENCES_PROPORTION = 0.5;
	const int FlipbitSuccessor::NUM_TOP_LABELS_KEEP = 2;
	const double FlipbitSuccessor::BINARY_CONFIDENCE_THRESHOLD = 0.75;

	FlipbitSuccessor::FlipbitSuccessor()
	{
	}

	FlipbitSuccessor::~FlipbitSuccessor()
	{
	}
	
	vector< ImgCandidate > FlipbitSuccessor::generateSuccessors(ImgFeatures& X, ImgLabeling& YPred, int timeStep, int timeBound)
	{
		clock_t tic = clock();

		vector<ImgCandidate> successors;

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

				ImgCandidate YCandidate;
				YCandidate.labeling = YNew;
				YCandidate.action = set<int>();
				YCandidate.action.insert(node);

				// add candidate to successors
				successors.push_back(YCandidate);
			}
		}

		LOG() << "num successors generated=" << successors.size() << endl;

		Global::settings->stats->addSuccessorCount(successors.size());

		clock_t toc = clock();
		LOG() << "successor total time: " << (double)(toc - tic)/CLOCKS_PER_SEC << endl;

		return successors;
	}

	/**************** Flipbit Neighbor Successor Function ****************/

	FlipbitNeighborSuccessor::FlipbitNeighborSuccessor()
	{
	}

	FlipbitNeighborSuccessor::~FlipbitNeighborSuccessor()
	{
	}
	
	vector< ImgCandidate > FlipbitNeighborSuccessor::generateSuccessors(ImgFeatures& X, ImgLabeling& YPred, int timeStep, int timeBound)
	{
		clock_t tic = clock();

		vector<ImgCandidate> successors;

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

				ImgCandidate YCandidate;
				YCandidate.labeling = YNew;
				YCandidate.action = set<int>();
				YCandidate.action.insert(node);

				// add candidate to successors
				successors.push_back(YCandidate);
			}
		}

		LOG() << "num successors generated=" << successors.size() << endl;

		Global::settings->stats->addSuccessorCount(successors.size());

		clock_t toc = clock();
		LOG() << "successor total time: " << (double)(toc - tic)/CLOCKS_PER_SEC << endl;

		return successors;
	}

	/**************** Flipbit Confidences Neighbor Successor Function ****************/

	FlipbitConfidencesNeighborSuccessor::FlipbitConfidencesNeighborSuccessor()
	{
	}

	FlipbitConfidencesNeighborSuccessor::~FlipbitConfidencesNeighborSuccessor()
	{
	}
	
	vector< ImgCandidate > FlipbitConfidencesNeighborSuccessor::generateSuccessors(ImgFeatures& X, ImgLabeling& YPred, int timeStep, int timeBound)
	{
		clock_t tic = clock();

		vector<ImgCandidate> successors;

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

				int topKConfidences = static_cast<int>(ceil(TOP_CONFIDENCES_PROPORTION * Global::settings->CLASSES.numClasses()));
				set<int> confidentSet = YPred.getTopConfidentLabels(node, topKConfidences);
				candidateLabelsSet.insert(confidentSet.begin(), confidentSet.end());
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

				ImgCandidate YCandidate;
				YCandidate.labeling = YNew;
				YCandidate.action = set<int>();
				YCandidate.action.insert(node);

				// add candidate to successors
				successors.push_back(YCandidate);
			}
		}

		LOG() << "num successors generated=" << successors.size() << endl;

		Global::settings->stats->addSuccessorCount(successors.size());

		clock_t toc = clock();
		LOG() << "successor total time: " << (double)(toc - tic)/CLOCKS_PER_SEC << endl;

		return successors;
	}

	/**************** Stochastic Successor Function ****************/

	const double StochasticSuccessor::TOP_CONFIDENCES_PROPORTION = 0.5;
	const double StochasticSuccessor::DEFAULT_T_PARM = 0.5;
	const double StochasticSuccessor::DEFAULT_MAX_THRESHOLD = 1.0;
	const double StochasticSuccessor::DEFAULT_MIN_THRESHOLD = 0.0;

	StochasticSuccessor::StochasticSuccessor()
	{
		this->cutParam = DEFAULT_T_PARM;
		this->cutEdgesIndependently = true;
		this->maxThreshold = DEFAULT_MAX_THRESHOLD;
		this->minThreshold = DEFAULT_MIN_THRESHOLD;
	}

	StochasticSuccessor::StochasticSuccessor(bool cutEdgesIndependently, double cutParam)
	{
		this->cutParam = cutParam;
		this->cutEdgesIndependently = cutEdgesIndependently;
		this->maxThreshold = DEFAULT_MAX_THRESHOLD;
		this->minThreshold = DEFAULT_MIN_THRESHOLD;
	}
	StochasticSuccessor::StochasticSuccessor(bool cutEdgesIndependently, double cutParam, double maxThreshold, double minThreshold)
	{
		this->cutParam = cutParam;
		this->cutEdgesIndependently = cutEdgesIndependently;
		this->maxThreshold = maxThreshold;
		this->minThreshold = minThreshold;
	}

	StochasticSuccessor::~StochasticSuccessor()
	{
	}
	
	vector< ImgCandidate > StochasticSuccessor::generateSuccessors(ImgFeatures& X, ImgLabeling& YPred, int timeStep, int timeBound)
	{
		clock_t tic = clock();

		// generate random threshold
		double threshold = Rand::unifDist(); // ~ Uniform(0, 1)
		threshold = threshold*(this->maxThreshold - this->minThreshold) + this->minThreshold;

		if (!this->cutEdgesIndependently)
			LOG() << "Cutting edges by state... Using threshold=" << threshold << endl;
		else
			LOG() << "Cutting edges independently..." << endl;

		// perform cut
		MyGraphAlgorithms::SubgraphSet* subgraphs = cutEdges(X, YPred, threshold, this->cutParam);

		LOG() << "generating stochastic successors..." << endl;

		// generate candidates
		vector< ImgCandidate > successors = createCandidates(YPred, subgraphs);

		LOG() << "num successors generated=" << successors.size() << endl;

		Global::settings->stats->addSuccessorCount(successors.size());

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
		if (X.edgeWeightsAvailable)
		{
			for (map< MyPrimitives::Pair<int, int>, double >::iterator it = X.edgeWeights.begin(); it != X.edgeWeights.end(); ++it)
			{
				// get weights
				double weight = it->second;
				edgeWeights.push_back(weight);

				// add
				MyPrimitives::Pair<int, int> key = it->first;
				edgeNodes.push_back(key);
			}
		}
		else
		{
			LOG() << "Computing edge weights from nodes..." << endl;

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
		}

		// given the edge weights, do the actual cutting!
		const int numEdges = edgeNodes.size();
		for (int i = 0; i < numEdges; i++)
		{
			MyPrimitives::Pair< int, int > nodePair = edgeNodes[i];
			int node1 = nodePair.first;
			int node2 = nodePair.second;

			bool decideToCut;
			//if (YPred.getLabel(node1) != YPred.getLabel(node2))
			//{
			//	// different node labels: always cut
			//	decideToCut = true;
			//}
			//else if (!cutEdgesIndependently)
			if (!cutEdgesIndependently)
			{
				// uniform state
				//decideToCut = edgeWeights[i] <= threshold;
				decideToCut =  threshold <= edgeWeights[i];
			}
			else
			{
				// bernoulli independent
				double biasedCoin = Rand::unifDist(); // ~ Uniform(0, 1)
				//decideToCut = biasedCoin <= 1-edgeWeights[i];
				decideToCut = biasedCoin <= edgeWeights[i];
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

	vector< ImgCandidate > StochasticSuccessor::createCandidates(ImgLabeling& YPred, MyGraphAlgorithms::SubgraphSet* subgraphs)
	{
		using namespace MyGraphAlgorithms;

		vector< Subgraph* > subgraphset = subgraphs->getSubgraphs();

		// shuffle
		std::random_shuffle(subgraphset.begin(), subgraphset.end());

		// successors set
		vector< ImgCandidate > successors;

		// loop over each sub graph
		int cumSumLabels = 0;
		int numSumLabels = 0;
		for (vector< Subgraph* >::iterator it = subgraphset.begin(); it != subgraphset.end(); ++it)
		{
			Subgraph* sub = *it;
			vector< ConnectedComponent* > ccset = sub->getConnectedComponents();

			// shuffle
			std::random_shuffle(ccset.begin(), ccset.end());

			// loop over each connected component
			for (vector< ConnectedComponent* >::iterator it2 = ccset.begin(); it2 != ccset.end(); ++it2)
			{
				ConnectedComponent* cc = *it2;

				set<int> candidateLabelsSet;
				int nodeLabel = cc->getLabel();
				candidateLabelsSet.insert(nodeLabel);
				
				// get labels
				getLabels(candidateLabelsSet, cc);
				
				candidateLabelsSet.erase(nodeLabel);

				cumSumLabels += candidateLabelsSet.size();
				numSumLabels++;

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
					set<int> action;
					for (set<int>::iterator it4 = component.begin(); it4 != component.end(); ++it4)
					{
						int node = *it4;
						YNew.graph.nodesData(node) = label;
						action.insert(node);
					}

					ImgCandidate YCandidate;
					YCandidate.labeling = YNew;
					YCandidate.action = action;

					successors.push_back(YCandidate);
				}
			}
		}

		if (numSumLabels > 0)
			LOG() << "average num labels=" << (1.0*cumSumLabels/numSumLabels) << endl;

		return successors;
	}

	void StochasticSuccessor::getLabels(set<int>& candidateLabelsSet, MyGraphAlgorithms::ConnectedComponent* cc)
	{
		getAllLabels(candidateLabelsSet, cc);	
	}

	void StochasticSuccessor::getAllLabels(set<int>& candidateLabelsSet, MyGraphAlgorithms::ConnectedComponent* cc)
	{
		// flip to any possible class
		candidateLabelsSet = Global::settings->CLASSES.getLabels();
	}

	void StochasticSuccessor::getNeighborLabels(set<int>& candidateLabelsSet, MyGraphAlgorithms::ConnectedComponent* cc)
	{
		if (cc->hasNeighbors())
		{
			// add only neighboring labels to candidate label set
			set<int> neighborSet = cc->getNeighborLabels();
			candidateLabelsSet.insert(neighborSet.begin(), neighborSet.end());
		}
		else
		{
			// if connected component is isolated without neighboring connected components, then flip to any possible class
			candidateLabelsSet = Global::settings->CLASSES.getLabels();
		}
	}

	void StochasticSuccessor::getConfidencesNeighborLabels(set<int>& candidateLabelsSet, MyGraphAlgorithms::ConnectedComponent* cc)
	{
		int topKConfidences = static_cast<int>(ceil(TOP_CONFIDENCES_PROPORTION * Global::settings->CLASSES.numClasses()));
		candidateLabelsSet = cc->getTopConfidentLabels(topKConfidences);
		if (cc->hasNeighbors())
		{
			// add only neighboring labels to candidate label set
			set<int> neighborSet = cc->getNeighborLabels();
			candidateLabelsSet.insert(neighborSet.begin(), neighborSet.end());
		}
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
		this->cutParam = DEFAULT_T_PARM;
		this->cutEdgesIndependently = true;
		this->maxThreshold = DEFAULT_MAX_THRESHOLD;
		this->minThreshold = DEFAULT_MIN_THRESHOLD;
	}
	
	StochasticNeighborSuccessor::StochasticNeighborSuccessor(bool cutEdgesIndependently, double cutParam)
	{
		this->cutParam = cutParam;
		this->cutEdgesIndependently = cutEdgesIndependently;
		this->maxThreshold = DEFAULT_MAX_THRESHOLD;
		this->minThreshold = DEFAULT_MIN_THRESHOLD;
	}

	StochasticNeighborSuccessor::StochasticNeighborSuccessor(bool cutEdgesIndependently, double cutParam, double maxThreshold, double minThreshold)
	{
		this->cutParam = cutParam;
		this->cutEdgesIndependently = cutEdgesIndependently;
		this->maxThreshold = maxThreshold;
		this->minThreshold = minThreshold;
	}

	StochasticNeighborSuccessor::~StochasticNeighborSuccessor()
	{
	}

	void StochasticNeighborSuccessor::getLabels(set<int>& candidateLabelsSet, MyGraphAlgorithms::ConnectedComponent* cc)
	{
		getNeighborLabels(candidateLabelsSet, cc);	
	}

	/**************** Stochastic Confidences Neighbor Successor Function ****************/

	StochasticConfidencesNeighborSuccessor::StochasticConfidencesNeighborSuccessor()
	{
		this->cutParam = DEFAULT_T_PARM;
		this->cutEdgesIndependently = true;
		this->maxThreshold = DEFAULT_MAX_THRESHOLD;
		this->minThreshold = DEFAULT_MIN_THRESHOLD;
	}

	StochasticConfidencesNeighborSuccessor::StochasticConfidencesNeighborSuccessor(bool cutEdgesIndependently, double cutParam)
	{
		this->cutParam = cutParam;
		this->cutEdgesIndependently = cutEdgesIndependently;
		this->maxThreshold = DEFAULT_MAX_THRESHOLD;
		this->minThreshold = DEFAULT_MIN_THRESHOLD;
	}

	StochasticConfidencesNeighborSuccessor::StochasticConfidencesNeighborSuccessor(bool cutEdgesIndependently, double cutParam, double maxThreshold, double minThreshold)
	{
		this->cutParam = cutParam;
		this->cutEdgesIndependently = cutEdgesIndependently;
		this->maxThreshold = maxThreshold;
		this->minThreshold = minThreshold;
	}

	StochasticConfidencesNeighborSuccessor::~StochasticConfidencesNeighborSuccessor()
	{
	}

	void StochasticConfidencesNeighborSuccessor::getLabels(set<int>& candidateLabelsSet, MyGraphAlgorithms::ConnectedComponent* cc)
	{
		getConfidencesNeighborLabels(candidateLabelsSet, cc);	
	}

	/**************** Cut Schedule Successor Function ****************/

	const int CutScheduleSuccessor::NUM_GOOD_SUBGRAPHS_THRESHOLD = 8;
	const double CutScheduleSuccessor::FINAL_THRESHOLD = 0.975;
	const double CutScheduleSuccessor::THRESHOLD_INCREMENT = 0.025;

	CutScheduleSuccessor::CutScheduleSuccessor()
	{
		this->cutParam = DEFAULT_T_PARM;
		this->cutEdgesIndependently = false;
	}

	CutScheduleSuccessor::CutScheduleSuccessor(double cutParam)
	{
		this->cutParam = cutParam;
		this->cutEdgesIndependently = false;
	}

	CutScheduleSuccessor::~CutScheduleSuccessor()
	{
	}
	
	vector< ImgCandidate > CutScheduleSuccessor::generateSuccessors(ImgFeatures& X, ImgLabeling& YPred, int timeStep, int timeBound)
	{
		clock_t tic = clock();

		double threshold = 0.025;

		// perform cut
		MyGraphAlgorithms::SubgraphSet* subgraphs = cutEdges(X, YPred, threshold, this->cutParam);

		LOG() << "generating cut schedule successors..." << endl;

		// generate candidates
		vector< ImgCandidate > successors = createCandidates(YPred, subgraphs);

		LOG() << "num successors generated=" << successors.size() << endl;

		Global::settings->stats->addSuccessorCount(successors.size());

		delete subgraphs;

		clock_t toc = clock();
		LOG() << "successor total time: " << (double)(toc - tic)/CLOCKS_PER_SEC << endl;

		return successors;
	}

	MyGraphAlgorithms::SubgraphSet* CutScheduleSuccessor::cutEdges(ImgFeatures& X, ImgLabeling& YPred, double threshold, double T)
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
					decideToCut = edgeWeights[i] <= thresholdAttempt;
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

	void CutScheduleSuccessor::getLabels(set<int>& candidateLabelsSet, MyGraphAlgorithms::ConnectedComponent* cc)
	{
		getAllLabels(candidateLabelsSet, cc);	
	}

	/**************** Cut Schedule Neighbor Successor Function ****************/

	CutScheduleNeighborSuccessor::CutScheduleNeighborSuccessor()
	{
		this->cutParam = DEFAULT_T_PARM;
		this->cutEdgesIndependently = false;
	}

	CutScheduleNeighborSuccessor::CutScheduleNeighborSuccessor(double cutParam)
	{
		this->cutParam = cutParam;
		this->cutEdgesIndependently = false;
	}

	CutScheduleNeighborSuccessor::~CutScheduleNeighborSuccessor()
	{
	}
	
	void CutScheduleNeighborSuccessor::getLabels(set<int>& candidateLabelsSet, MyGraphAlgorithms::ConnectedComponent* cc)
	{
		getNeighborLabels(candidateLabelsSet, cc);	
	}

	/**************** Cut Schedule Confidences Neighbor Successor Function ****************/

	CutScheduleConfidencesNeighborSuccessor::CutScheduleConfidencesNeighborSuccessor()
	{
		this->cutParam = DEFAULT_T_PARM;
		this->cutEdgesIndependently = false;
	}

	CutScheduleConfidencesNeighborSuccessor::CutScheduleConfidencesNeighborSuccessor(double cutParam)
	{
		this->cutParam = cutParam;
		this->cutEdgesIndependently = false;
	}

	CutScheduleConfidencesNeighborSuccessor::~CutScheduleConfidencesNeighborSuccessor()
	{
	}
	
	void CutScheduleConfidencesNeighborSuccessor::getLabels(set<int>& candidateLabelsSet, MyGraphAlgorithms::ConnectedComponent* cc)
	{
		getConfidencesNeighborLabels(candidateLabelsSet, cc);	
	}

	/**************** Stochastic Schedule Successor Function ****************/

	const double StochasticScheduleSuccessor::TOP_CONFIDENCES_PROPORTION = 0.5;
	const double StochasticScheduleSuccessor::DEFAULT_T_PARAM = 0.5;
	const double StochasticScheduleSuccessor::DEFAULT_NODE_CLAMP_THRESHOLD = 0.9;
	const double StochasticScheduleSuccessor::DEFAULT_EDGE_CLAMP_POSITIVE_THRESHOLD = 0.9;
	const double StochasticScheduleSuccessor::DEFAULT_EDGE_CLAMP_NEGATIVE_THRESHOLD = 0.1;

	StochasticScheduleSuccessor::StochasticScheduleSuccessor()
	{
		this->cutParam = DEFAULT_T_PARAM;
		this->cutEdgesIndependently = true;
		this->clampNodes = false;
		this->clampEdges = false;
		this->nodeClampThreshold = DEFAULT_NODE_CLAMP_THRESHOLD;
		this->edgeClampPositiveThreshold = DEFAULT_EDGE_CLAMP_POSITIVE_THRESHOLD;
		this->edgeClampNegativeThreshold = DEFAULT_EDGE_CLAMP_NEGATIVE_THRESHOLD;
	}

	StochasticScheduleSuccessor::StochasticScheduleSuccessor(bool cutEdgesIndependently, double cutParam)
	{
		this->cutParam = cutParam;
		this->cutEdgesIndependently = cutEdgesIndependently;
		this->clampNodes = false;
		this->clampEdges = false;
		this->nodeClampThreshold = DEFAULT_NODE_CLAMP_THRESHOLD;
		this->edgeClampPositiveThreshold = DEFAULT_EDGE_CLAMP_POSITIVE_THRESHOLD;
		this->edgeClampNegativeThreshold = DEFAULT_EDGE_CLAMP_NEGATIVE_THRESHOLD;
	}

	StochasticScheduleSuccessor::StochasticScheduleSuccessor(bool cutEdgesIndependently, double cutParam, 
	bool clampNodes, bool clampEdges, double nodeClampThreshold, double edgeClampPositiveThreshold, double edgeClampNegativeThreshold)
	{
		this->cutParam = cutParam;
		this->cutEdgesIndependently = cutEdgesIndependently;
		this->clampNodes = clampNodes;
		this->clampEdges = clampEdges;
		this->nodeClampThreshold = nodeClampThreshold;
		this->edgeClampPositiveThreshold = edgeClampPositiveThreshold;
		this->edgeClampNegativeThreshold = edgeClampNegativeThreshold;
	}

	StochasticScheduleSuccessor::~StochasticScheduleSuccessor()
	{
	}
	
	vector< ImgCandidate > StochasticScheduleSuccessor::generateSuccessors(ImgFeatures& X, ImgLabeling& YPred, int timeStep, int timeBound)
	{
		clock_t tic = clock();

		// generate random threshold
		double threshold = Rand::unifDist(); // ~ Uniform(0, 1) schedule
		LOG() << "Using threshold=" << threshold << endl;

		// perform cut
		MyGraphAlgorithms::SubgraphSet* subgraphs = cutEdges(X, YPred, threshold, this->cutParam, timeStep, timeBound);

		LOG() << "generating stochastic schedule successors..." << endl;

		// generate candidates
		vector< ImgCandidate > successors = createCandidates(YPred, subgraphs);

		LOG() << "num successors generated=" << successors.size() << endl;

		Global::settings->stats->addSuccessorCount(successors.size());

		delete subgraphs;

		clock_t toc = clock();
		LOG() << "successor total time: " << (double)(toc - tic)/CLOCKS_PER_SEC << endl;

		return successors;
	}

	MyGraphAlgorithms::SubgraphSet* StochasticScheduleSuccessor::cutEdges(ImgFeatures& X, ImgLabeling& YPred, double threshold, double T, int timeStep, int timeBound)
	{
		const int numNodes = X.getNumNodes();
		map< int, set<int> > edges = YPred.graph.adjList;

		// store new cut edges
		map< int, set<int> > cutEdges;

		// convert to format storing (node1, node2) pairs
		vector< MyPrimitives::Pair< int, int > > edgeNodes;

		// edge weights using KL divergence measure
		vector<double> edgeWeights;

		// keep track of which edges to clamp
		vector<bool> positiveEdgeClamps;
		vector<bool> negativeEdgeClamps;

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

				// for now do not clamp
				// TODO: read weights and make decision with threshold
				positiveEdgeClamps.push_back(false);
				negativeEdgeClamps.push_back(false);

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

			// get clamp decisions
			bool positiveClamp = positiveEdgeClamps[i];
			bool negativeClamp = negativeEdgeClamps[i];

			bool willCut;
			if (positiveClamp && !negativeClamp)
			{
				willCut = false;
			}
			else if (negativeClamp && !positiveClamp)
			{
				willCut = true;
			}
			else if (!cutEdgesIndependently)
			{
				// uniform state
				double inverseThreshold = 1.0 - threshold;
				double scheduleRatio = 1.0 - 1.0*max(0.0, min(1.0, (1.0*(timeStep+timeBound/4)/timeBound)));
				double scheduledInverseThreshold = 1.0*max(0.0, min(1.0, scheduleRatio)) * inverseThreshold;
				double scheduledThreshold = 1 - scheduledInverseThreshold;
				willCut = edgeWeights[i] <= scheduledThreshold;

				//LOG() << "\toriginal threshold=" << threshold << ", scheduled threshold=" << scheduledThreshold << ", will cut=" << willCut << endl;
			}
			else
			{
				// bernoulli independent
				//double biasedCoin = Rand::unifDist(); // ~ Uniform(0, 1)
				//willCut = biasedCoin <= 1-edgeWeights[i];

				double indepThreshold = Rand::unifDist(); // ~ Uniform(0, 1)
				double inverseThreshold = 1.0 - indepThreshold;
				double scheduleRatio = 1.0 - 1.0*max(0.0, min(1.0, (1.0*(timeStep+timeBound/4)/timeBound)));
				double scheduledInverseThreshold = 1.0*max(0.0, min(1.0, scheduleRatio)) * inverseThreshold;
				double scheduledThreshold = 1 - scheduledInverseThreshold;
				willCut = edgeWeights[i] <= scheduledThreshold;

				//LOG() << "\toriginal threshold=" << indepThreshold << ", scheduled threshold=" << scheduledThreshold << ", will cut=" << willCut << endl;
			}

			if (!willCut)
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

	vector< ImgCandidate > StochasticScheduleSuccessor::createCandidates(ImgLabeling& YPred, MyGraphAlgorithms::SubgraphSet* subgraphs)
	{
		using namespace MyGraphAlgorithms;

		vector< Subgraph* > subgraphset = subgraphs->getSubgraphs();

		// node clamp decisions
		vector<bool> nodeClampDecisions;
		int numNodeClamp = 0;
		for (int node = 0; node < YPred.getNumNodes(); node++)
		{
			bool decision;
			if (!YPred.confidencesAvailable)
				decision = false;
			else if (!this->clampNodes)
				decision = false;
			else
			{
				// get most confident label's confidence
				int label = YPred.getMostConfidentLabel(node);
				double confidence = YPred.getConfidence(node, label);

				// make decision
				decision = (confidence >= nodeClampThreshold);
			}

			nodeClampDecisions.push_back(decision);

			if (decision)
				numNodeClamp++;
		}

		// successors set
		vector< ImgCandidate > successors;

		// loop over each sub graph
		int cumSumLabels = 0;
		int numSumLabels = 0;
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
				
				// get labels
				getLabels(candidateLabelsSet, cc);
				
				candidateLabelsSet.erase(nodeLabel);

				cumSumLabels += candidateLabelsSet.size();
				numSumLabels++;

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
					set<int> action;
					for (set<int>::iterator it4 = component.begin(); it4 != component.end(); ++it4)
					{
						int node = *it4;
						if (this->clampNodes && nodeClampDecisions[node])
							YNew.graph.nodesData(node) = YPred.getMostConfidentLabel(node);
						else
							YNew.graph.nodesData(node) = label;
						action.insert(node);
					}

					ImgCandidate YCandidate;
					YCandidate.labeling = YNew;
					YCandidate.action = action;

					successors.push_back(YCandidate);
				}
			}
		}

		if (numSumLabels > 0)
			LOG() << "average num labels=" << (1.0*cumSumLabels/numSumLabels) << endl;

		LOG() << "num nodes clamped=" << numNodeClamp << " out of " << YPred.getNumNodes() << endl;

		return successors;
	}

	void StochasticScheduleSuccessor::getLabels(set<int>& candidateLabelsSet, MyGraphAlgorithms::ConnectedComponent* cc)
	{
		getAllLabels(candidateLabelsSet, cc);	
	}

	void StochasticScheduleSuccessor::getAllLabels(set<int>& candidateLabelsSet, MyGraphAlgorithms::ConnectedComponent* cc)
	{
		// flip to any possible class
		candidateLabelsSet = Global::settings->CLASSES.getLabels();
	}

	void StochasticScheduleSuccessor::getNeighborLabels(set<int>& candidateLabelsSet, MyGraphAlgorithms::ConnectedComponent* cc)
	{
		if (cc->hasNeighbors())
		{
			// add only neighboring labels to candidate label set
			set<int> neighborSet = cc->getNeighborLabels();
			candidateLabelsSet.insert(neighborSet.begin(), neighborSet.end());
		}
		else
		{
			// if connected component is isolated without neighboring connected components, then flip to any possible class
			candidateLabelsSet = Global::settings->CLASSES.getLabels();
		}
	}

	void StochasticScheduleSuccessor::getConfidencesNeighborLabels(set<int>& candidateLabelsSet, MyGraphAlgorithms::ConnectedComponent* cc)
	{
		int topKConfidences = static_cast<int>(ceil(TOP_CONFIDENCES_PROPORTION * Global::settings->CLASSES.numClasses()));
		candidateLabelsSet = cc->getTopConfidentLabels(topKConfidences);
		if (cc->hasNeighbors())
		{
			// add only neighboring labels to candidate label set
			set<int> neighborSet = cc->getNeighborLabels();
			candidateLabelsSet.insert(neighborSet.begin(), neighborSet.end());
		}
	}

	double StochasticScheduleSuccessor::computeKL(const VectorXd& p, const VectorXd& q)
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

	StochasticScheduleNeighborSuccessor::StochasticScheduleNeighborSuccessor()
	{
		this->cutParam = DEFAULT_T_PARAM;
		this->cutEdgesIndependently = true;
		this->clampNodes = false;
		this->clampEdges = false;
		this->nodeClampThreshold = DEFAULT_NODE_CLAMP_THRESHOLD;
		this->edgeClampPositiveThreshold = DEFAULT_EDGE_CLAMP_POSITIVE_THRESHOLD;
		this->edgeClampNegativeThreshold = DEFAULT_EDGE_CLAMP_NEGATIVE_THRESHOLD;
	}

	StochasticScheduleNeighborSuccessor::StochasticScheduleNeighborSuccessor(bool cutEdgesIndependently, double cutParam)
	{
		this->cutParam = cutParam;
		this->cutEdgesIndependently = cutEdgesIndependently;
		this->clampNodes = false;
		this->clampEdges = false;
		this->nodeClampThreshold = DEFAULT_NODE_CLAMP_THRESHOLD;
		this->edgeClampPositiveThreshold = DEFAULT_EDGE_CLAMP_POSITIVE_THRESHOLD;
		this->edgeClampNegativeThreshold = DEFAULT_EDGE_CLAMP_NEGATIVE_THRESHOLD;
	}

	StochasticScheduleNeighborSuccessor::StochasticScheduleNeighborSuccessor(bool cutEdgesIndependently, double cutParam, 
	bool clampNodes, bool clampEdges, double nodeClampThreshold, double edgeClampPositiveThreshold, double edgeClampNegativeThreshold)
	{
		this->cutParam = cutParam;
		this->cutEdgesIndependently = cutEdgesIndependently;
		this->clampNodes = clampNodes;
		this->clampEdges = clampEdges;
		this->nodeClampThreshold = nodeClampThreshold;
		this->edgeClampPositiveThreshold = edgeClampPositiveThreshold;
		this->edgeClampNegativeThreshold = edgeClampNegativeThreshold;
	}

	StochasticScheduleNeighborSuccessor::~StochasticScheduleNeighborSuccessor()
	{
	}

	void StochasticScheduleNeighborSuccessor::getLabels(set<int>& candidateLabelsSet, MyGraphAlgorithms::ConnectedComponent* cc)
	{
		getNeighborLabels(candidateLabelsSet, cc);	
	}

	/**************** Stochastic Confidences Neighbor Successor Function ****************/

	StochasticScheduleConfidencesNeighborSuccessor::StochasticScheduleConfidencesNeighborSuccessor()
	{
		this->cutParam = DEFAULT_T_PARAM;
		this->cutEdgesIndependently = true;
		this->clampNodes = false;
		this->clampEdges = false;
		this->nodeClampThreshold = DEFAULT_NODE_CLAMP_THRESHOLD;
		this->edgeClampPositiveThreshold = DEFAULT_EDGE_CLAMP_POSITIVE_THRESHOLD;
		this->edgeClampNegativeThreshold = DEFAULT_EDGE_CLAMP_NEGATIVE_THRESHOLD;
	}

	StochasticScheduleConfidencesNeighborSuccessor::StochasticScheduleConfidencesNeighborSuccessor(bool cutEdgesIndependently, double cutParam)
	{
		this->cutParam = cutParam;
		this->cutEdgesIndependently = cutEdgesIndependently;
		this->clampNodes = false;
		this->clampEdges = false;
		this->nodeClampThreshold = DEFAULT_NODE_CLAMP_THRESHOLD;
		this->edgeClampPositiveThreshold = DEFAULT_EDGE_CLAMP_POSITIVE_THRESHOLD;
		this->edgeClampNegativeThreshold = DEFAULT_EDGE_CLAMP_NEGATIVE_THRESHOLD;
	}

	StochasticScheduleConfidencesNeighborSuccessor::StochasticScheduleConfidencesNeighborSuccessor(bool cutEdgesIndependently, double cutParam, 
	bool clampNodes, bool clampEdges, double nodeClampThreshold, double edgeClampPositiveThreshold, double edgeClampNegativeThreshold)
	{
		this->cutParam = cutParam;
		this->cutEdgesIndependently = cutEdgesIndependently;
		this->clampNodes = clampNodes;
		this->clampEdges = clampEdges;
		this->nodeClampThreshold = nodeClampThreshold;
		this->edgeClampPositiveThreshold = edgeClampPositiveThreshold;
		this->edgeClampNegativeThreshold = edgeClampNegativeThreshold;
	}

	StochasticScheduleConfidencesNeighborSuccessor::~StochasticScheduleConfidencesNeighborSuccessor()
	{
	}

	void StochasticScheduleConfidencesNeighborSuccessor::getLabels(set<int>& candidateLabelsSet, MyGraphAlgorithms::ConnectedComponent* cc)
	{
		getConfidencesNeighborLabels(candidateLabelsSet, cc);	
	}

	/**************** Stochastic Constrained Successor Function ****************/

	const double StochasticConstrainedSuccessor::TOP_CONFIDENCES_PROPORTION = 0.5;
	const double StochasticConstrainedSuccessor::DEFAULT_T_PARAM = 0.5;
	const double StochasticConstrainedSuccessor::DEFAULT_NODE_CLAMP_THRESHOLD = 0.9;
	const double StochasticConstrainedSuccessor::DEFAULT_EDGE_CLAMP_POSITIVE_THRESHOLD = 0.9;
	const double StochasticConstrainedSuccessor::DEFAULT_EDGE_CLAMP_NEGATIVE_THRESHOLD = 0.1;

	StochasticConstrainedSuccessor::StochasticConstrainedSuccessor()
	{
		this->cutParam = DEFAULT_T_PARAM;
		this->cutEdgesIndependently = true;
		this->clampNodes = false;
		this->clampEdges = false;
		this->nodeClampThreshold = DEFAULT_NODE_CLAMP_THRESHOLD;
		this->edgeClampPositiveThreshold = DEFAULT_EDGE_CLAMP_POSITIVE_THRESHOLD;
		this->edgeClampNegativeThreshold = DEFAULT_EDGE_CLAMP_NEGATIVE_THRESHOLD;
	}

	StochasticConstrainedSuccessor::StochasticConstrainedSuccessor(bool cutEdgesIndependently, double cutParam, 
	bool clampNodes, bool clampEdges, double nodeClampThreshold, double edgeClampPositiveThreshold, double edgeClampNegativeThreshold)
	{
		this->cutParam = cutParam;
		this->cutEdgesIndependently = cutEdgesIndependently;
		this->clampNodes = clampNodes;
		this->clampEdges = clampEdges;
		this->nodeClampThreshold = nodeClampThreshold;
		this->edgeClampPositiveThreshold = edgeClampPositiveThreshold;
		this->edgeClampNegativeThreshold = edgeClampNegativeThreshold;
	}

	StochasticConstrainedSuccessor::~StochasticConstrainedSuccessor()
	{
	}
	
	vector< ImgCandidate > StochasticConstrainedSuccessor::generateSuccessors(ImgFeatures& X, ImgLabeling& YPred, int timeStep, int timeBound)
	{
		using namespace MyPrimitives;

		clock_t tic = clock();

		LOG() << "generating stochastic constrained successors..." << endl;

		bool useConstraints = true;
		if (!YPred.confidencesAvailable)
		{
			LOG(WARNING) << "node confidences are not available for constrained successor. turning off clamp.";
			useConstraints = false;
		}
		else if (!X.edgeWeightsAvailable)
		{
			LOG(WARNING) << "edge weights are not available for constrained successor. turning off clamp.";
			useConstraints = false;
		}
		//TODO: use edge weights or just compute from KL

		// data strutures to keep track of node and edge clamping
		vector< bool > nodesClamped;
		map< Pair<int, int>, bool > edgesClamped;
		map< Pair<int, int>, bool > edgesCut;

		// keep track of new YPred with constrained labels
		ImgLabeling YPredConstrained = YPred;

		// assign node clamping
		const int numNodes = YPred.getNumNodes();
		int numClampedNodes = 0;
		for (int node = 0; node < numNodes; node++)
		{
			if (!useConstraints || !this->clampNodes)
			{
				nodesClamped.push_back(false);
			}
			else
			{
				int label = YPred.getLabel(node);
				double confidence = YPred.getConfidence(node, label);
				bool clamp = confidence >= this->nodeClampThreshold;
				nodesClamped.push_back(clamp);
				if (clamp)
					numClampedNodes++;
			}
		}

		// assign edge clamping
		const int numEdges = X.getNumEdges();
		int numPositiveClampedEdges = 0;
		int numNegativeClampedEdges = 0;
		for (map< Pair<int, int>, double >::iterator it = X.edgeWeights.begin(); it != X.edgeWeights.end(); ++it)
		{
			Pair<int, int> key = it->first;
			double edgeWeight = it->second;

			if (!useConstraints || !this->clampEdges)
			{
				edgesClamped[key] = false;
			}
			else
			{
				if (edgeWeight >= this->edgeClampPositiveThreshold)
				{
					edgesClamped[key] = true;
					edgesCut[key] = false;
					numPositiveClampedEdges++;
				}
				else if (edgeWeight <= this->edgeClampNegativeThreshold)
				{
					edgesClamped[key] = true;
					edgesCut[key] = true;
					numNegativeClampedEdges++;
				}
				else
				{
					edgesClamped[key] = false;
				}
			}
		}

		LOG() << "num clamped nodes=" << numClampedNodes << "/" << numNodes << endl;
		LOG() << "num positive clamped edges=" << numPositiveClampedEdges << "/" << numEdges << endl;
		LOG() << "num negative clamped edges=" << numNegativeClampedEdges << "/" << numEdges << endl;

		// constraint propagation 1: propagate information with must-link edges
		// 1) compute transitive closure on must-link edges
		AdjList_t closure = transitiveClosurePositiveEdges(edgesClamped, edgesCut, YPred.getNumNodes());

		// 2) if a node in CC is clamped, then propagate label
		for (AdjList_t::iterator it = closure.begin(); it != closure.end(); ++it)
		{
			// if node is clamped, propagate information to connected component
			int node1 = it->first;
			if (nodesClamped[node1])
			{
				NeighborSet_t neighbors = it->second;
				for (NeighborSet_t::iterator it2 = neighbors.begin(); it2 != neighbors.end(); ++it2)
				{
					int node2 = *it2;
					YPredConstrained.graph.nodesData(node2) = YPred.graph.nodesData(node1);
				}
			}
			//TODO: what if connected components has multiple clamped nodes with different labels?
		}

		// cut edges without clamping (stochastic cutting)
		for (map< Pair<int, int>, bool >:: iterator it = edgesClamped.begin(); it != edgesClamped.end(); ++it)
		{
			bool isClamped = it->second;
			if (!isClamped)
			{
				Pair<int, int> edge = it->first;
				double edgeWeight = X.edgeWeights[edge];

				// perform cutting based on stochastic threshold
				double indepThreshold = Rand::unifDist(); // ~ Uniform(0, 1)
				double inverseThreshold = 1.0 - indepThreshold;
				double scheduleRatio = 1.0 - 1.0*max(0.0, min(1.0, (1.0*(timeStep+timeBound/4)/timeBound)));
				double scheduledInverseThreshold = 1.0*max(0.0, min(1.0, scheduleRatio)) * inverseThreshold;
				double scheduledThreshold = 1 - scheduledInverseThreshold;
				edgesCut[edge] = edgeWeight <= scheduledThreshold;

				//TODO options: uniform vs. independent edge thresholding
			}
		}

		// compute subgraphs
		ImgLabeling Ycopy;
		Ycopy.confidences = YPredConstrained.confidences;
		Ycopy.confidencesAvailable = YPredConstrained.confidencesAvailable;
		Ycopy.graph = YPredConstrained.graph;
		LOG() << "Getting subgraphs..." << endl;
		MyGraphAlgorithms::SubgraphSet* subgraphs = new MyGraphAlgorithms::SubgraphSet(Ycopy, edgesCut);

		// constraint propagation 2: generate successors and propose labels that satisfy must-not-link edges
		vector< ImgCandidate > successors = createCandidates(YPredConstrained, subgraphs, nodesClamped, edgesClamped, edgesCut);

		LOG() << "num successors generated=" << successors.size() << endl;

		Global::settings->stats->addSuccessorCount(successors.size());

		delete subgraphs;

		clock_t toc = clock();
		LOG() << "successor total time: " << (double)(toc - tic)/CLOCKS_PER_SEC << endl;

		return successors;
	}

	vector< ImgCandidate > StochasticConstrainedSuccessor::createCandidates(ImgLabeling& YPred, MyGraphAlgorithms::SubgraphSet* subgraphs, 
		vector< bool > nodesClamped, map< MyPrimitives::Pair<int, int>, bool > edgesClamped, map< MyPrimitives::Pair<int, int>, bool > edgesCut)
	{
		using namespace MyGraphAlgorithms;

		vector< Subgraph* > subgraphset = subgraphs->getSubgraphs();

		// successors set
		vector< ImgCandidate > successors;

		// loop over each sub graph
		int cumSumLabels = 0;
		int numSumLabels = 0;
		int numEdgeConstraintEnforcement = 0;
		for (vector< Subgraph* >::iterator it = subgraphset.begin(); it != subgraphset.end(); ++it)
		{
			Subgraph* sub = *it;
			vector< ConnectedComponent* > ccset = sub->getConnectedComponents();

			// loop over each connected component
			for (vector< ConnectedComponent* >::iterator it2 = ccset.begin(); it2 != ccset.end(); ++it2)
			{
				ConnectedComponent* cc = *it2;

				// setup
				set<int> candidateLabelsSet;
				int nodeLabel = cc->getLabel();
				candidateLabelsSet.insert(nodeLabel);
				
				// get labels - top 4 confidences
				int topKConfidences = static_cast<int>(ceil(TOP_CONFIDENCES_PROPORTION * Global::settings->CLASSES.numClasses()));
				candidateLabelsSet = cc->getTopConfidentLabels(topKConfidences);
				if (cc->hasNeighbors())
				{
					// add only neighboring labels to candidate label set
					set<int> neighborSet = cc->getNeighborLabels();
					candidateLabelsSet.insert(neighborSet.begin(), neighborSet.end());
				}
				candidateLabelsSet.erase(nodeLabel);

				// remove from label set if there is a must-not link edge constraint with clamped neighbor node
				set<int> component = cc->getNodes();
				for (set<int>::iterator it4 = component.begin(); it4 != component.end(); ++it4)
				{
					int node1 = *it4;
					set<int> neighbors = YPred.graph.adjList[node1];
					for (set<int>::iterator it5 = neighbors.begin(); it5 != neighbors.end(); ++it5)
					{
						int node2 = *it5;
						MyPrimitives::Pair<int, int> edge = MyPrimitives::Pair<int, int>(node1, node2);
						if (edgesCut[edge] && edgesClamped[edge] && nodesClamped[node2])
						{
							int clampedLabel = YPred.getLabel(node2);
							if (candidateLabelsSet.count(clampedLabel) != 0)
							{
								candidateLabelsSet.erase(clampedLabel);
								numEdgeConstraintEnforcement++;
							}
						}
					}
				}

				// statistics purposes
				cumSumLabels += candidateLabelsSet.size();
				numSumLabels++;

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
					set<int> action;
					for (set<int>::iterator it4 = component.begin(); it4 != component.end(); ++it4)
					{
						int node = *it4;
						// clamp node otherwise flip it
						if (nodesClamped[node])
							YNew.graph.nodesData(node) = YPred.getLabel(node);
						else
							YNew.graph.nodesData(node) = label;
						action.insert(node);
					}

					ImgCandidate YCandidate;
					YCandidate.labeling = YNew;
					YCandidate.action = action;

					successors.push_back(YCandidate);
				}
			}
		}

		if (numSumLabels > 0)
			LOG() << "average num labels=" << (1.0*cumSumLabels/numSumLabels) << endl;

		LOG() << "num negative edge constraint enforcements=" << numEdgeConstraintEnforcement << endl;

		return successors;
	}

	AdjList_t StochasticConstrainedSuccessor::transitiveClosurePositiveEdges(map< MyPrimitives::Pair<int, int>, bool > edgesClamped, 
			map< MyPrimitives::Pair<int, int>, bool > edgesCut, int numNodes)
	{
		using namespace MyPrimitives;

		// initialize closure matrix
		MatrixXi closureMatrix = MatrixXi::Zero(numNodes, numNodes);
		for (map< Pair<int, int>, bool >:: iterator it = edgesClamped.begin(); it != edgesClamped.end(); ++it)
		{
			bool isClamped = it->second;
			if (isClamped)
			{
				Pair<int, int> edge = it->first;
				if (edgesCut.count(edge) != 0)
				{
					if (!edgesCut[edge])
					{
						// this edge is a positive edge (must-link) so add to closureMatrix
						closureMatrix(edge.first, edge.second) = 1;
					}
				}
				else
				{
					LOG(ERROR) << "edges cut should not be empty when edges are clamped!";
				}
			}
		}

		// compute transitive closure
		for (int k = 0; k < numNodes; k++)
		{
			for (int i = 0; i < numNodes; i++)
			{
				for (int j = 0; j < numNodes; j++)
				{
					// really naive way to do the following:
					// w_ij = w_ij || (w_ik && w_kj)

					int a = 0;
					if (closureMatrix(i, k) == 1 && closureMatrix(k, j) == 1)
						a = 1;

					closureMatrix(i, j) = 1;
					if (closureMatrix(i, j) == 0 && a == 0)
						closureMatrix(i, j) = 0;
				}
			}
		}

		// convert matrix to list form
		AdjList_t closure;
		for (int node1 = 0; node1 < numNodes; node1++)
		{
			for (int node2 = 0; node2 < numNodes; node2++)
			{
				if (closureMatrix(node1, node2) == 1)
				{
					// add
					if (closure.count(node1) == 0)
					{
						closure[node1] = NeighborSet_t();
					}
					closure[node1].insert(node2);
				}
			}
		}

		return closure;
	}
}