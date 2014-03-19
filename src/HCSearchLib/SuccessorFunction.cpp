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
		this->maxNumSuccessorCandidates = Global::settings->RANDOM_SUCCESSOR_PRUNE_MAX_CANDIDATES;
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

		LOG() << "num successors generated=" << successors.size() << endl;

		// prune to the bound
		if (Global::settings->RANDOM_SUCCESSOR_PRUNE)
		{
			const int originalSize = successors.size();
			if (originalSize > maxNumSuccessorCandidates)
			{
				random_shuffle(successors.begin(), successors.end());
				for (int i = 0; i < originalSize - maxNumSuccessorCandidates; i++)
					successors.pop_back();

				LOG() << "\tpruned to num successors=" << successors.size() << endl;
			}
		}

		Global::settings->stats->addSuccessorCount(successors.size());

		clock_t toc = clock();
		LOG() << "successor total time: " << (double)(toc - tic)/CLOCKS_PER_SEC << endl;

		return successors;
	}

	/**************** Flipbit Neighbor Successor Function ****************/

	FlipbitNeighborSuccessor::FlipbitNeighborSuccessor()
	{
		this->maxNumSuccessorCandidates = Global::settings->RANDOM_SUCCESSOR_PRUNE_MAX_CANDIDATES;
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

		LOG() << "num successors generated=" << successors.size() << endl;

		// prune to the bound
		if (Global::settings->RANDOM_SUCCESSOR_PRUNE)
		{
			const int originalSize = successors.size();
			if (originalSize > maxNumSuccessorCandidates)
			{
				random_shuffle(successors.begin(), successors.end());
				for (int i = 0; i < originalSize - maxNumSuccessorCandidates; i++)
					successors.pop_back();

				LOG() << "\tpruned to num successors=" << successors.size() << endl;
			}
		}

		Global::settings->stats->addSuccessorCount(successors.size());

		clock_t toc = clock();
		LOG() << "successor total time: " << (double)(toc - tic)/CLOCKS_PER_SEC << endl;

		return successors;
	}

	/**************** Flipbit Confidences Neighbor Successor Function ****************/

	FlipbitConfidencesNeighborSuccessor::FlipbitConfidencesNeighborSuccessor()
	{
		this->maxNumSuccessorCandidates = Global::settings->RANDOM_SUCCESSOR_PRUNE_MAX_CANDIDATES;
	}

	FlipbitConfidencesNeighborSuccessor::FlipbitConfidencesNeighborSuccessor(int maxNumSuccessorCandidates)
	{
		this->maxNumSuccessorCandidates = maxNumSuccessorCandidates;
	}

	FlipbitConfidencesNeighborSuccessor::~FlipbitConfidencesNeighborSuccessor()
	{
	}
	
	vector< ImgLabeling > FlipbitConfidencesNeighborSuccessor::generateSuccessors(ImgFeatures& X, ImgLabeling& YPred)
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

				// add candidate to successors
				successors.push_back(YNew);
			}
		}

		LOG() << "num successors generated=" << successors.size() << endl;

		// prune to the bound
		if (Global::settings->RANDOM_SUCCESSOR_PRUNE)
		{
			const int originalSize = successors.size();
			if (originalSize > maxNumSuccessorCandidates)
			{
				random_shuffle(successors.begin(), successors.end());
				for (int i = 0; i < originalSize - maxNumSuccessorCandidates; i++)
					successors.pop_back();

				LOG() << "\tpruned to num successors=" << successors.size() << endl;
			}
		}

		Global::settings->stats->addSuccessorCount(successors.size());

		clock_t toc = clock();
		LOG() << "successor total time: " << (double)(toc - tic)/CLOCKS_PER_SEC << endl;

		return successors;
	}

	/**************** Stochastic Successor Function ****************/

	const double StochasticSuccessor::TOP_CONFIDENCES_PROPORTION = 0.5;
	const double StochasticSuccessor::DEFAULT_T_PARM = 0.5;

	StochasticSuccessor::StochasticSuccessor()
	{
		this->maxNumSuccessorCandidates = Global::settings->RANDOM_SUCCESSOR_PRUNE_MAX_CANDIDATES;
		this->cutParam = DEFAULT_T_PARM;
		this->cutEdgesIndependently = true;
	}

	StochasticSuccessor::StochasticSuccessor(bool cutEdgesIndependently, double cutParam)
	{
		this->maxNumSuccessorCandidates = Global::settings->RANDOM_SUCCESSOR_PRUNE_MAX_CANDIDATES;
		this->cutParam = cutParam;
		this->cutEdgesIndependently = cutEdgesIndependently;
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

		LOG() << "num successors generated=" << successors.size() << endl;

		// prune to the bound
		if (Global::settings->RANDOM_SUCCESSOR_PRUNE)
		{
			const int originalSize = successors.size();
			if (originalSize > maxNumSuccessorCandidates)
			{
				random_shuffle(successors.begin(), successors.end());
				for (int i = 0; i < originalSize - maxNumSuccessorCandidates; i++)
					successors.pop_back();

				LOG() << "\tpruned to num successors=" << successors.size() << endl;
			}
		}

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
					for (set<int>::iterator it4 = component.begin(); it4 != component.end(); ++it4)
					{
						int node = *it4;
						YNew.graph.nodesData(node) = label;
					}

					successors.push_back(YNew);
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
		this->maxNumSuccessorCandidates = Global::settings->RANDOM_SUCCESSOR_PRUNE_MAX_CANDIDATES;
		this->cutParam = DEFAULT_T_PARM;
		this->cutEdgesIndependently = true;
	}
	
	StochasticNeighborSuccessor::StochasticNeighborSuccessor(bool cutEdgesIndependently, double cutParam)
	{
		this->maxNumSuccessorCandidates = Global::settings->RANDOM_SUCCESSOR_PRUNE_MAX_CANDIDATES;
		this->cutParam = cutParam;
		this->cutEdgesIndependently = cutEdgesIndependently;
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

	void StochasticNeighborSuccessor::getLabels(set<int>& candidateLabelsSet, MyGraphAlgorithms::ConnectedComponent* cc)
	{
		getNeighborLabels(candidateLabelsSet, cc);	
	}

	/**************** Stochastic Confidences Neighbor Successor Function ****************/

	StochasticConfidencesNeighborSuccessor::StochasticConfidencesNeighborSuccessor()
	{
		this->maxNumSuccessorCandidates = Global::settings->RANDOM_SUCCESSOR_PRUNE_MAX_CANDIDATES;
		this->cutParam = DEFAULT_T_PARM;
		this->cutEdgesIndependently = true;
	}

	StochasticConfidencesNeighborSuccessor::StochasticConfidencesNeighborSuccessor(bool cutEdgesIndependently, double cutParam)
	{
		this->maxNumSuccessorCandidates = Global::settings->RANDOM_SUCCESSOR_PRUNE_MAX_CANDIDATES;
		this->cutParam = cutParam;
		this->cutEdgesIndependently = cutEdgesIndependently;
	}

	StochasticConfidencesNeighborSuccessor::StochasticConfidencesNeighborSuccessor(bool cutEdgesIndependently, double cutParam, int maxNumSuccessorCandidates)
	{
		this->maxNumSuccessorCandidates = maxNumSuccessorCandidates;
		this->cutParam = cutParam;
		this->cutEdgesIndependently = cutEdgesIndependently;
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
		this->maxNumSuccessorCandidates = Global::settings->RANDOM_SUCCESSOR_PRUNE_MAX_CANDIDATES;
		this->cutParam = DEFAULT_T_PARM;
		this->cutEdgesIndependently = false;
	}

	CutScheduleSuccessor::CutScheduleSuccessor(double cutParam)
	{
		this->maxNumSuccessorCandidates = Global::settings->RANDOM_SUCCESSOR_PRUNE_MAX_CANDIDATES;
		this->cutParam = cutParam;
		this->cutEdgesIndependently = false;
	}

	CutScheduleSuccessor::CutScheduleSuccessor(double cutParam, int maxNumSuccessorCandidates)
	{
		this->maxNumSuccessorCandidates = maxNumSuccessorCandidates;
		this->cutParam = cutParam;
		this->cutEdgesIndependently = false;
	}

	CutScheduleSuccessor::~CutScheduleSuccessor()
	{
	}
	
	vector< ImgLabeling > CutScheduleSuccessor::generateSuccessors(ImgFeatures& X, ImgLabeling& YPred)
	{
		clock_t tic = clock();

		double threshold = 0.025;

		// perform cut
		MyGraphAlgorithms::SubgraphSet* subgraphs = cutEdges(X, YPred, threshold, this->cutParam);

		LOG() << "generating successors..." << endl;

		// generate candidates
		vector< ImgLabeling > successors = createCandidates(YPred, subgraphs);

		LOG() << "num successors generated=" << successors.size() << endl;

		// prune to the bound
		if (Global::settings->RANDOM_SUCCESSOR_PRUNE)
		{
			const int originalSize = successors.size();
			if (originalSize > maxNumSuccessorCandidates)
			{
				random_shuffle(successors.begin(), successors.end());
				for (int i = 0; i < originalSize - maxNumSuccessorCandidates; i++)
					successors.pop_back();

				LOG() << "\tpruned to num successors=" << successors.size() << endl;
			}
		}

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
		this->maxNumSuccessorCandidates = Global::settings->RANDOM_SUCCESSOR_PRUNE_MAX_CANDIDATES;
		this->cutParam = DEFAULT_T_PARM;
		this->cutEdgesIndependently = false;
	}

	CutScheduleNeighborSuccessor::CutScheduleNeighborSuccessor(double cutParam)
	{
		this->maxNumSuccessorCandidates = Global::settings->RANDOM_SUCCESSOR_PRUNE_MAX_CANDIDATES;
		this->cutParam = cutParam;
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
	
	void CutScheduleNeighborSuccessor::getLabels(set<int>& candidateLabelsSet, MyGraphAlgorithms::ConnectedComponent* cc)
	{
		getNeighborLabels(candidateLabelsSet, cc);	
	}

	/**************** Cut Schedule Confidences Neighbor Successor Function ****************/

	CutScheduleConfidencesNeighborSuccessor::CutScheduleConfidencesNeighborSuccessor()
	{
		this->maxNumSuccessorCandidates = Global::settings->RANDOM_SUCCESSOR_PRUNE_MAX_CANDIDATES;
		this->cutParam = DEFAULT_T_PARM;
		this->cutEdgesIndependently = false;
	}

	CutScheduleConfidencesNeighborSuccessor::CutScheduleConfidencesNeighborSuccessor(double cutParam)
	{
		this->maxNumSuccessorCandidates = Global::settings->RANDOM_SUCCESSOR_PRUNE_MAX_CANDIDATES;
		this->cutParam = cutParam;
		this->cutEdgesIndependently = false;
	}

	CutScheduleConfidencesNeighborSuccessor::CutScheduleConfidencesNeighborSuccessor(double cutParam, int maxNumSuccessorCandidates)
	{
		this->maxNumSuccessorCandidates = maxNumSuccessorCandidates;
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
}