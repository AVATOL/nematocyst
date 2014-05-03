#include "MyGraphAlgorithms.hpp"
#include "Globals.hpp"

namespace MyGraphAlgorithms
{
	/**************** Disjoint Set ****************/

	DisjointSet::DisjointSet(int size)
	{
		this->data = -1*VectorXi::Ones(size);
	}

	DisjointSet::DisjointSet()
	{
	}

	DisjointSet::~DisjointSet()
	{
	}

	void DisjointSet::Union(int i, int j)
	{
		// weighted union
		int parent_i = FindSet(i);
		int parent_j = FindSet(j);

		// don't do anything if already unioned
		if (parent_i == parent_j)
			return;

		int w_i = -data(parent_i);
		int w_j = -data(parent_j);
		if (w_i < w_j)
		{
			data(parent_i) = parent_j;
			data(parent_j) = -(w_i + w_j);
		}
		else
		{
			data(parent_j) = parent_i;
			data(parent_i) = -(w_i + w_j);
		}
	}

	int DisjointSet::FindSet(int i)
	{
		int r = i;
		while (data(r) >= 0) // find root
		{
			r = data(r);
		}
		if (i != r) // compress path
		{
			int k = data(i);
			while (k != r)
			{
				data(i) = r;
				k = data(k);
			}
		}

		return r;
	}

	/**************** Connected Components ****************/

	ConnectedComponent::ConnectedComponent()
	{
	}

	ConnectedComponent::ConnectedComponent(ConnectedComponentSet* ccSet)
	{
		this->ccSet = ccSet;
	}

	ConnectedComponent::~ConnectedComponent()
	{
	}

	int ConnectedComponent::size()
	{
		return this->nodes.size();
	}

	set<int> ConnectedComponent::getNodes()
	{
		return this->nodes;
	}

	void ConnectedComponent::addNode(int node)
	{
		this->nodes.insert(node);
		this->label = this->ccSet->getOriginalLabeling().getLabel(node);
	}

	int ConnectedComponent::getLabel()
	{
		return this->label;
	}

	set<int> ConnectedComponent::getNeighborLabels()
	{
		set<int> labels;
		labels.insert(this->label);

		HCSearch::ImgLabeling original = this->ccSet->getOriginalLabeling();

		// get nodes in connected component
		for (set<int>::iterator it = nodes.begin(); it != nodes.end(); ++it)
		{
			int node1 = *it;

			// get neighbors
			set<int> neighborLabels = original.getNeighborLabels(node1);
			for (set<int>::iterator it2 = neighborLabels.begin(); 
				it2 != neighborLabels.end(); ++it2)
			{
				labels.insert(*it2);
			}
		}

		labels.erase(this->label);
		return labels;
	}

	set<int> ConnectedComponent::getTopConfidentLabels(int K)
	{
		// check if confidences are available
		HCSearch::ImgLabeling original = this->ccSet->getOriginalLabeling();
		const int numLabels = original.confidences.cols();
		if (!original.confidencesAvailable)
		{
			LOG(WARNING) << "confidences not available to get top K confident labels.";
			return set<int>();
		}

		// bad cases
		if (K > numLabels)
		{
			return HCSearch::Global::settings->CLASSES.getLabels();
		}
		else if (K == 0)
		{
			return set<int>();
		}
		else if (K < 0)
		{
			LOG(ERROR) << "K cannot be negative!";
			HCSearch::abort();
		}

		set<int> labels;
		labels.insert(this->label);

		// get nodes in connected component and recompute label distribution
		int numNodes = 0;
		VectorXd labelDist = VectorXd::Zero(numLabels);
		for (set<int>::iterator it = nodes.begin(); it != nodes.end(); ++it)
		{
			int node1 = *it;

			labelDist += original.confidences.row(node1);
			numNodes++;
		}
		if (numNodes == 0)
		{
			LOG(ERROR) << "connected component contains no nodes";
			HCSearch::abort();
		}
		labelDist /= numNodes;

		// get top K confident labels
		HCSearch::LabelConfidencePQ sortedByConfidence;
		for (int i = 0; i < numLabels; i++)
		{
			int label = HCSearch::Global::settings->CLASSES.getClassLabel(i);
			double confidence = labelDist(i);
			sortedByConfidence.push(MyPrimitives::Pair<int, double>(label, confidence));
		}
		for (int i = 0; i < K; i++)
		{
			MyPrimitives::Pair<int, double> p = sortedByConfidence.top();
			sortedByConfidence.pop();
			labels.insert(p.first);
		}

		labels.erase(this->label);
		return labels;

		//set<int> labels;
		//labels.insert(this->label);

		//// get nodes in connected component
		//for (set<int>::iterator it = nodes.begin(); it != nodes.end(); ++it)
		//{
		//	int node1 = *it;

		//	// get top K confident labels
		//	HCSearch::LabelConfidencePQ sortedByConfidence;
		//	for (int i = 0; i < numLabels; i++)
		//	{
		//		int label = HCSearch::Global::settings->CLASSES.getClassLabel(i);
		//		double confidence = original.confidences(node1, i);
		//		sortedByConfidence.push(MyPrimitives::Pair<int, double>(label, confidence));
		//	}
		//	for (int i = 0; i < K; i++)
		//	{
		//		MyPrimitives::Pair<int, double> p = sortedByConfidence.top();
		//		sortedByConfidence.pop();
		//		labels.insert(p.first);
		//	}
		//}

		//labels.erase(this->label);
		//return labels;
	}

	bool ConnectedComponent::hasNeighbors()
	{
		bool hasNeighbors = false;

		HCSearch::ImgLabeling original = this->ccSet->getOriginalLabeling();

		// get nodes in connected component
		for (set<int>::iterator it = nodes.begin(); it != nodes.end(); ++it)
		{
			int node1 = *it;
			
			// get neighbors
			set<int> neighbors = original.graph.adjList[node1];
			set<int> uniqueNeighbors;

			// find left overs to see if "outside" neighbors of connected component exist
			set_difference(neighbors.begin(), neighbors.end(), 
				nodes.begin(), nodes.end(), 
				inserter(uniqueNeighbors, uniqueNeighbors.end()));

			if (!uniqueNeighbors.empty())
			{
				hasNeighbors = true;
				break;
			}
		}

		return hasNeighbors;
	}

	/**************** Connected Component Set ****************/
	
	ConnectedComponentSet::ConnectedComponentSet()
	{
	}

	ConnectedComponentSet::ConnectedComponentSet(HCSearch::ImgLabeling& labeling)
	{
		this->connectedComponents = vector< ConnectedComponent* >();
		this->original = labeling;

		const int numNodes = this->original.getNumNodes();

		int numForeground = 0;
		ConnectedComponent* foregroundCC = NULL;

		// first pass: union-find
		DisjointSet ds = DisjointSet(numNodes);
		for (int node1 = 0; node1 < numNodes; node1++)
		{
			if (!this->original.hasNeighbors(node1))
				continue;

			// get neighbors (ending nodes) of starting node
			set<int> neighbors = this->original.getNeighbors(node1);
			for (set<int>::iterator it2 = neighbors.begin(); it2 != neighbors.end(); ++it2)
			{
				int node2 = *it2;

				// get labels
				int nodeLabel1 = this->original.getLabel(node1);
				int nodeLabel2 = this->original.getLabel(node2);

				if (nodeLabel1 == nodeLabel2)
				{
					ds.Union(node1, node2);
				}
			}
		}

		// second pass: write to output labeling, record components

		map< int, ConnectedComponent* > ccs; // node index -> cc

		for (int node = 0; node < numNodes; node++)
		{
			int index = ds.FindSet(node);

			// record connected components
			if (ccs.count(index) == 0)
			{
				ccs[index] = new ConnectedComponent(this);

				if (!HCSearch::Global::settings->CLASSES.classLabelIsBackground(this->original.getLabel(node)))
				{
					numForeground++;
					foregroundCC = ccs[index];
				}
			}
			ccs[index]->addNode(node);
		}

		// add to list of connected components

		for (map< int, ConnectedComponent* >::iterator it = ccs.begin(); it != ccs.end(); ++it)
		{
			ConnectedComponent* cc = it->second;
			this->connectedComponents.push_back(cc);
		}

		if (numForeground == 1)
		{
			this->exactlyOnePositiveCC = true;
			this->foreground = foregroundCC;
		}
		else
		{
			this->exactlyOnePositiveCC = false;
			this->foreground = NULL;
		}
	}

	ConnectedComponentSet::ConnectedComponentSet(Subgraph* subgraph)
	{
		this->connectedComponents = vector< ConnectedComponent* >();
		this->original = subgraph->getOriginalLabeling();

		const int numNodes = this->original.getNumNodes();

		int numForeground = 0;
		ConnectedComponent* foregroundCC = NULL;

		// first pass: union-find
		DisjointSet ds = DisjointSet(numNodes);
		set<int> subNodes = subgraph->getNodes();
		for (set<int>::iterator it = subNodes.begin(); it != subNodes.end(); ++it)
		{
			// only consider nodes in the subgraph
			int node1 = *it;

			if (!this->original.hasNeighbors(node1))
				continue;

			// get neighbors (ending nodes) of starting node
			set<int> neighbors = this->original.getNeighbors(node1);
			for (set<int>::iterator it2 = neighbors.begin(); it2 != neighbors.end(); ++it2)
			{
				int node2 = *it2;

				if (subNodes.count(node2) == 0)
					continue;

				// get labels
				int nodeLabel1 = this->original.getLabel(node1);
				int nodeLabel2 = this->original.getLabel(node2);

				if (nodeLabel1 == nodeLabel2)
				{
					ds.Union(node1, node2);
				}
			}
		}

		// second pass: write to output labeling, record components

		map< int, ConnectedComponent* > ccs; // node index -> cc

		for (set<int>::iterator it = subNodes.begin(); it != subNodes.end(); ++it)
		{
			// only consider nodes in the subgraph
			int node = *it;
			int index = ds.FindSet(node);

			// record connected components
			if (ccs.count(index) == 0)
			{
				ccs[index] = new ConnectedComponent(this);

				if (!HCSearch::Global::settings->CLASSES.classLabelIsBackground(this->original.getLabel(node)))
				{
					numForeground++;
					foregroundCC = ccs[index];
				}
			}
			ccs[index]->addNode(node);
		}

		// add to list of connected components

		for (map< int, ConnectedComponent* >::iterator it = ccs.begin(); it != ccs.end(); ++it)
		{
			ConnectedComponent* cc = it->second;
			this->connectedComponents.push_back(cc);
		}

		if (numForeground == 1)
		{
			this->exactlyOnePositiveCC = true;
			this->foreground = foregroundCC;
		}
		else
		{
			this->exactlyOnePositiveCC = false;
			this->foreground = NULL;
		}
	}

	ConnectedComponentSet::~ConnectedComponentSet()
	{
		for (vector< ConnectedComponent* >::iterator it = connectedComponents.begin(); it != connectedComponents.end(); ++it)
		{
			ConnectedComponent* cc = *it;
			delete cc;
		}
	}

	int ConnectedComponentSet::size()
	{
		return this->connectedComponents.size();
	}

	HCSearch::ImgLabeling ConnectedComponentSet::getOriginalLabeling()
	{
		return this->original;
	}

	vector< ConnectedComponent* > ConnectedComponentSet::getConnectedComponents()
	{
		return this->connectedComponents;
	}

	bool ConnectedComponentSet::hasExactlyOnePositiveCC()
	{
		return this->exactlyOnePositiveCC;
	}

	/**************** Subgraphs ****************/

	Subgraph::Subgraph()
	{
		this->subgraphSet = NULL;
	}

	Subgraph::Subgraph(SubgraphSet* subgraphSet)
	{
		this->subgraphSet = subgraphSet;
	}

	Subgraph::~Subgraph()
	{
		delete connectedComponents;
	}

	int Subgraph::size()
	{
		return this->nodes.size();
	}

	set<int> Subgraph::getNodes()
	{
		return this->nodes;
	}

	void Subgraph::addNode(int node)
	{
		this->nodes.insert(node);
	}

	void Subgraph::processConnectedComponents()
	{
		this->connectedComponents = new ConnectedComponentSet(this);
	}

	HCSearch::ImgLabeling Subgraph::getOriginalLabeling()
	{
		return this->subgraphSet->getOriginalLabeling();
	}

	vector< ConnectedComponent* > Subgraph::getConnectedComponents()
	{
		return this->connectedComponents->getConnectedComponents();
	}

	bool Subgraph::hasExactlyOnePositiveCC()
	{
		return this->connectedComponents->hasExactlyOnePositiveCC();
	}

	/**************** Subgraph Set ****************/

	SubgraphSet::SubgraphSet()
	{
	}

	SubgraphSet::SubgraphSet(HCSearch::ImgLabeling& labeling, map< int, set<int> > cuts)
	{
		constructorHelper(labeling, cuts);
	}

	SubgraphSet::SubgraphSet(HCSearch::ImgLabeling& labeling, map< MyPrimitives::Pair<int, int>, bool > cuts)
	{
		map< int, set<int> > cutsConverted;

		// convert cuts to format
		for (map< MyPrimitives::Pair<int, int>, bool >::iterator it = cuts.begin(); it != cuts.end(); ++it)
		{
			// if not cut
			if (!it->second)
			{
				MyPrimitives::Pair<int, int> edge = it->first;
				int node1 = edge.first;
				int node2 = edge.second;

				if (cutsConverted.count(node1) == 0)
				{
					cutsConverted[node1] = set<int>();
				}
				cutsConverted[node1].insert(node2);
			}
		}

		constructorHelper(labeling, cutsConverted);
	}

	SubgraphSet::~SubgraphSet()
	{
		for (vector< Subgraph* >::iterator it = subgraphs.begin(); it != subgraphs.end(); ++it)
		{
			Subgraph* sub = *it;
			delete sub;
		}
	}

	void SubgraphSet::constructorHelper(HCSearch::ImgLabeling& labeling, map< int, set<int> > cuts)
	{
		this->cuts = cuts;
		this->original = labeling;

		// determine subgraphs

		const int numNodes = labeling.getNumNodes();

		// first pass: union-find
		DisjointSet ds = DisjointSet(numNodes);
		for (int node1 = 0; node1 < numNodes; node1++)
		{
			if (cuts.count(node1) == 0)
				continue;

			// get neighbors (ending nodes) of starting node
			set<int> neighbors = cuts[node1];
			for (set<int>::iterator it = neighbors.begin(); it != neighbors.end(); ++it)
			{
				int node2 = *it;

				// automatically union-find neighbors
				ds.Union(node1, node2);
			}
		}

		// second pass: write to output labeling, record components

		map< int, Subgraph* > subgraphs; // node index -> subgraph

		for (int node = 0; node < numNodes; node++)
		{
			int index = ds.FindSet(node);

			// record subgraphs
			if (subgraphs.count(index) == 0)
			{
				subgraphs[index] = new Subgraph(this);
			}
			subgraphs[index]->addNode(node);
		}

		// add to list of subgraphs

		this->subgraphs = vector< Subgraph* >();
		for (map< int, Subgraph* >::iterator it = subgraphs.begin(); it != subgraphs.end(); ++it)
		{
			Subgraph* sub = it->second;
			sub->processConnectedComponents();
			this->subgraphs.push_back(sub);

			if (sub->hasExactlyOnePositiveCC())
			{
				this->exactlyOnePositiveCCSubgraphs.push_back(sub);
			}
		}
	}

	int SubgraphSet::size()
	{
		return this->subgraphs.size();
	}

	HCSearch::ImgLabeling SubgraphSet::getOriginalLabeling()
	{
		return this->original;
	}

	vector< Subgraph* > SubgraphSet::getSubgraphs()
	{
		return this->subgraphs;
	}

	map< int, set<int> > SubgraphSet::getCuts()
	{
		return this->cuts;
	}

	vector< Subgraph* > SubgraphSet::getExactlyOnePositiveCCSubgraphs()
	{
		return this->exactlyOnePositiveCCSubgraphs;
	}
}