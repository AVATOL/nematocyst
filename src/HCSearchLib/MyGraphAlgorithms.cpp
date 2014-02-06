#include "MyGraphAlgorithms.hpp"

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

	/**************** Connected Component Set ****************/
	
	ConnectedComponentSet::ConnectedComponentSet()
	{
	}

	ConnectedComponentSet::ConnectedComponentSet(HCSearch::ImgLabeling& labeling)
	{
		this->connectedComponents = vector< ConnectedComponent* >();
		this->original = labeling;

		const int numNodes = this->original.getNumNodes();

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
			}
			ccs[index]->addNode(node);
		}

		// add to list of connected components

		for (map< int, ConnectedComponent* >::iterator it = ccs.begin(); it != ccs.end(); ++it)
		{
			ConnectedComponent* cc = it->second;
			this->connectedComponents.push_back(cc);
		}
	}

	ConnectedComponentSet::ConnectedComponentSet(Subgraph* subgraph)
	{
		this->connectedComponents = vector< ConnectedComponent* >();
		this->original = subgraph->getOriginalLabeling();

		const int numNodes = this->original.getNumNodes();

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
			}
			ccs[index]->addNode(node);
		}

		// add to list of connected components

		for (map< int, ConnectedComponent* >::iterator it = ccs.begin(); it != ccs.end(); ++it)
		{
			ConnectedComponent* cc = it->second;
			this->connectedComponents.push_back(cc);
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

	/**************** Subgraph Set ****************/

	SubgraphSet::SubgraphSet()
	{
	}

	SubgraphSet::SubgraphSet(HCSearch::ImgLabeling& labeling, map< int, set<int> > cuts)
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
		}
	}

	SubgraphSet::~SubgraphSet()
	{
		for (vector< Subgraph* >::iterator it = subgraphs.begin(); it != subgraphs.end(); ++it)
		{
			Subgraph* sub = *it;
			delete sub;
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
}