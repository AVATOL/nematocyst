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
}