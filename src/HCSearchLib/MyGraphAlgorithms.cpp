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
}