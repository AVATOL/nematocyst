#ifndef MYGRAPHALGORITHMS_HPP
#define MYGRAPHALGORITHMS_HPP

#include <vector>
#include <map>
#include <set>
#include "../../external/Eigen/Eigen/Dense"
#include "DataStructures.hpp"

using namespace Eigen;
using namespace std;

namespace MyGraphAlgorithms
{
	class ConnectedComponent;
	class ConnectedComponentSet;
	class Subgraph;
	class SubgraphSet;

	/**************** Disjoint Set ****************/

	/*!
	 * DisjointSet class for union-find.
	 */
	class DisjointSet
	{
		VectorXi data;

	public:
		DisjointSet(int size);
		DisjointSet();
		~DisjointSet();

		void Union(int i, int j);
		int FindSet(int i);
	};

	/**************** Connected Components ****************/

	class ConnectedComponent
	{
	private:
		set<int> nodes;
		int label;
		ConnectedComponentSet* ccSet;

	public:
		ConnectedComponent();
		ConnectedComponent(ConnectedComponentSet* ccSet);
		~ConnectedComponent();

		int size();
		set<int> getNodes();
		void addNode(int node);
		int getLabel();
		set<int> getNeighborLabels();
	};

	/**************** Connected Component Set ****************/

	class ConnectedComponentSet
	{
	private:
		vector< ConnectedComponent* > connectedComponents;
		HCSearch::ImgLabeling original;

	public:
		ConnectedComponentSet();
		ConnectedComponentSet(HCSearch::ImgLabeling& labeling);
		ConnectedComponentSet(Subgraph* subgraph);
		~ConnectedComponentSet();

		int size();
		HCSearch::ImgLabeling getOriginalLabeling();
		vector< ConnectedComponent* > getConnectedComponents();
	};

	/**************** Subgraphs ****************/

	class Subgraph
	{
	private:
		set<int> nodes;
		ConnectedComponentSet* connectedComponents;
		SubgraphSet* subgraphSet;

	public:
		Subgraph();
		Subgraph(SubgraphSet* subgraphSet);
		~Subgraph();

		int size();
		set<int> getNodes();
		void addNode(int node);
		void processConnectedComponents();
		HCSearch::ImgLabeling getOriginalLabeling();
		vector< ConnectedComponent* > getConnectedComponents();
	};

	/**************** Subgraph Set ****************/

	class SubgraphSet
	{
	private:
		vector< Subgraph* > subgraphs;
		map< int, set<int> > cuts;
		HCSearch::ImgLabeling original;

	public:
		SubgraphSet();
		SubgraphSet(HCSearch::ImgLabeling& labeling, map< int, set<int> > cuts);
		~SubgraphSet();

		int size();
		HCSearch::ImgLabeling getOriginalLabeling();
		vector< Subgraph* > getSubgraphs();
	};
}

#endif