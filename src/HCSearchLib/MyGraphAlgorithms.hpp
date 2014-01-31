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
	 * @brief DisjointSet class for union-find.
	 */
	class DisjointSet
	{
		VectorXi data;

	public:
		DisjointSet(int size);
		DisjointSet();
		~DisjointSet();

		/*!
		 * @brief Union joins two elements.
		 */
		void Union(int i, int j);

		/*!
		 * @brief FindSet finds the parent.
		 */
		int FindSet(int i);
	};

	/**************** Connected Components ****************/

	/*!
	 * @brief Connected component contains nodes of the same label.
	 */
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

		/*!
		 * @brief Get the number of nodes in the connected component.
		 */
		int size();

		/*!
		 * @brief Get the nodes in the connected component.
		 */
		set<int> getNodes();

		/*!
		 * @brief Add a node into the connected component.
		 */
		void addNode(int node);

		/*!
		 * @brief Get the label of the connected component.
		 */
		int getLabel();

		/*!
		 * @brief Get the labels of the connected component's neighbors.
		 */
		set<int> getNeighborLabels();
	};

	/**************** Connected Component Set ****************/

	/*!
	 * @brief Connected component set contains a set of connected components.
	 * 
	 * Can construct from a labeling or a subgraph.
	 */
	class ConnectedComponentSet
	{
	private:
		vector< ConnectedComponent* > connectedComponents;
		HCSearch::ImgLabeling original;

	public:
		ConnectedComponentSet();

		/*!
		 * @brief Construct a connected component set from a labeling.
		 */
		ConnectedComponentSet(HCSearch::ImgLabeling& labeling);
		
		/*!
		 * @brief Construct a connected component set from a subgraph.
		 */
		ConnectedComponentSet(Subgraph* subgraph);
		
		~ConnectedComponentSet();

		/*!
		 * @brief Get the number of connected components.
		 */
		int size();

		/*!
		 * @brief Get the original labeling.
		 */
		HCSearch::ImgLabeling getOriginalLabeling();

		/*!
		 * @brief Get connected components.
		 */
		vector< ConnectedComponent* > getConnectedComponents();
	};

	/**************** Subgraphs ****************/

	/*!
	 * @brief Subgraph contains some nodes with connected components.
	 */
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

		/*!
		 * @brief Get the number of nodes in the subgraph.
		 */
		int size();

		/*!
		 * @brief Get the nodes in the subgraph.
		 */
		set<int> getNodes();

		/*!
		 * @brief Add a node into the subgraph.
		 */
		void addNode(int node);

		/*!
		 * @brief Find the connected components in the subgraph.
		 *
		 * Should only be called when no more nodes are added!
		 */
		void processConnectedComponents();

		/*!
		 * @brief Get the original labeling.
		 */
		HCSearch::ImgLabeling getOriginalLabeling();

		/*!
		 * @brief Get the connected components.
		 */
		vector< ConnectedComponent* > getConnectedComponents();
	};

	/**************** Subgraph Set ****************/

	/*!
	 * @brief Subgraph set contains a set of subgraphs, which are partitions of a labeling.
	 */
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

		/*!
		 * @brief Get the number of subgraphs.
		 */
		int size();

		/*!
		 * @brief Get the original labeling.
		 */
		HCSearch::ImgLabeling getOriginalLabeling();

		/*!
		 * @brief Get the subgraphs.
		 */
		vector< Subgraph* > getSubgraphs();

		/*!
		 * @brief Get the stochastic cuts.
		 */
		map< int, set<int> > getCuts();
	};
}

#endif