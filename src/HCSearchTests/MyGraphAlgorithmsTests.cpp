#include "stdafx.h"
#include "CppUnitTest.h"

#include <iostream>
#include "HCSearch.hpp"
#include "MyGraphAlgorithms.hpp"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace HCSearch;

namespace Testing
{		
	TEST_CLASS(MyGraphAlgorithmsTests)
	{
	private:
		ImgLabeling Y;
		map< int, set<int> > cuts;

		static ImgLabeling setupImgLabelingHelper()
		{
			LabelGraph graph;
			graph.nodesData = VectorXi::Zero(9);
			graph.nodesData << 1, 1, 0, 1, 0, 1, 0, 1, 1;
			graph.adjList[0].insert(1);
			graph.adjList[0].insert(3);
			graph.adjList[1].insert(0);
			graph.adjList[1].insert(2);
			graph.adjList[1].insert(4);
			graph.adjList[2].insert(1);
			graph.adjList[2].insert(5);
			graph.adjList[3].insert(0);
			graph.adjList[3].insert(4);
			graph.adjList[3].insert(6);
			graph.adjList[4].insert(1);
			graph.adjList[4].insert(3);
			graph.adjList[4].insert(7);
			graph.adjList[4].insert(5);
			graph.adjList[5].insert(2);
			graph.adjList[5].insert(4);
			graph.adjList[5].insert(8);
			graph.adjList[6].insert(3);
			graph.adjList[6].insert(7);
			graph.adjList[7].insert(4);
			graph.adjList[7].insert(6);
			graph.adjList[7].insert(8);
			graph.adjList[8].insert(5);
			graph.adjList[8].insert(7);
			
			ImgLabeling Y;
			Y.graph = graph;
			return Y;
		}

		static map< int, set<int> > setupCutsHelper()
		{
			map< int, set<int> > cuts;
			cuts[0].insert(3);
			cuts[1].insert(2);
			cuts[1].insert(4);
			cuts[2].insert(1);
			cuts[2].insert(5);
			cuts[3].insert(0);
			cuts[3].insert(6);
			cuts[4].insert(1);
			cuts[4].insert(5);
			cuts[5].insert(2);
			cuts[5].insert(4);
			cuts[5].insert(8);
			cuts[6].insert(3);
			cuts[6].insert(7);
			cuts[7].insert(6);
			cuts[8].insert(5);
			return cuts;
		}

	public:
		
		TEST_METHOD_INITIALIZE(SubgraphSetInit)
		{
			Settings* settings = new Settings();
			settings->refresh("input", "output", "");
			Global::settings = settings;

			this->Y = setupImgLabelingHelper();	
			this->cuts = setupCutsHelper();
		}

		TEST_METHOD_CLEANUP(SubgraphSetTeardown)
		{
			delete Global::settings;
		}

		TEST_METHOD(SubgraphSetConstructionTest)
		{
			// check num of subgraphs
			MyGraphAlgorithms::SubgraphSet subgraphset(Y, cuts);
			Assert::AreEqual(subgraphset.size(), 2);

			// check subgraph sizes and cc sizes
			vector< MyGraphAlgorithms::Subgraph* > subs = subgraphset.getSubgraphs();
			for (vector< MyGraphAlgorithms::Subgraph* >::iterator it = subs.begin();
				it != subs.end(); ++it)
			{
				MyGraphAlgorithms::Subgraph* sub = *it;
				Assert::IsTrue(sub->size() == 4 || sub->size() == 5);
				Assert::IsTrue(sub->getConnectedComponents().size() == 3 || sub->getConnectedComponents().size() == 4);
			}
			// check num of ccs
			MyGraphAlgorithms::ConnectedComponentSet ccset(Y);
			Assert::AreEqual(ccset.size(), 5);

			// check cc sizes
			vector< MyGraphAlgorithms::ConnectedComponent* > ccs = ccset.getConnectedComponents();
			for (vector< MyGraphAlgorithms::ConnectedComponent* >::iterator it = ccs.begin();
				it != ccs.end(); ++it)
			{
				MyGraphAlgorithms::ConnectedComponent* cc = *it;
				if (cc->getLabel() == 0)
					Assert::AreEqual(cc->size(), 1);
				else if (cc->getLabel() == 1)
					Assert::AreEqual(cc->size(), 3);
			}
		}

		TEST_METHOD(DijsointSetUnionTest)
		{
			MyGraphAlgorithms::DisjointSet ds(10);
			ds.Union(1,2);
			Assert::AreEqual(ds.FindSet(1), ds.FindSet(2));
			Assert::AreNotEqual(ds.FindSet(1), ds.FindSet(3));

			ds.Union(1,2);
			Assert::AreEqual(ds.FindSet(1), ds.FindSet(2));
			Assert::AreNotEqual(ds.FindSet(1), ds.FindSet(3));

			ds.Union(2,3);
			Assert::AreEqual(ds.FindSet(1), ds.FindSet(2));
			Assert::AreEqual(ds.FindSet(2), ds.FindSet(3));
			Assert::AreEqual(ds.FindSet(1), ds.FindSet(3));
		}
	};
}