#include "stdafx.h"
#include "CppUnitTest.h"

#include <iostream>
#include "HCSearch.hpp"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace HCSearch;

namespace Testing
{		
	TEST_CLASS(SearchSpaceTests)
	{
	public:
		
		TEST_METHOD(HammingLossTestZero)
		{
			LabelGraph graph1;
			graph1.nodesData = VectorXi::Zero(4);
			graph1.nodesData <<	1, 2, 3, 4;

			LabelGraph graph2;
			graph2.nodesData = VectorXi::Zero(4);
			graph2.nodesData <<	1, 2, 3, 4;

			ImgLabeling YPred;
			YPred.graph = graph1;
			ImgLabeling YTruth;
			YTruth.graph = graph2;

			ILossFunction* hammingLoss = new HammingLoss();
			double loss = hammingLoss->computeLoss(YPred, YTruth);

			Assert::AreEqual(loss, 0.0);
		}

		TEST_METHOD(HammingLossTestOne)
		{
			LabelGraph graph1;
			graph1.nodesData = VectorXi::Zero(4);
			graph1.nodesData <<	1, 2, 3, 4;

			LabelGraph graph2;
			graph2.nodesData = VectorXi::Zero(4);
			graph2.nodesData <<	5, 6, 7, 8;

			ImgLabeling YPred;
			YPred.graph = graph1;
			ImgLabeling YTruth;
			YTruth.graph = graph2;

			ILossFunction* hammingLoss = new HammingLoss();
			double loss = hammingLoss->computeLoss(YPred, YTruth);

			Assert::AreEqual(loss, 1.0);
		}

		TEST_METHOD(HammingLossTest)
		{
			LabelGraph graph1;
			graph1.nodesData = VectorXi::Zero(4);
			graph1.nodesData <<	1, 2, 3, 4;

			LabelGraph graph2;
			graph2.nodesData = VectorXi::Zero(4);
			graph2.nodesData <<	0, 2, -1, 4;

			ImgLabeling YPred;
			YPred.graph = graph1;
			ImgLabeling YTruth;
			YTruth.graph = graph2;

			ILossFunction* hammingLoss = new HammingLoss();
			double loss = hammingLoss->computeLoss(YPred, YTruth);

			Assert::AreEqual(loss, 0.5);
		}

		TEST_METHOD(FeatureFunctionTest)
		{
			Global::settings = new Settings();
			Global::settings->CLASSES.addClass(0, 1, false);
			Global::settings->CLASSES.addClass(1, 0, false);
			Global::settings->CLASSES.addClass(2, -1, true);
			Global::settings->CLASSES.setBackgroundLabel(-1);

			double EPSILON = 0.0001;

			AdjList_t edgeNeighbors = AdjList_t();
			NeighborSet_t v0Neighbors = NeighborSet_t();
			v0Neighbors.insert(3);
			v0Neighbors.insert(1);
			NeighborSet_t v1Neighbors = NeighborSet_t();
			v1Neighbors.insert(0);
			v1Neighbors.insert(2);
			NeighborSet_t v2Neighbors = NeighborSet_t();
			v2Neighbors.insert(1);
			v2Neighbors.insert(3);
			NeighborSet_t v3Neighbors = NeighborSet_t();
			v3Neighbors.insert(2);
			v3Neighbors.insert(0);

			edgeNeighbors[0] = v0Neighbors;
			edgeNeighbors[1] = v1Neighbors;
			edgeNeighbors[2] = v2Neighbors;
			edgeNeighbors[3] = v3Neighbors;

			LabelGraph graph1;
			graph1.nodesData = VectorXi::Zero(4);
			graph1.nodesData << 0, 1, 1, 0;
			graph1.adjList = edgeNeighbors;

			ImgLabeling YPred;
			YPred.graph = graph1;

			FeatureGraph graph2;
			graph2.nodesData = MatrixXd(4, 3);
			graph2.nodesData << 0.1, 0.2, 0.3,
					0.2, 0.4, 0.6,
					1.5, 3.0, 4.5,
					3.1, 4.5, 9.2;
			graph2.adjList = edgeNeighbors;

			ImgFeatures X;
			X.graph = graph2;

			IFeatureFunction* featureFunctionTest = new StandardAltFeatures();
			set<int> action;
			RankFeatures phi = featureFunctionTest->computeFeatures(X, YPred, action);

			const int featureDim = X.getFeatureDim();
			const int numClasses = Global::settings->CLASSES.numClasses();
			const int numBackgroundClasses = Global::settings->CLASSES.numBackgroundClasses();
			const int numForegroundClasses = numClasses - numBackgroundClasses;
	
			// check feature function dimension is correct
			const int unaryFeatDim = 1+featureDim;
			const int pairwiseFeatDim = featureDim;

			// check feature size
			Assert::AreEqual(phi.data.size(), numClasses*unaryFeatDim + (numClasses+1)*pairwiseFeatDim);

			// check unary potential
			VectorXd unaryTruth = VectorXd(numClasses*unaryFeatDim);
			unaryTruth <<	2, 1.5+0.2, 3.0+0.4, 4.5+0.6, 
							2, 3.1+0.1, 4.5+0.2, 9.2+0.3, 
							0, 0, 0, 0;

			VectorXd unaryDiff = phi.data.segment(0, numClasses*unaryFeatDim) - unaryTruth;
			bool unaryOkay = unaryDiff.cwiseAbs().sum() < EPSILON;
			Assert::AreEqual(unaryOkay, true);

			// check pairwise potential
			VectorXd pairwiseTruth = VectorXd((numClasses+1)*pairwiseFeatDim);
			pairwiseTruth <<	1-exp(-pow(abs(0.2-1.5), 2)), 1-exp(-pow(abs(0.4-3.0), 2)), 1-exp(-pow(abs(0.6-4.5), 2)),
								1-exp(-pow(abs(0.1-3.1), 2)), 1-exp(-pow(abs(0.2-4.5), 2)), 1-exp(-pow(abs(0.3-9.2), 2)),
								0, 0, 0,
								exp(-pow(abs(0.1-0.2), 2))+exp(-pow(abs(3.1-1.5), 2)), 
								exp(-pow(abs(0.2-0.4), 2))+exp(-pow(abs(4.5-3.0), 2)), 
								exp(-pow(abs(0.3-0.6), 2))+exp(-pow(abs(9.2-4.5), 2));

			VectorXd pairwiseDiff = phi.data.segment(numClasses*unaryFeatDim, (numClasses+1)*pairwiseFeatDim) - pairwiseTruth;
			bool pairwiseOkay = pairwiseDiff.cwiseAbs().sum() < EPSILON;
			Assert::AreEqual(pairwiseOkay, true);
		}
	};
}