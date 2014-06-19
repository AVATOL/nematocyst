#include "LossFunction.hpp"
#include "MyLogger.hpp"

namespace HCSearch
{
	/**************** Loss Functions ****************/

	HammingLoss::HammingLoss()
	{
	}

	HammingLoss::~HammingLoss()
	{
	}

	double HammingLoss::computeLoss(ImgLabeling& YPred, const ImgLabeling& YTruth)
	{
		Matrix<bool, Dynamic, 1> diff = YPred.graph.nodesData.array() != YTruth.graph.nodesData.array();
		double loss = 0.0;
		for (int i = 0; i < diff.size(); i++)
		{
			if (diff(i))
				loss++;
		}
		return loss/diff.size();
	}

	PixelHammingLoss::PixelHammingLoss()
	{
	}

	PixelHammingLoss::~PixelHammingLoss()
	{
	}

	double PixelHammingLoss::computeLoss(ImgLabeling& YPred, const ImgLabeling& YTruth)
	{
		if (!YTruth.nodeWeightsAvailable)
		{
			LOG(WARNING) << "node weights are not available for computing pixel hamming loss.";
		}

		Matrix<bool, Dynamic, 1> diff = YPred.graph.nodesData.array() != YTruth.graph.nodesData.array();
		double loss = 0.0;
		for (int i = 0; i < diff.size(); i++)
		{
			if (diff(i))
				if (YTruth.nodeWeightsAvailable)
					loss += YTruth.nodeWeights(i);
				else
					loss += 1.0/diff.size();
		}
		return loss;
	}
}