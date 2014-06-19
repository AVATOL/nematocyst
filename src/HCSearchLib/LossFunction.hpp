#ifndef LOSSFUNCTION_HPP
#define LOSSFUNCTION_HPP

#include "DataStructures.hpp"

namespace HCSearch
{
	/**************** Search Space Function Abstract Definitions ****************/

	/*!
	 * @brief Loss function interface
	 */
	class ILossFunction
	{
	public:
		virtual ~ILossFunction() {}

		/*!
		 * @brief Compute loss.
		 */
		virtual double computeLoss(ImgLabeling& YPred, const ImgLabeling& YTruth)=0;
	};

	/**************** Loss Functions ****************/

	/*!
	 * @brief Hamming loss function.
	 */
	class HammingLoss : public ILossFunction
	{
	public:
		HammingLoss();
		~HammingLoss();

		virtual double computeLoss(ImgLabeling& YPred, const ImgLabeling& YTruth);
	};

	/*!
	 * @brief Pixel Hamming loss function.
	 */
	class PixelHammingLoss : public ILossFunction
	{
	public:
		PixelHammingLoss();
		~PixelHammingLoss();

		virtual double computeLoss(ImgLabeling& YPred, const ImgLabeling& YTruth);
	};
}

#endif