#ifndef INITIALSTATEFUNCTION_HPP
#define INITIALSTATEFUNCTION_HPP

#include "DataStructures.hpp"

namespace HCSearch
{
	/**************** Search Space Function Abstract Definitions ****************/

	/*!
	 * @brief Initial prediction function interface
	 */
	class IInitialPredictionFunction
	{
	public:
		virtual ~IInitialPredictionFunction() {}

		/*!
		 * @brief Get initial prediction.
		 */
		virtual ImgLabeling getInitialPrediction(ImgFeatures& X)=0;
	};

	/**************** Initial Prediction Functions ****************/

	/*!
	 * @brief IID logistic regression initial prediction function.
	 */
	class LogRegInit : public IInitialPredictionFunction
	{
	protected:
		static const double DEFAULT_C;
		static const double BINARY_CONFIDENCE_THRESHOLD;

	public:
		LogRegInit();
		~LogRegInit();

		virtual ImgLabeling getInitialPrediction(ImgFeatures& X);

		/*!
		 * @brief Train logistic regression model.
		 */
		void trainClassifier();

	protected:
		static void imgfeatures2liblinear(ImgFeatures& X, string filename);
		static void liblinear2imglabeling(ImgLabeling& Y, string filename);
		bool hasForegroundNeighbors(ImgLabeling& Y, int node);

		/*!
		 * @brief Eliminate 1-islands.
		 */
		void eliminateIslands(ImgLabeling& Y);
	};

	class MutexLogRegInit : public LogRegInit
	{
		static const int MUTEX_THRESHOLD;

	protected:
		map<string, int> mutex;
		bool initialized;

	public:
		MutexLogRegInit();
		~MutexLogRegInit();

		virtual ImgLabeling getInitialPrediction(ImgFeatures& X);

		void setMutex(map<string, int>& mutex);
		map<string, int> getMutex();

	private:
		string mutexStringHelper(int class1, int class2, string config);
	};
}

#endif