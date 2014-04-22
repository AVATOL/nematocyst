#ifndef FEATUREFUNCTION_HPP
#define FEATUREFUNCTION_HPP

#include "../../external/Eigen/Eigen/Dense"
#include "DataStructures.hpp"

namespace HCSearch
{
	/**************** Search Space Function Abstract Definitions ****************/

	/*!
	 * @brief Feature function interface.
	 */
	class IFeatureFunction
	{
	public:
		virtual ~IFeatureFunction() {}

		/*!
		 * @brief Compute features.
		 */
		virtual RankFeatures computeFeatures(ImgFeatures& X, ImgLabeling& Y, set<int> action)=0;

		/*!
		 * @brief Get dimension of computed feature vector given structured features and labeling.
		 */
		virtual int featureSize(ImgFeatures& X, ImgLabeling& Y, set<int> action);
	};

	/**************** Feature Functions ****************/

	/*!
	 * @brief Standard CRF features with raw unary and raw pairwise potentials.
	 */
	class StandardFeatures : public IFeatureFunction
	{
	public:
		StandardFeatures();
		~StandardFeatures();

		virtual RankFeatures computeFeatures(ImgFeatures& X, ImgLabeling& Y, set<int> action);
		virtual int featureSize(ImgFeatures& X, ImgLabeling& Y, set<int> action);

	protected:
		/*!
		 * @brief Compute unary term.
		 */
		virtual VectorXd computeUnaryTerm(ImgFeatures& X, ImgLabeling& Y);
		
		/*!
		 * @brief Compute pairwise term.
		 */
		virtual VectorXd computePairwiseTerm(ImgFeatures& X, ImgLabeling& Y);

		/*!
		 * @brief Compute pairwise features.
		 */
		virtual VectorXd computePairwiseFeatures(VectorXd& nodeFeatures1, VectorXd& nodeFeatures2, 
			int nodeLabel1, int nodeLabel2, int& classIndex);
	};

	/*!
	 * @brief Standard CRF features with raw unary and raw pairwise potentials, 
	 * but pairwise does not distinguish non-class matches.
	 */
	class StandardAltFeatures : public IFeatureFunction
	{
	public:
		StandardAltFeatures();
		~StandardAltFeatures();

		virtual RankFeatures computeFeatures(ImgFeatures& X, ImgLabeling& Y, set<int> action);
		virtual int featureSize(ImgFeatures& X, ImgLabeling& Y, set<int> action);

	protected:
		/*!
		 * @brief Compute unary term.
		 */
		virtual VectorXd computeUnaryTerm(ImgFeatures& X, ImgLabeling& Y);
		
		/*!
		 * @brief Compute pairwise term.
		 */
		virtual VectorXd computePairwiseTerm(ImgFeatures& X, ImgLabeling& Y);

		/*!
		 * @brief Compute pairwise features.
		 */
		virtual VectorXd computePairwiseFeatures(VectorXd& nodeFeatures1, VectorXd& nodeFeatures2, 
			int nodeLabel1, int nodeLabel2, int& classIndex);
	};

	/*!
	 * @brief Standard CRF features with confidence unary and raw pairwise potentials.
	 */
	class StandardConfFeatures : public IFeatureFunction
	{
	public:
		StandardConfFeatures();
		~StandardConfFeatures();

		virtual RankFeatures computeFeatures(ImgFeatures& X, ImgLabeling& Y, set<int> action);
		virtual int featureSize(ImgFeatures& X, ImgLabeling& Y, set<int> action);

	protected:
		/*!
		 * @brief Compute unary term.
		 */
		virtual VectorXd computeUnaryTerm(ImgFeatures& X, ImgLabeling& Y);
		
		/*!
		 * @brief Compute pairwise term.
		 */
		virtual VectorXd computePairwiseTerm(ImgFeatures& X, ImgLabeling& Y);

		/*!
		 * @brief Compute pairwise features.
		 */
		virtual VectorXd computePairwiseFeatures(VectorXd& nodeFeatures1, VectorXd& nodeFeatures2, 
			int nodeLabel1, int nodeLabel2, int& classIndex);
	};

	/*!
	 * @brief Unary only raw features.
	 */
	class UnaryFeatures : public StandardFeatures
	{
	public:
		UnaryFeatures();
		~UnaryFeatures();

		virtual RankFeatures computeFeatures(ImgFeatures& X, ImgLabeling& Y, set<int> action);
		virtual int featureSize(ImgFeatures& X, ImgLabeling& Y, set<int> action);
	};

	/*!
	 * @brief Unary only confidences features.
	 */
	class UnaryConfFeatures : public StandardConfFeatures
	{
	public:
		UnaryConfFeatures();
		~UnaryConfFeatures();

		virtual RankFeatures computeFeatures(ImgFeatures& X, ImgLabeling& Y, set<int> action);
		virtual int featureSize(ImgFeatures& X, ImgLabeling& Y, set<int> action);
	};

	/*!
	 * @brief Standard CRF features with raw unary and co-occurence counts pairwise potentials.
	 */
	class StandardPairwiseCountsFeatures : public StandardFeatures
	{
	public:
		StandardPairwiseCountsFeatures();
		~StandardPairwiseCountsFeatures();

		virtual RankFeatures computeFeatures(ImgFeatures& X, ImgLabeling& Y, set<int> action);
		virtual int featureSize(ImgFeatures& X, ImgLabeling& Y, set<int> action);

	protected:
		virtual VectorXd computePairwiseTerm(ImgFeatures& X, ImgLabeling& Y);
		virtual VectorXd computePairwiseFeatures(VectorXd& nodeFeatures1, VectorXd& nodeFeatures2, 
			int nodeLabel1, int nodeLabel2, int& classIndex);
	};

	/*!
	 * @brief Standard CRF features with confidence unary and co-occurence counts pairwise potentials.
	 */
	class StandardConfPairwiseCountsFeatures : public StandardConfFeatures
	{
	public:
		StandardConfPairwiseCountsFeatures();
		~StandardConfPairwiseCountsFeatures();

		virtual RankFeatures computeFeatures(ImgFeatures& X, ImgLabeling& Y, set<int> action);
		virtual int featureSize(ImgFeatures& X, ImgLabeling& Y, set<int> action);

	protected:
		virtual VectorXd computePairwiseTerm(ImgFeatures& X, ImgLabeling& Y);
		virtual VectorXd computePairwiseFeatures(VectorXd& nodeFeatures1, VectorXd& nodeFeatures2, 
			int nodeLabel1, int nodeLabel2, int& classIndex);
	};

	/*!
	 * @brief Dense CRF features with unary and pairwise potentials.
	 */
	class DenseCRFFeatures : public IFeatureFunction
	{
	public:
		DenseCRFFeatures();
		~DenseCRFFeatures();

		virtual RankFeatures computeFeatures(ImgFeatures& X, ImgLabeling& Y, set<int> action);
		virtual int featureSize(ImgFeatures& X, ImgLabeling& Y, set<int> action);

	protected:
		virtual VectorXd computeUnaryTerm(ImgFeatures& X, ImgLabeling& Y);
		virtual VectorXd computePairwiseTerm(ImgFeatures& X, ImgLabeling& Y);
		virtual VectorXd computePairwiseFeatures(VectorXd& nodeFeatures1, VectorXd& nodeFeatures2, 
			double nodeLocationX1, double nodeLocationY1, double nodeLocationX2, double nodeLocationY2, 
			int nodeLabel1, int nodeLabel2, int& classIndex);
	};

	/*!
	 * @brief Interface for standard CRF features plus a global potential.
	 */
	class IGlobalFeatures : public StandardFeatures
	{
	protected:
		/*!
		 * Codebook for bag-of-words.
		 */
		struct CodeBook
		{
			/*!
			 * Codebook data
			 * rows = codewords, cols = histogram features
			 */
			MatrixXd data;
		};

		/*!
		 * Dictionary for global potential.
		 */
		CodeBook dictionary;

	public:
		virtual RankFeatures computeFeatures(ImgFeatures& X, ImgLabeling& Y, set<int> action)=0;

	protected:
		/*!
		 * @brief Compute global term.
		 */
		virtual VectorXd computeGlobalTerm(ImgFeatures& X, ImgLabeling& Y)=0;

		void loadDictionary(string fileName);
	};

	/*!
	 * @brief Standard CRF features with a sum global potential.
	 */
	class SumGlobalFeatures : public IGlobalFeatures
	{
	public:
		/*!
		 * @brief Construct without loading dictionary.
		 */
		SumGlobalFeatures();

		/*!
		 * @brief Construct and load dictionary.
		 */
		SumGlobalFeatures(string fileName);
		
		~SumGlobalFeatures();

		virtual RankFeatures computeFeatures(ImgFeatures& X, ImgLabeling& Y, set<int> action);
		virtual int featureSize(ImgFeatures& X, ImgLabeling& Y, set<int> action);

	protected:
		virtual VectorXd computeGlobalTerm(ImgFeatures& X, ImgLabeling& Y);
	};

	/*!
	 * @brief Standard CRF features with a max global potential.
	 */
	class MaxGlobalFeatures : public IGlobalFeatures
	{
	public:
		/*!
		 * @brief Construct without loading dictionary.
		 */
		MaxGlobalFeatures();

		/*!
		 * @brief Construct and load dictionary.
		 */
		MaxGlobalFeatures(string fileName);

		~MaxGlobalFeatures();

		virtual RankFeatures computeFeatures(ImgFeatures& X, ImgLabeling& Y, set<int> action);
		virtual int featureSize(ImgFeatures& X, ImgLabeling& Y, set<int> action);

	protected:
		virtual VectorXd computeGlobalTerm(ImgFeatures& X, ImgLabeling& Y);
	};

	/**************** Prune Feature Functions ****************/

	/*!
	 * @brief Standard prune features.
	 */
	class StandardPruneFeatures : public IFeatureFunction
	{
		static const int MUTEX_THRESHOLD;

	protected:
		map<string, int> mutex;
		bool initialized;

	public:
		StandardPruneFeatures();
		~StandardPruneFeatures();

		virtual RankFeatures computeFeatures(ImgFeatures& X, ImgLabeling& Y, set<int> action);
		virtual int featureSize(ImgFeatures& X, ImgLabeling& Y, set<int> action);

		void setMutex(map<string, int>& mutex);
		map<string, int> getMutex();

	protected:
		VectorXd computeHoleTerm(ImgFeatures& X, ImgLabeling& Y, set<int> action);
		
		VectorXd computeSpatialEntropyTerm(ImgFeatures& X, ImgLabeling& Y, set<int> action);

		VectorXd computeMutexTermManually(ImgFeatures& X, ImgLabeling& Y, set<int> action);

		VectorXd computeMutexTerm(ImgFeatures& X, ImgLabeling& Y, set<int> action);
		VectorXd computeMutexFeatures(VectorXd& nodeFeatures1, VectorXd& nodeFeatures2, 
			double nodeLocationX1, double nodeLocationY1, double nodeLocationX2, double nodeLocationY2, 
			int nodeLabel1, int nodeLabel2, int& classIndex);

	private:
		string mutexStringHelper(int class1, int class2, string config);
	};
}

#endif