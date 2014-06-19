#ifndef DATASTRUCTURES_HPP
#define DATASTRUCTURES_HPP

#include <map>
#include <set>
#include <queue>
#include <fstream>
#include "../../external/Eigen/Eigen/Dense"
#include "MyPrimitives.hpp"
#include "Settings.hpp"

using namespace Eigen;
using namespace std;

namespace HCSearch
{
	/**************** Constants ****************/

	enum CompareSearchNodeType { HEURISTIC, COST };
	enum SearchType { LL=0, HL, LC, HC, 
		LEARN_H, LEARN_C, LEARN_C_ORACLE_H,
		LEARN_PRUNE, DISCOVER_PAIRWISE };
	enum DatasetType { TEST=0, TRAIN, VALIDATION };
	enum StochasticCutMode { STATE, EDGES };

	const extern string SearchTypeStrings[];
	const extern string DatasetTypeStrings[];

	/**************** Priority Queues ****************/

	class CompareByConfidence
	{
	public:
		bool operator() (MyPrimitives::Pair<int, double>& lhs, MyPrimitives::Pair<int, double>& rhs) const;
	};

	typedef priority_queue<MyPrimitives::Pair<int, double>, vector< MyPrimitives::Pair<int, double> >, CompareByConfidence> LabelConfidencePQ;

	/**************** Graph ****************/

	/*!
	 * Set of (neighbor) nodes.
	 */
	typedef set< int > NeighborSet_t;

	/*!
	 * Adjacency List. 
	 * Mapping: node -> set of neighbor nodes
	 */
	typedef map< int, NeighborSet_t > AdjList_t;

	/*!
	 * @brief Anstract graph implementation to avoid redundancy.
	 */
	struct IGraph
	{
		/*!
		 * Adjacency list of the graph.
		 * Node -> set of neighbor nodes
		 */
		AdjList_t adjList;

		/*!
		 * Get the number of edges in the graph.
		 */
		int getNumEdges();
	};

	/*!
	 * @brief Basic graph implementation with feature data at nodes.
	 * 
	 * This data structure is publicly accessible so you can perform 
	 * direct reading and manipulations.
	 */
	struct FeatureGraph : public IGraph
	{
		/*!
		 * Node data stores features at nodes.
		 * Rows = nodes, cols = feature data
		 */
		MatrixXd nodesData;
	};

	/*!
	 * @brief Basic graph implementation with label data at nodes. 
	 * 
	 * This data structure is publicly accessible so you can perform 
	 * direct reading and manipulations.
	 */
	struct LabelGraph : public IGraph
	{
		/*!
		 * Nodes with labels.
		 */
		VectorXi nodesData;
	};

	/**************** Features and Labelings ****************/

	/*!
	 * @brief Structured input: graph with feature nodes.
	 * 
	 * This stores a graph where nodes hold feature data. 
	 * Usually the nodes correspond to pixels, superpixels or patches, 
	 * and the data at each node correspond to the features of that
	 * pixel, superpixel or patch.
	 */
	class ImgFeatures
	{
	public:
		/*!
		 * Graph with feature nodes. Public access allows "low level" manipulations.
		 */
		FeatureGraph graph;

		/*!
		 * File name of this image.
		 */
		string filename;

		/*!
		 * Segments matrix that has node IDs.
		 * Dimensions: original image height x original image width
		 * Make sure to check if they are available using ImgLabeling::segmentsAvailable.
		 */
		MatrixXi segments;

		/*!
		 * Matrix of node locations.
		 * Dimensions: number of nodes x 2 (x, y coordinates)
		 * Make sure to check if they are available using ImgLabeling::nodeLocationsAvailable.
		 */
		MatrixXd nodeLocations;

		/*!
		 * Edge weights.
		 * Indexed by (node 1, node 2)
		 */
		map< MyPrimitives::Pair<int, int>, double > edgeWeights;

		/*!
		 * Edge features.
		 * Indexed by (node 1, node 2)
		 */
		map< MyPrimitives::Pair<int, int>, VectorXd > edgeFeatures;

		bool segmentsAvailable;
		bool nodeLocationsAvailable;
		bool edgeWeightsAvailable;
		bool edgeFeaturesAvailable;

	public:
		ImgFeatures();
		~ImgFeatures();

		/*!
		 * Convenience function to get the feature dimension of nodes.
		 * @return Returns the feature dimension
		 */
		int getFeatureDim();

		/*!
		 * Convenience function to get the number of nodes.
		 * @return Returns the number of nodes
		 */
		int getNumNodes();

		/*!
		 * Function to get the number of edges.
		 * @return Returns the number of edges
		 */
		int getNumEdges();

		/*!
		 * Convenience function to get a feature component at a node.
		 * @param[in] node Node index
		 * @param[in] featIndex Index of feature component
		 * @return Returns the feature component of the node
		 */
		double getFeature(int node, int featIndex);

		/*!
		 * Get the file name that the features came from.
		 * @return Returns the file name
		 */
		string getFileName();

		/*!
		 * Get the normalized x-position of the node.
		 * @param[in] node Node index
		 * @return Returns the X position of node
		 */
		double getNodeLocationX(int node);

		/*!
		 * Get the normalized y-position of the node.
		 * @param[in] node Node index
		 * @return Returns the Y position of node
		 */
		double getNodeLocationY(int node);
	};

	/*!
	 * @brief Structured output labeling: graph with label nodes.
	 * 
	 * This stores a graph where nodes hold label data. 
	 * Usually the nodes correspond to pixels, superpixels or patches. 
	 * Each pixel, superpixel or patch has an integer label usually corresponding 
	 * to a class.
	 */
	class ImgLabeling
	{
	public:
		/*!
		 * Graph with label nodes. Public access allows "low level" manipulations.
		 */
		LabelGraph graph;

		/*!
		 * Class confidences of nodes from initial prediction. 
		 * rows = nodes, cols = label probabilities (sorted by class index)
		 * Make sure to check if they are available using ImgLabeling::confidencesAvailable.
		 */
		MatrixXd confidences;

		/*!
		 * Adjacency list of edges that survived stochastic cuts.  
		 * node -> set of neighbor nodes
		 * Make sure to check if they are available using ImgLabeling::stochasticCutsAvailable.
		 */
		map< int, set<int> > stochasticCuts;

		/*!
		 * Node weights.
		 */
		VectorXd nodeWeights;

		bool confidencesAvailable;
		bool stochasticCutsAvailable;
		bool nodeWeightsAvailable;

	public:
		ImgLabeling();
		~ImgLabeling();

		/*!
		 * Convenience function to get the number of nodes.
		 * @return Returns the number of nodes
		 */
		int getNumNodes();

		/*!
		 * Function to get the number of edges.
		 * @return Returns the number of edges
		 */
		int getNumEdges();

		/*!
		 * Convenience function to get a node's label.
		 * @param[in] node Node index
		 * @return Returns the label of the node
		 */
		int getLabel(int node);

		/*!
		 * Convenience function to get labels of the node's neighbors.
		 * @param[in] node Node index
		 * @return Returns the set of labels of neighboring nodes
		 */
		set<int> getNeighborLabels(int node);

		/*!
		 * Convenience function to get node's neighbors.
		 * @param[in] node Node index
		 * @return Returns the set of neighboring nodes (indices)
		 */
		set<int> getNeighbors(int node);

		/*!
		 * Convenience function to check if node has neighbors.
		 * @param[in] node Node index
		 * @return Returns true if there are neighbors
		 */
		bool hasNeighbors(int node);

		/*!
		 * @brief Get the labels of the top K confident labels.
		 * @param[in] node Node index
		 * @param[in] K top K confident labels to return
		 * @return Returns the set of top K confident labels for the node
		 */
		set<int> getTopConfidentLabels(int node, int K);

		vector<int> getLabelsByConfidence(int node);

		int getMostConfidentLabel(int node);

		double getConfidence(int node, int label);
	};

	/*!
	 * @brief Structured output labeling candidate: labeling and action.
	 * 
	 * This stores an ImgLabeling with a corresponding action, i.e. the set of nodes that changed.
	 */
	class ImgCandidate
	{
	public:
		/*!
		 * New labeling.
		 */
		ImgLabeling labeling;

		/*!
		 * Set of nodes that changed.
		 */
		set<int> action;
	};

	/**************** Classify/Rank Features ****************/

	/*!
	 * @brief Stores features for ranking or classification.
	 * 
	 * This is nothing more than a wrapper around a VectorXd object.
	 */
	class GenericFeatures
	{
	public:
		/*!
		 * Features data
		 */
		VectorXd data;

		/*!
		 * Default constructor does nothing.
		 */
		GenericFeatures();

		/*!
		 * Constructor to initialize features data.
		 */
		GenericFeatures(VectorXd features);
		
		~GenericFeatures();
	};

	typedef GenericFeatures RankFeatures;

	/**************** Rank Model ****************/

	/*!
	 * @brief Abstract class for model/weights for ranking.
	 * 
	 * The main purpose is to rank features using a rank model. 
	 * Abstract class is useful for extending different kinds of rankers.
	 */
	class IRankModel
	{
	protected:
		bool initialized;

	public:
		virtual ~IRankModel() {}

		/*!
		 * Use the model to rank the features.
		 * @param[in] features Features for ranking
		 * @return Returns the ranking of the feature
		 */
		virtual double rank(RankFeatures features)=0;

		/*!
		 * Use the model to rank a list of features.
		 * @param[in] featuresList List of features for ranking
		 * @return Returns the list of ranking values of the features
		 */
		virtual vector<double> rank(vector<RankFeatures> featuresList)=0;

		/*!
		 * Get the ranker type.
		 */
		virtual RankerType rankerType()=0;

		/*!
		 * Load weights data from file.
		 */
		virtual void load(string fileName)=0;

		/*!
		 * Save weights data to file.
		 */
		virtual void save(string fileName)=0;

	protected:
		/*!
		 * Manual implementation of the vector dot product. 
		 * Used to correct a mysterious problem with Eigen's implementation.
		 */
		template <class TVector>
		static double vectorDot(TVector v1, TVector v2)
		{
			double result = 0;
			for (int i = 0; i < v1.size(); i++)
			{
				if (i >= v2.size())
					break;

				result += v1(i)*v2(i);
			}
			return result;
		}
	};

	/**************** SVM-Rank Model ****************/

	/*!
	 * @brief Rank model for SVM-Rank.
	 * 
	 * Has methods to learn weights from training examples and ranking.
	 */
	class SVMRankModel : public IRankModel
	{
	private:
		/*!
		 * SVM-rank weights
		 */
		VectorXd weights;

		/*!
		 * Output stream to training file for learning
		 */
		ofstream* rankingFile;

		/*!
		 * Training file name
		 */
		string rankingFileName;

		/*!
		 * Model file name
		 */
		string modelFileName;

		/*!
		 * Cumulative QID for learning
		 */
		int qid;

		/*!
		 * True if currently used for learning
		 */
		bool learningMode;

	public:
		SVMRankModel();

		/*!
		 * Construct with SVM weights from model file.
		 */
		SVMRankModel(string fileName);
		
		virtual double rank(RankFeatures features);
		virtual RankerType rankerType();
		virtual vector<double> rank(vector<RankFeatures> featuresList);
		virtual void load(string fileName);
		virtual void save(string fileName);

		/*!
		 * Get weights.
		 */
		VectorXd getWeights();

		/*!
		 * Initialize learning.
		 */
		void startTraining(string featuresFileName);

		/*!
		 * Add training example.
		 */
		void addTrainingExample(RankFeatures betterFeature, RankFeatures worseFeature);

		/*!
		 * Add training examples.
		 */
		void addTrainingExamples(vector< RankFeatures >& betterSet, vector< RankFeatures >& worseSet);

		/*!
		 * End learning.
		 *
		 * Calls SVM Rank program to train on examples and produce model.
		 */
		void finishTraining(string modelFileName, SearchType searchType);

		/*!
		 * Cancel learning. Closes training file.
		 */
		void cancelTraining();

	private:
		/*!
		 * Load weights from file. 
		 * File format is the SVM-Rank model file 
		 * (weights are on the 12th line).
		 */
		static VectorXd parseModelFile(string fileName);

		/*!
		 * Convert vector into SVM-Rank line.
		 */
		static string vector2svmrank(RankFeatures features, int target, int qid);

		/*!
		 * Write weights to file.
		 * 
		 * File format:
		 *     ...
		 *	   line 12: 1:val 2:val ...
		 */
		static void writeModelFile(string fileName, const VectorXd& weights);

		/*!
		 * Merge SVM-Rank feature files when using MPI.
		 */
		static int mergeRankingFiles(string fileNameBase, int numProcesses, int totalMasterQID);
	};

	/**************** Vowpal Wabbit Model ****************/

	/*!
	 * @brief Rank model for Vowpal Wabbit.
	 * 
	 * Has methods to learn weights from training examples and ranking.
	 */
	class VWRankModel : public IRankModel
	{
	private:
		/*!
		 * Rank weights
		 */
		VectorXd weights;

		/*!
		 * Output stream to training file for learning
		 */
		ofstream* rankingFile;

		/*!
		 * Training file name
		 */
		string rankingFileName;

		/*!
		 * Model file name
		 */
		string modelFileName;

		/*!
		 * True if currently used for learning
		 */
		bool learningMode;

	public:
		VWRankModel();

		/*!
		 * Construct with weights from model file.
		 */
		VWRankModel(string fileName);
		
		virtual double rank(RankFeatures features);
		virtual RankerType rankerType();
		virtual vector<double> rank(vector<RankFeatures> featuresList);
		virtual void load(string fileName);
		virtual void save(string fileName);

		/*!
		 * Get weights.
		 */
		VectorXd getWeights();

		/*!
		 * Initialize learning.
		 */
		void startTraining(string featuresFileName);

		/*!
		 * Add training example.
		 */
		void addTrainingExample(RankFeatures better, RankFeatures worse, double betterLoss, double worstLoss);

		/*!
		 * Add training examples.
		 */
		void addTrainingExamples(vector< RankFeatures >& betterSet, vector< RankFeatures >& worseSet, vector< double >& betterLosses, vector< double >& worstLosses);

		/*!
		 * End learning.
		 *
		 * Calls SVM Rank program to train on examples and produce model.
		 */
		void finishTraining(string modelFileName, SearchType searchType);

		/*!
		 * Cancel learning. Closes training file.
		 */
		void cancelTraining();

	private:
		/*!
		 * Load weights from file. 
		 * File format is the SVM-Rank model file 
		 * (weights are on the 12th line).
		 */
		static VectorXd parseModelFile(string fileName);

		/*!
		 * Convert vector difference to VW format line.
		 */
		static string vector2vwformat(RankFeatures bestfeature, RankFeatures worstfeature, double loss);

		/*!
		 * Write weights to file.
		 * 
		 * File format:
		 *     ...
		 *	   line 12: 1:val 2:val ...
		 */
		static void writeModelFile(string fileName, const VectorXd& weights);

		/*!
		 * Merge feature files when using MPI.
		 */
		static void mergeRankingFiles(string fileNameBase, int numProcesses);
	};
}

#endif